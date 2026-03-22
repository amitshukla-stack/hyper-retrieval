"""
Stage 4 - Summarize each cluster via an LLM (any OpenAI-compatible endpoint).

Safeguards:
  - Exponential backoff on rate limits (429) up to 5 retries
  - 2s delay between cluster calls
  - Resumes from partial progress if output already exists
  - Saves after every cluster (crash-safe)
  - Stratified node sampling for huge clusters (spread across modules)
  - Robust JSON extraction (handles reasoning models that think before answering)
  - Per-cluster seed for stratified sampling (avoids bias from global shuffle)
  - Cross-cluster external callers included in prompt (from call_graph.json)

Input:  output/graph_clustered.json
Output: output/graph_with_summaries.json
        demo_artifact/graph_with_summaries.json
"""
import json, pathlib, os, sys, time, random, re
from collections import defaultdict

OUT_DIR      = pathlib.Path(os.environ.get("OUTPUT_DIR",    pathlib.Path(__file__).parent / "output"))
ARTIFACT_DIR = pathlib.Path(os.environ.get("ARTIFACT_DIR", pathlib.Path(__file__).parent / "demo_artifact"))
ARTIFACT_DIR.mkdir(exist_ok=True)

API_KEY  = os.environ.get("LLM_API_KEY",  "")
BASE_URL = os.environ.get("LLM_BASE_URL", "")
MODEL    = os.environ.get("LLM_MODEL",    "reasoning-large-model")

MAX_NODES_PER_CLUSTER = int(os.environ.get("MAX_NODES_PER_CLUSTER", "80"))
MIN_CLUSTER_SIZE      = int(os.environ.get("MIN_CLUSTER_SIZE", "100"))
CALL_DELAY            = float(os.environ.get("CALL_DELAY", "2.0"))
MAX_RETRIES           = 5

SYSTEM_PROMPT = (
    "You are an expert software architect. "
    "You output ONLY raw valid JSON — no markdown fences, no reasoning text, "
    "no explanation before or after. Your entire response must be a single JSON object "
    "that can be passed directly to json.loads()."
)

# ── Load optional call graph ──────────────────────────────────────────────────

CALL_GRAPH_PATH = OUT_DIR / "call_graph.json"
call_graph = {}
if CALL_GRAPH_PATH.exists():
    try:
        call_graph = json.loads(CALL_GRAPH_PATH.read_text())
        print(f"[call_graph] Loaded {len(call_graph)} entries from {CALL_GRAPH_PATH}")
    except Exception as e:
        print(f"[call_graph] Failed to load: {e} — skipping cross-cluster callers")


# ── Resume filter ─────────────────────────────────────────────────────────────

def _is_bad_summary(s: dict) -> bool:
    """Return True if summary is an error/parse-fail placeholder, not a real LLM result."""
    name    = s.get("name", "")
    purpose = s.get("purpose", "")
    if not name:
        return True
    # Error placeholders always have a numeric cluster ID as name: "cluster_42"
    if re.match(r'^cluster_\d+$', name):
        return True
    # Explicit error/parse-fail markers set by call_with_backoff
    if any(marker in purpose for marker in [
        "Error:", "Parse failed", "No JSON", "Empty response", "Max retries"
    ]):
        return True
    return False


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text):
    """
    Extract the first complete JSON object from text.
    Handles reasoning models that prepend thinking before the JSON.
    """
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    # Find outermost { ... }
    start = text.find("{")
    if start == -1:
        return None, text
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None, text
    return text[start:end+1], text


# ── Stratified sampling ───────────────────────────────────────────────────────

def stratified_sample(nodes, n, cluster_id=0):
    """
    Sample up to n nodes spread evenly across modules.
    Uses a per-cluster seed so different clusters get different orderings.
    """
    if len(nodes) <= n:
        return nodes
    by_module = defaultdict(list)
    for node in nodes:
        by_module[node.get("module", "unknown")].append(node)
    modules = list(by_module.keys())
    # Per-cluster seed prevents every cluster from seeing the same module ordering
    random.seed(42 + hash(str(cluster_id)) % 10000)
    random.shuffle(modules)
    sampled, per_module = [], max(1, n // len(modules))
    for mod in modules:
        sampled.extend(by_module[mod][:per_module])
        if len(sampled) >= n:
            break
    return sampled[:n]


# ── Prompt building ───────────────────────────────────────────────────────────

def build_cluster_prompt(cluster_id, nodes, cluster_meta):
    services   = list(cluster_meta.get("services", {}).keys())
    langs      = list(cluster_meta.get("langs", {}).keys())
    ghost_deps = cluster_meta.get("ghost_deps", [])
    n_modules  = len(set(n.get("module") for n in nodes))

    cid_int = int(cluster_id) if str(cluster_id).isdigit() else 0
    sample = stratified_sample(nodes, MAX_NODES_PER_CLUSTER, cluster_id=cid_int)
    lines  = []
    for n in sample:
        lang    = n.get("lang", "?")
        kind    = n.get("kind", "?")
        name    = n.get("name", "?")
        typ     = n.get("type", "")
        ghosts  = ", ".join(n.get("ghost_deps", []))
        history = "; ".join(c["msg"] for c in n.get("commit_history", [])[:2])
        line    = f"  [{lang}/{kind}] {name}"
        if typ:     line += f" :: {typ[:100]}"
        if ghosts:  line += f"  [ext: {ghosts}]"
        if history: line += f"  [git: {history[:80]}]"
        lines.append(line)

    prompt = (
        f"Analyse this cluster: {len(nodes)} symbols across {n_modules} modules\n"
        f"Services: {', '.join(services)}  |  Languages: {', '.join(langs)}\n"
    )
    if ghost_deps:
        prompt += f"Known external deps: {', '.join(ghost_deps)}\n"

    # Cross-cluster external callers (from call_graph.json)
    if call_graph:
        cluster_node_ids = set(n["id"] for n in nodes)
        external_callers = set()
        for nid, info in call_graph.items():
            if nid not in cluster_node_ids:
                for callee in info.get("callees", []):
                    if callee in cluster_node_ids or any(callee in nid2 for nid2 in cluster_node_ids):
                        mod = ".".join(nid.split(".")[:3])
                        external_callers.add(mod)
                        break
        if external_callers:
            prompt += f"External modules calling into this cluster: {', '.join(list(external_callers)[:5])}\n"

    prompt += (
        f"\nRepresentative sample ({len(sample)} of {len(nodes)} symbols, stratified across modules):\n"
        + "\n".join(lines)
        + "\n\nReturn this JSON object (max 3 items per array, keep values under 15 words each):\n"
        '{\n'
        '  "name": "<3-5 word subsystem name>",\n'
        '  "purpose": "<1-2 sentences on what this cluster does>",\n'
        '  "contracts": ["<guarantees or expectations>"],\n'
        '  "data_flows": ["<data entering and exiting>"],\n'
        '  "ghost_deps": ["<external systems: postgres, redis, kafka, etc>"],\n'
        '  "risk_flags": ["<fragile, undocumented, or concerning patterns>"],\n'
        '  "cross_service_links": ["<other services this likely interacts with>"]\n'
        '}'
    )
    return prompt


# ── LLM call with backoff ─────────────────────────────────────────────────────

def call_with_backoff(client, prompt, cluster_id):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content or ""
            if not raw.strip():
                print(f"\n    [warn] Empty response")
                return {"name": f"cluster_{cluster_id}", "purpose": "Empty response"}

            json_str, original = extract_json(raw)
            if json_str is None:
                print(f"\n    [warn] No JSON found | raw[:120]: {original[:120]!r}")
                return {"name": f"cluster_{cluster_id}", "purpose": "No JSON in response"}

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            print(f"\n    [warn] JSON decode failed: {e}")
            return {"name": f"cluster_{cluster_id}", "purpose": "Parse failed", "error": str(e)}

        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate" in err.lower() or "limit" in err.lower()
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 2)
                print(f"\n    [rate limit] waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            print(f"\n    [error] {e}")
            return {"name": f"cluster_{cluster_id}", "purpose": f"Error: {e}"}

    return {"name": f"cluster_{cluster_id}", "purpose": "Max retries exceeded"}


# ── Progress saving ───────────────────────────────────────────────────────────

def save_progress(data, summaries, out_path):
    data["cluster_summaries"] = summaries
    out_path.write_text(json.dumps(data, indent=2))
    (ARTIFACT_DIR / "graph_with_summaries.json").write_text(json.dumps(data, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    try:
        available = [m.id for m in client.models.list().data]
        if MODEL not in available:
            print(f"Warning: {MODEL} not in model list: {available}")
        else:
            print(f"Connected  model={MODEL}")
    except Exception as e:
        print(f"Cannot connect to {BASE_URL}: {e}")
        sys.exit(1)

    data       = json.loads((OUT_DIR / "graph_clustered.json").read_text())
    nodes      = data["nodes"]
    clusters   = data["clusters"]
    node_index = {n["id"]: n for n in nodes}
    out_path   = OUT_DIR / "graph_with_summaries.json"

    # Resume — discard only genuine error/placeholder summaries
    cluster_summaries = {}
    if out_path.exists():
        try:
            existing  = json.loads(out_path.read_text())
            all_saved = existing.get("cluster_summaries", {})
            cluster_summaries = {
                cid: s for cid, s in all_saved.items()
                if not _is_bad_summary(s)
            }
            skipped = len(all_saved) - len(cluster_summaries)
            if cluster_summaries:
                print(f"Resuming - {len(cluster_summaries)} good summaries kept"
                      + (f", {skipped} bad ones discarded" if skipped else ""))
        except Exception:
            pass

    real_clusters = {
        cid: meta for cid, meta in clusters.items()
        if cid != "-1" and len(meta["nodes"]) >= MIN_CLUSTER_SIZE
    }
    pending = {
        cid: meta for cid, meta in real_clusters.items()
        if cid not in cluster_summaries
    }

    print(f"Clusters: {len(real_clusters)} total  {len(cluster_summaries)} done  {len(pending)} pending")
    print(f"Settings: MIN_CLUSTER_SIZE={MIN_CLUSTER_SIZE}  sample={MAX_NODES_PER_CLUSTER}  delay={CALL_DELAY}s\n")

    for i, (cid, meta) in enumerate(sorted(pending.items(), key=lambda x: -len(x[1]["nodes"]))):
        cluster_nodes = [node_index[nid] for nid in meta["nodes"] if nid in node_index]
        n_modules = len(set(n.get("module") for n in cluster_nodes))
        print(f"  [{i+1}/{len(pending)}] cluster_{cid}: {len(cluster_nodes)} nodes / {n_modules} modules "
              f"({meta.get('dominant_service','?')})", end=" ", flush=True)

        prompt  = build_cluster_prompt(cid, cluster_nodes, meta)
        summary = call_with_backoff(client, prompt, cid)
        meta["summary"]        = summary
        cluster_summaries[cid] = summary
        print(f"-> {summary.get('name', '?')}")

        save_progress(data, cluster_summaries, out_path)

        if i < len(pending) - 1:
            time.sleep(CALL_DELAY)

    # Propagate names back to nodes
    for n in nodes:
        cid = str(n.get("cluster", -1))
        if cid in cluster_summaries:
            n["cluster_name"]    = cluster_summaries[cid].get("name", "")
            n["cluster_purpose"] = cluster_summaries[cid].get("purpose", "")

    save_progress(data, cluster_summaries, out_path)

    print(f"\nDone: {len(cluster_summaries)} cluster summaries -> {out_path}")
    print("\nClusters:")
    for cid, s in sorted(cluster_summaries.items(), key=lambda x: x[0]):
        print(f"  cluster_{cid:>4}: {s.get('name','?'):35s} - {s.get('purpose','?')[:60]}")


if __name__ == "__main__":
    main()
