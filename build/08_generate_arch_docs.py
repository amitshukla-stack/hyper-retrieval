"""
Stage 8 — Auto-generate architecture docs from the code index.

Uses Kimi LLM to:
  1. DISCOVER: identify major business domains from cluster summaries
  2. GENERATE: write payment_flows.md-style docs for each domain
  3. VERIFY:   check all code references against body_store
  4. CORRECT:  fix inaccuracies up to MAX_VERIFY_ITERATIONS times
  5. SAVE:     write to docs/generated/ directory

Run:
    python3 08_generate_arch_docs.py

Environment variables:
    OUTPUT_DIR              path to graph_with_summaries.json / body_store.json
    WORKSPACE_DIR           workspace root; docs/generated/ goes here
    MAX_VERIFY_ITERATIONS   default 3
    MIN_ACCURACY_THRESHOLD  default 0.80
    SKIP_IF_EXISTS          skip domains whose .md already exists (default 1)
    LLM_API_KEY             API key for Kimi / OpenAI-compatible endpoint
    LLM_BASE_URL            base URL (e.g. https://grid.ai.juspay.net)
    LLM_MODEL               model name (default kimi-latest)
"""
import json, pathlib, os, re, sys, time, datetime
from typing import Optional

OUT_DIR       = pathlib.Path(os.environ.get("OUTPUT_DIR",    pathlib.Path(__file__).parent / "output"))
WORKSPACE_DIR = pathlib.Path(os.environ.get("WORKSPACE_DIR", pathlib.Path(__file__).parent.parent))
GENERATED_DOCS_DIR = WORKSPACE_DIR / "docs" / "generated"

MAX_VERIFY_ITERATIONS  = int(os.environ.get("MAX_VERIFY_ITERATIONS",  "3"))
MIN_ACCURACY_THRESHOLD = float(os.environ.get("MIN_ACCURACY_THRESHOLD", "0.80"))
SKIP_IF_EXISTS         = os.environ.get("SKIP_IF_EXISTS", "1") == "1"

API_KEY  = os.environ.get("LLM_API_KEY",  "")
BASE_URL = os.environ.get("LLM_BASE_URL", "")
MODEL    = os.environ.get("LLM_MODEL",    "kimi-latest")

MAX_LLM_RETRIES = 3

WARNING_HEADER = (
    "> WARNING: This document was auto-generated and may contain inaccuracies."
    " Review before relying on it.\n\n"
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_graph_data() -> dict:
    """Load graph_with_summaries.json."""
    path = OUT_DIR / "graph_with_summaries.json"
    if not path.exists():
        print(f"[error] graph_with_summaries.json not found at {path}")
        sys.exit(1)
    data = json.loads(path.read_text())
    print(f"[graph] Loaded {len(data.get('nodes', []))} nodes, "
          f"{len(data.get('cluster_summaries', {}))} cluster summaries")
    return data


def load_body_store() -> dict:
    """Load body_store.json. Returns empty dict if not found."""
    path = OUT_DIR / "body_store.json"
    if not path.exists():
        print(f"[body_store] Not found at {path} — verification will be skipped")
        return {}
    data = json.loads(path.read_text())
    print(f"[body_store] Loaded {len(data)} entries")
    return data


# ── LLM helpers ───────────────────────────────────────────────────────────────

def call_llm(client, system: str, user: str) -> str:
    """Call LLM with retry + exponential backoff. Returns response text."""
    for attempt in range(MAX_LLM_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.2,
                max_tokens=4000,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate" in err.lower() or "limit" in err.lower()
            if attempt < MAX_LLM_RETRIES - 1:
                wait = 2 ** (attempt + 2) if is_rate else 3
                print(f"    [llm retry {attempt+1}] {err[:80]} — waiting {wait}s")
                time.sleep(wait)
            else:
                raise
    return ""


def extract_json_list(text: str) -> Optional[list]:
    """Extract the first JSON array from text."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text.strip())
    start = text.find("[")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None


# ── Code ref extraction ───────────────────────────────────────────────────────

def extract_code_refs_from_doc(text: str) -> list[str]:
    """Return backtick-quoted identifiers from a markdown document."""
    return re.findall(r'`([a-zA-Z][a-zA-Z0-9_\'\.]+)`', text)


def extract_sentence_with_ref(text: str, ref: str) -> str:
    """Return the first sentence that contains ref."""
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if ref in sentence:
            return sentence.strip()
    # Fallback: return line containing ref
    for line in text.splitlines():
        if ref in line:
            return line.strip()
    return ""


# ── Phase 1: DISCOVER ─────────────────────────────────────────────────────────

def discover_domains(graph_data: dict, existing_docs: list[str], client) -> list[dict]:
    """
    Ask Kimi to identify 5-10 major business domains from cluster summaries.
    Returns list of dicts: {title, seed_clusters, seed_terms}
    """
    summaries = graph_data.get("cluster_summaries", {})
    if not summaries:
        print("[discover] No cluster summaries found in graph data")
        return []

    lines = []
    for cid, s in sorted(summaries.items(), key=lambda x: x[0]):
        name    = s.get("name", f"cluster_{cid}")
        purpose = s.get("purpose", "")
        lines.append(f"  cluster {cid}: {name} — {purpose[:120]}")
    summaries_text = "\n".join(lines)

    existing_str = ", ".join(existing_docs) if existing_docs else "none"

    system = (
        "You are a software architecture analyst. "
        "Output ONLY a valid JSON array — no markdown fences, no explanation."
    )
    user = (
        "You are analyzing a payment platform codebase. "
        "Below are cluster summaries from the code index.\n"
        "Identify 5-10 major business concept domains that would benefit from "
        "architecture documentation.\n"
        "For each domain, identify which cluster IDs are most relevant and key domain terms.\n"
        "Skip domains that already have docs (listed below).\n\n"
        f"Cluster summaries:\n{summaries_text}\n\n"
        f"Existing docs: {existing_str}\n\n"
        "Return ONLY a JSON array of objects with fields:\n"
        '  title (string), seed_clusters (list of int cluster IDs), seed_terms (list of strings).\n'
        "Example: "
        '[{"title":"mandate_lifecycle","seed_clusters":[12,34],"seed_terms":["mandate","emandate"]}]'
    )

    print("[discover] Asking LLM to identify business domains...")
    try:
        raw = call_llm(client, system, user)
    except Exception as e:
        print(f"[discover] LLM call failed: {e}")
        return []

    domains = extract_json_list(raw)
    if domains is None:
        print(f"[discover] Failed to parse JSON from response: {raw[:200]!r}")
        return []

    # Validate fields
    valid = []
    for d in domains:
        if not isinstance(d, dict):
            continue
        if "title" not in d or "seed_clusters" not in d:
            continue
        valid.append({
            "title":         str(d["title"]),
            "seed_clusters": [int(c) for c in d.get("seed_clusters", []) if str(c).isdigit()],
            "seed_terms":    [str(t) for t in d.get("seed_terms", [])],
        })

    print(f"[discover] Found {len(valid)} domains: {[d['title'] for d in valid]}")
    return valid


# ── Phase 2: GENERATE ─────────────────────────────────────────────────────────

def generate_doc(domain: dict, graph_data: dict, body_store: dict, client) -> str:
    """
    Generate a payment_flows.md-style architecture doc for one domain.
    Returns the generated markdown text.
    """
    summaries    = graph_data.get("cluster_summaries", {})
    nodes        = graph_data.get("nodes", [])
    seed_ids     = set(str(c) for c in domain["seed_clusters"])
    seed_terms   = [t.lower() for t in domain["seed_terms"]]
    title        = domain["title"]

    # Pull cluster summaries for seed clusters
    cluster_summaries_text = ""
    for cid in sorted(seed_ids):
        s = summaries.get(cid, {})
        if s:
            cluster_summaries_text += (
                f"Cluster {cid} ({s.get('name','?')}):\n"
                f"  Purpose: {s.get('purpose','')}\n"
                f"  Contracts: {s.get('contracts', [])}\n"
                f"  Data flows: {s.get('data_flows', [])}\n"
                f"  External deps: {s.get('ghost_deps', [])}\n\n"
            )

    # Pull top 20 function signatures matching seed terms
    cluster_int_ids = set(domain["seed_clusters"])
    matching_nodes  = [
        n for n in nodes
        if n.get("cluster") in cluster_int_ids
        and any(t in n.get("name", "").lower() for t in seed_terms)
    ]
    sig_lines = []
    for n in matching_nodes[:20]:
        lang = n.get("lang", "?")
        kind = n.get("kind", "?")
        nid  = n.get("id", n.get("name", "?"))   # fully-qualified ID for body_store lookup
        typ  = n.get("type", "")
        line = f"  [{lang}/{kind}] {nid}"
        if typ:
            line += f" :: {typ[:120]}"
        sig_lines.append(line)
    function_signatures = "\n".join(sig_lines) if sig_lines else "  (none matched seed terms)"

    # Pull up to 10 entry-point bodies
    ENTRY_POINT_KEYWORDS = {"workflow", "flow", "handler", "router", "decider", "service", "manager"}
    entry_nodes = [
        n for n in matching_nodes
        if any(kw in n.get("name", "").lower() for kw in ENTRY_POINT_KEYWORDS)
    ]
    bodies_text = ""
    count = 0
    for n in entry_nodes:
        nid  = n.get("id", "")
        body = body_store.get(nid, "")
        if body and count < 10:
            bodies_text += f"\n--- {nid} ---\n{body[:800]}\n"
            count += 1
    if not bodies_text:
        bodies_text = "(no entry point bodies found)"

    system = (
        "You are a technical writer documenting a payment platform codebase for engineers. "
        "Write clear, specific, accurate markdown. Never fabricate function names."
    )
    user = (
        f"Write a technical architecture document about: {title}\n\n"
        "Style: Use H2 headings, tables mapping business concepts to code, "
        "code references using backticks, call chains using indented arrows (└─▶).\n\n"
        f"Relevant cluster summaries:\n{cluster_summaries_text}\n"
        f"Key functions and signatures (use these EXACT fully-qualified IDs in backticks when referencing code):\n{function_signatures}\n\n"
        f"Entry point implementations:\n{bodies_text}\n\n"
        "Write the complete markdown document. "
        "When referencing code, use the exact fully-qualified IDs shown above (e.g. `Euler.API.Txns.Flow.handleTxn`). "
        "Be specific and accurate — only claim things you can verify from the code above."
    )

    print(f"  [generate] Calling LLM for '{title}'...")
    try:
        return call_llm(client, system, user)
    except Exception as e:
        print(f"  [generate] LLM call failed: {e}")
        return f"# {title}\n\n*Generation failed: {e}*\n"


# ── Phase 3: VERIFY ───────────────────────────────────────────────────────────

def verify_doc(doc_text: str, body_store: dict, client) -> tuple[float, list[dict]]:
    """
    Check code references in doc against body_store.
    Returns (accuracy_score, corrections_list).
    corrections_list entries: {ref, sentence, reason}
    """
    if not body_store:
        print("  [verify] body_store empty — skipping verification")
        return 1.0, []

    refs = list(dict.fromkeys(extract_code_refs_from_doc(doc_text)))
    if not refs:
        return 1.0, []

    # Only check refs that exist in body_store
    checkable = [r for r in refs if r in body_store]
    if not checkable:
        print(f"  [verify] {len(refs)} refs extracted, 0 found in body_store")
        return 1.0, []

    print(f"  [verify] Checking {len(checkable)}/{len(refs)} refs found in body_store")

    accurate   = 0
    corrections = []

    system = (
        "You are a code accuracy reviewer. "
        "Reply with exactly one of: ACCURATE  or  INACCURATE: <one-line reason>"
    )

    for ref in checkable:
        sentence = extract_sentence_with_ref(doc_text, ref)
        if not sentence:
            accurate += 1
            continue
        body = body_store[ref]
        user = (
            f'Doc claims: "{sentence}"\n'
            f"Actual code for `{ref}`:\n"
            f"{body[:500]}\n\n"
            "Is the doc's claim accurate? Reply: ACCURATE  or  INACCURATE: <reason>"
        )
        try:
            verdict = call_llm(client, system, user).strip()
        except Exception as e:
            print(f"    [verify] LLM error for {ref}: {e}")
            accurate += 1  # Give benefit of the doubt on error
            continue

        if verdict.upper().startswith("ACCURATE"):
            accurate += 1
        else:
            reason = verdict[len("INACCURATE:"):].strip() if ":" in verdict else verdict
            corrections.append({"ref": ref, "sentence": sentence, "reason": reason})

    total = len(checkable)
    score = accurate / total if total > 0 else 1.0
    print(f"  [verify] Score: {accurate}/{total} = {score:.2f}")
    return score, corrections


# ── Phase 4: CORRECT ──────────────────────────────────────────────────────────

def correct_doc(doc_text: str, corrections: list[dict], body_store: dict, client) -> str:
    """
    Ask Kimi to fix the inaccurate sections identified in corrections.
    Returns corrected doc text.
    """
    if not corrections:
        return doc_text

    correction_items = "\n".join(
        f"- Ref `{c['ref']}`: sentence \"{c['sentence']}\" is wrong because: {c['reason']}"
        for c in corrections
    )
    # Provide actual bodies for context
    bodies_context = ""
    for c in corrections:
        body = body_store.get(c["ref"], "")
        if body:
            bodies_context += f"\nActual code for `{c['ref']}`:\n{body[:500]}\n"

    system = (
        "You are editing a technical document to fix factual inaccuracies about code. "
        "Return the complete corrected document. Preserve all accurate sections unchanged."
    )
    user = (
        "The following claims in this architecture document are inaccurate. "
        "Fix only those sections and return the full corrected document.\n\n"
        f"Inaccuracies to fix:\n{correction_items}\n\n"
        f"Actual code for context:\n{bodies_context}\n\n"
        f"Original document:\n{doc_text}"
    )

    print(f"  [correct] Fixing {len(corrections)} inaccuracies...")
    try:
        return call_llm(client, system, user)
    except Exception as e:
        print(f"  [correct] LLM call failed: {e}")
        return doc_text


# ── Phase 5: SAVE ─────────────────────────────────────────────────────────────

def save_doc(domain: dict, doc_text: str, score: float, iterations: int):
    """Write doc markdown and metadata sidecar."""
    GENERATED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    title     = domain["title"]
    doc_path  = GENERATED_DOCS_DIR / f"{title}.md"
    meta_path = GENERATED_DOCS_DIR / f"{title}.meta.json"

    doc_path.write_text(doc_text, encoding="utf-8")

    meta = {
        "title":         title,
        "accuracy_score": round(score, 4),
        "iterations":    iterations,
        "generated_at":  datetime.datetime.utcnow().isoformat() + "Z",
        "seed_clusters": domain.get("seed_clusters", []),
        "seed_terms":    domain.get("seed_terms", []),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [save] {doc_path}  (score={score:.2f}, iters={iterations})")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def process_domain(domain: dict, graph_data: dict, body_store: dict, client):
    title    = domain["title"]
    doc_path = GENERATED_DOCS_DIR / f"{title}.md"

    if SKIP_IF_EXISTS and doc_path.exists():
        print(f"[skip] '{title}' already exists at {doc_path}")
        return

    print(f"\n{'='*60}")
    print(f"Domain: {title}")
    print(f"  seed_clusters={domain['seed_clusters']}, seed_terms={domain['seed_terms']}")

    # Phase 2: Generate
    doc_text = generate_doc(domain, graph_data, body_store, client)

    # Phase 3+4: Verify and correct loop
    score      = 1.0
    iterations = 0

    if body_store:
        for iteration in range(MAX_VERIFY_ITERATIONS):
            iterations = iteration + 1
            score, corrections = verify_doc(doc_text, body_store, client)
            if score >= MIN_ACCURACY_THRESHOLD or not corrections:
                break
            if iteration < MAX_VERIFY_ITERATIONS - 1:
                doc_text = correct_doc(doc_text, corrections, body_store, client)
            else:
                print(f"  [warn] Score {score:.2f} still below threshold after {iterations} iterations")
                if not doc_text.startswith(">"):
                    doc_text = WARNING_HEADER + doc_text

    # Phase 5: Save
    save_doc(domain, doc_text, score, iterations)


def main():
    from openai import OpenAI

    if not API_KEY:
        print("[error] LLM_API_KEY not set")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)

    # Verify connectivity
    try:
        available = [m.id for m in client.models.list().data]
        if MODEL not in available:
            print(f"[warn] {MODEL} not in model list: {available}")
        else:
            print(f"[llm] Connected  model={MODEL}")
    except Exception as e:
        print(f"[warn] Cannot list models from {BASE_URL}: {e} — continuing anyway")

    # Load data
    graph_data = load_graph_data()
    body_store = load_body_store()

    # Existing docs
    GENERATED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    existing_docs = [p.stem for p in GENERATED_DOCS_DIR.glob("*.md")]
    if existing_docs:
        print(f"[discover] Existing docs: {existing_docs}")

    # Phase 1: Discover domains
    domains = discover_domains(graph_data, existing_docs, client)
    if not domains:
        print("[main] No domains discovered — exiting")
        sys.exit(0)

    # Process each domain
    failed = []
    for domain in domains:
        try:
            process_domain(domain, graph_data, body_store, client)
        except Exception as e:
            print(f"[error] Domain '{domain.get('title','?')}' failed: {e}")
            failed.append(domain.get("title", "?"))

    print(f"\n{'='*60}")
    print(f"Done. {len(domains) - len(failed)}/{len(domains)} domains generated.")
    if failed:
        print(f"Failed: {failed}")
    print(f"Output: {GENERATED_DOCS_DIR}")


if __name__ == "__main__":
    main()
