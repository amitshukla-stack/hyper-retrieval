"""
Build requirement-graph index for search_requirements MCP tool (RFC 015).

For each active module, extracts a functional requirement description via LLM,
then embeds it for semantic search. Stores results to:
  artifacts/requirements.json     — raw requirement text per module
  artifacts/requirements.lance    — vector index for retrieval

Cost control: only processes modules with body_store entries AND activity_score > ACTIVITY_THRESHOLD.
Default ~30K modules @ ~230 tokens each = ~7M tokens total.

Usage:
  python3 build/11_build_requirements.py [--dry-run] [--limit N]
"""

import argparse, json, os, re, sys, time, requests
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path(os.environ.get(
    "ARTIFACT_DIR",
    os.path.expanduser("~/projects/workspaces/juspay/artifacts")
))
EMBED_SERVER_URL = os.environ.get("EMBED_SERVER_URL", "http://localhost:8001")
ACTIVITY_THRESHOLD = float(os.environ.get("ACTIVITY_THRESHOLD", "0.01"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))
RESUME = os.environ.get("RESUME", "1") == "1"

# LLM config
def _load_env():
    env_file = Path.home() / "carlsbert" / ".env"
    if not env_file.exists(): return
    for line in open(env_file):
        line = line.strip()
        if not line or line.startswith("#"): continue
        sep = "=" if "=" in line else ":" if ":" in line else None
        if not sep: continue
        k, _, v = line.partition(sep)
        k = k.strip().lower(); v = v.strip().strip('"').strip("'")
        if k == "api_key": os.environ.setdefault("KIMI_API_KEY", v)
        elif k == "base_url":
            b = v.rstrip("/")
            if not b.endswith("/v1"): b += "/v1"
            os.environ.setdefault("KIMI_BASE_URL", b)

_load_env()
KIMI_KEY   = os.environ.get("KIMI_API_KEY")
KIMI_BASE  = os.environ.get("KIMI_BASE_URL", "https://grid.ai.juspay.net/v1")
KIMI_MODEL = os.environ.get("KIMI_MODEL", "kimi-latest")

EXTRACT_SYS = (
    "You extract functional requirements from code artifacts. "
    "A functional requirement describes WHAT a module does for users of the system, "
    "not how it is implemented. Write exactly one sentence starting with a verb. "
    "Example: 'Validates JWT tokens and gates incoming API requests.'"
)
EXTRACT_TMPL = (
    "Module: {name}\n"
    "Code artifacts:\n{artifacts}\n\n"
    "Write one sentence describing the functional requirement this module satisfies."
)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_artifacts():
    print("Loading artifacts...", flush=True)

    graph_path = ARTIFACT_DIR / "graph_with_summaries.json"
    print(f"  Loading graph: {graph_path}", flush=True)
    graph = json.loads(open(graph_path).read())
    nodes_list = graph.get("nodes", [])
    # nodes is a list — convert to dict keyed by id for fast lookup
    nodes = {n["id"]: n for n in nodes_list if "id" in n}
    print(f"  {len(nodes)} nodes", flush=True)

    body_path = ARTIFACT_DIR.parent / "output" / "body_store.json"
    print(f"  Loading body_store: {body_path}", flush=True)
    body_store = json.loads(open(body_path).read()) if body_path.exists() else {}

    activity_path = ARTIFACT_DIR / "activity_index.json"
    print(f"  Loading activity_index: {activity_path}", flush=True)
    activity_index = json.loads(open(activity_path).read()) if activity_path.exists() else {}

    return nodes, body_store, activity_index


def select_modules(nodes, body_store, activity_index, limit=None):
    """Group nodes by module, select active modules, aggregate artifacts.

    Returns list of (module_name, aggregated_embed_texts, activity_score).
    Operates at MODULE level (not symbol level) to match activity_index granularity.
    """
    from collections import defaultdict
    module_symbols = defaultdict(list)

    for node_id, node in nodes.items():
        module = node.get("module") or node.get("id", "")
        embed_text = node.get("embed_text") or node.get("docstring") or ""
        body = body_store.get(node_id, "")
        if embed_text or body:
            module_symbols[module].append({
                "name": node.get("name", ""),
                "embed_text": embed_text[:200],
                "body": str(body)[:300] if body else "",
            })

    selected = []
    for module, symbols in module_symbols.items():
        act = activity_index.get(module, {}).get("activity_score", 0.0)
        if act < ACTIVITY_THRESHOLD: continue

        # Aggregate: take embed_text from top-3 symbols
        top_syms = symbols[:3]
        artifacts = "\n".join(
            f"  {s['name']}: {s['embed_text']}" for s in top_syms if s['embed_text']
        )
        if not artifacts: continue
        selected.append((module, artifacts, act))

    selected.sort(key=lambda x: -x[2])  # sort by activity desc
    print(f"  Selected {len(selected)} active modules (threshold={ACTIVITY_THRESHOLD})", flush=True)
    if limit:
        selected = selected[:limit]
        print(f"  Limited to {limit} for this run", flush=True)
    return selected


# ── LLM extraction ───────────────────────────────────────────────────────────
_VERB_RE = re.compile(r'^[A-Z][a-z]+(s|es|ed|ing)\s')
_PREAMBLE = ('the user', 'looking at', 'i ', "i'll", 'based on', 'from the',
             'these ', 'this module', 'it seems', 'note:', 'they appear', 'here ')

def _parse_requirement(text: str) -> str:
    """Extract the clean one-sentence answer from a reasoning model's chain-of-thought."""
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    # Pass 1: verb-start sentence (highest quality)
    for line in reversed(lines):
        low = line.lower()
        if (line.endswith('.') and len(line.split()) >= 5
                and _VERB_RE.match(line)
                and not any(low.startswith(p) for p in _PREAMBLE)):
            return line
    # Pass 2: any complete sentence not starting with preamble
    for line in reversed(lines):
        low = line.lower()
        if (line.endswith('.') and len(line.split()) >= 5
                and line[0].isupper()
                and not any(low.startswith(p) for p in _PREAMBLE)):
            return line
    # Fallback: last non-empty line (truncated)
    return lines[-1][:200] if lines else ""


def build_artifacts_text(name, node, body):
    parts = []
    summary = node.get("summary") or node.get("embed_text") or ""
    if summary: parts.append(f"Summary: {summary[:300]}")
    if body: parts.append(f"Body:\n{str(body)[:500]}")
    return "\n".join(parts) if parts else name


def extract_requirement(name, node, body, dry_run=False) -> str:
    """Call LLM to extract a one-sentence requirement description."""
    if dry_run:
        return f"[DRY RUN] Handles {name.split('.')[-1]} functionality."

    artifacts_text = build_artifacts_text(name, node, body)
    prompt = EXTRACT_TMPL.format(name=name, artifacts=artifacts_text)

    try:
        r = requests.post(
            f"{KIMI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {KIMI_KEY}", "Content-Type": "application/json"},
            json={
                "model": KIMI_MODEL,
                "messages": [
                    {"role": "system", "content": EXTRACT_SYS},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,
                "temperature": 0,
            },
            timeout=15
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        return _parse_requirement(raw)
    except Exception as e:
        return ""  # skip on error, will be retried on next run


# ── Embedding ────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the embed server."""
    r = requests.post(
        f"{EMBED_SERVER_URL}/embed",
        json={"texts": texts},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["embeddings"]


# ── Lance storage ─────────────────────────────────────────────────────────────
def save_to_lance(records: list[dict], output_path: Path):
    """Save requirement vectors to a LanceDB table."""
    try:
        import lance
        import numpy as np
        import pyarrow as pa
    except ImportError:
        print("  lance/pyarrow not available — skipping vector storage", flush=True)
        return

    vecs = np.array([r["vector"] for r in records], dtype=np.float32)
    names = [r["name"] for r in records]
    reqs = [r["requirement"] for r in records]

    tbl = pa.table({
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vecs.flatten(), type=pa.float32()), 4096),
        "name": pa.array(names, type=pa.string()),
        "requirement": pa.array(reqs, type=pa.string()),
    })
    lance.write_dataset(tbl, str(output_path), mode="overwrite")
    print(f"  Saved {len(records)} vectors to {output_path}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call LLM, use placeholder requirements")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process first N modules (for testing)")
    parser.add_argument("--embed", action="store_true",
                        help="Also generate embeddings (requires embed server)")
    args = parser.parse_args()

    nodes, body_store, activity_index = load_artifacts()
    modules = select_modules(nodes, body_store, activity_index, limit=args.limit)

    # Resume support: load existing requirements
    out_json = ARTIFACT_DIR / "requirements.json"
    existing = {}
    if RESUME and out_json.exists():
        existing = json.loads(open(out_json).read())
        print(f"  Resuming: {len(existing)} already extracted", flush=True)

    # Extract requirements
    results = dict(existing)
    todo = [(n, arts, a) for n, arts, a in modules if n not in existing]
    print(f"\nExtracting requirements for {len(todo)} modules...", flush=True)
    if args.dry_run:
        print("  DRY RUN — no LLM calls", flush=True)

    for i, (name, artifacts_text, act) in enumerate(todo):
        # Build a minimal node proxy for extract_requirement
        node_proxy = {"embed_text": artifacts_text}
        req = extract_requirement(name, node_proxy, "", dry_run=args.dry_run)
        if req:
            results[name] = {"requirement": req, "activity_score": act}

        if (i + 1) % BATCH_SIZE == 0:
            # Save progress checkpoint
            open(out_json, "w").write(json.dumps(results, indent=2))
            print(f"  [{i+1}/{len(todo)}] saved checkpoint", flush=True)

    # Final save
    open(out_json, "w").write(json.dumps(results, indent=2))
    n_extracted = len(results) - len(existing)
    print(f"\nDone. {n_extracted} new requirements extracted.", flush=True)
    print(f"Total in requirements.json: {len(results)}", flush=True)
    print(f"Output: {out_json}", flush=True)

    # Embed if requested
    if args.embed:
        print("\nEmbedding requirements...", flush=True)
        items = list(results.items())
        vectors = []
        for batch_start in range(0, len(items), 32):
            batch = items[batch_start:batch_start + 32]
            texts = [v["requirement"] for _, v in batch]
            try:
                embs = embed_texts(texts)
                for (name, meta), vec in zip(batch, embs):
                    vectors.append({"name": name, "requirement": meta["requirement"], "vector": vec})
            except Exception as e:
                print(f"  Embed error at {batch_start}: {e}", flush=True)

        save_to_lance(vectors, ARTIFACT_DIR / "requirements.lance")
        print(f"Embedded {len(vectors)} requirements.", flush=True)


if __name__ == "__main__":
    main()
