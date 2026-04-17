"""
Stage 11 — Guardrail Generation

For the top N critical modules (from criticality_index.json), use LLM to:
- Read the code + git history + co-change neighbors
- Infer: what does this code protect? what invariant must hold?
- Generate a guardrail document per critical section

Output: ARTIFACT_DIR/guardrails/*.md (one per guardrail)
        ARTIFACT_DIR/guardrails_index.json (summary)

Requires: LLM API (LM Studio, Kimi, or any OpenAI-compatible endpoint)
"""
import json, os, pathlib, time, re

ARTIFACT_DIR = pathlib.Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
OUTPUT_DIR = pathlib.Path(os.environ.get("OUTPUT_DIR", "output"))
GIT_HISTORY = pathlib.Path(os.environ.get("GIT_HISTORY", "git_history.json"))

API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://172.19.144.1:1234/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3-coder-30b-a3b-instruct")

TOP_N = int(os.environ.get("GUARDRAIL_TOP_N", "50"))

GUARDRAIL_DIR = ARTIFACT_DIR / "guardrails"
GUARDRAIL_DIR.mkdir(exist_ok=True)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: {path}: {e}")
        return {}


def get_module_context(mod_name, graph, cochange, ownership, criticality_entry):
    """Build a context string for the LLM about this module."""
    parts = []

    # Basic info from criticality
    parts.append(f"Module: {mod_name}")
    parts.append(f"Criticality Score: {criticality_entry.get('score', 0):.3f} (rank #{criticality_entry.get('rank', '?')})")
    parts.append(f"Key signals: {', '.join(criticality_entry.get('reasons', []))}")

    # Co-change neighbors
    edges = cochange.get("edges", {})
    neighbors = edges.get(mod_name, {})
    if isinstance(neighbors, dict):
        top_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_neighbors:
            parts.append(f"\nCo-change neighbors (files that always change together):")
            for nb, weight in top_neighbors:
                parts.append(f"  - {nb} (weight: {weight})")

    # Ownership
    modules = ownership.get("modules", {})
    own_data = modules.get(mod_name, [])
    if own_data:
        if isinstance(own_data, list):
            authors = own_data[:5]
        elif isinstance(own_data, dict):
            authors = own_data.get("authors", [])[:5]
        else:
            authors = []
        if authors:
            parts.append(f"\nCode owners:")
            for a in authors:
                if isinstance(a, dict):
                    parts.append(f"  - {a.get('name', a.get('email', '?'))}: {a.get('commits', '?')} commits")
                else:
                    parts.append(f"  - {a}")

    # Code snippet from graph (if available)
    if isinstance(graph, dict):
        nodes = graph.get("nodes", [])
        # Find matching nodes
        matching = [n for n in nodes if mod_name in n.get("module", "") or mod_name in n.get("id", "")][:3]
        if matching:
            parts.append(f"\nKey symbols in this module:")
            for n in matching:
                parts.append(f"  - {n.get('name', '?')} ({n.get('kind', '?')})")
                if n.get("type"):
                    parts.append(f"    Signature: {n['type'][:150]}")

    return "\n".join(parts)


def generate_guardrail(mod_name, context, client):
    """Use LLM to generate a guardrail document."""
    prompt = f"""You are a senior software architect reviewing critical code modules.
Given the following information about a critical module in a large enterprise codebase,
generate a guardrail document that would help any developer (or AI) understand why this
code is critical and what must stay true when changing it.

{context}

Generate a guardrail in this exact format:

# Guardrail: [module name]

**Criticality Score:** [score]

**Why it's critical:** [2-3 sentences explaining why this module matters based on the signals]

**What it protects:** [Infer the business invariant this code protects]

**What must stay true:** [Specific conditions that must hold — e.g., "lock must be held during X", "transaction must wrap Y"]

**Blast radius if broken:** [List affected modules/services based on co-change data]

**Co-change dependencies:** [Files that MUST change together with this one]

**Recommended reviewers:** [From ownership data]

**Review checklist for changes:**
- [ ] [Specific check 1]
- [ ] [Specific check 2]
- [ ] [Specific check 3]

Be specific to THIS module. Do not be generic. Use the co-change and ownership data provided.
If you can infer the business domain (payments, transactions, config), mention it explicitly."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"# Guardrail: {mod_name}\n\n**Error generating guardrail:** {e}"


def main():
    from openai import OpenAI

    print("=" * 60)
    print(f"Stage 11: Guardrail Generation (top {TOP_N} critical modules)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    criticality = load_json(ARTIFACT_DIR / "criticality_index.json")
    cochange = load_json(ARTIFACT_DIR / "cochange_index.json")
    ownership = load_json(ARTIFACT_DIR / "ownership_index.json")

    # Load graph (large — only need nodes for context)
    graph_path = ARTIFACT_DIR / "graph_with_summaries.json"
    print(f"  Loading graph from {graph_path}...")
    graph = load_json(graph_path)

    modules = criticality.get("modules", {})
    if not modules:
        print("ERROR: No modules in criticality index!")
        return

    # Get top N by score
    ranked = sorted(modules.items(), key=lambda x: x[1]["score"], reverse=True)[:TOP_N]
    print(f"\n  Generating guardrails for top {len(ranked)} modules...")

    # Connect to LLM
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    guardrails_index = []
    for i, (mod_name, crit_data) in enumerate(ranked, 1):
        print(f"\n  [{i}/{len(ranked)}] {mod_name[:60]}... (score: {crit_data['score']:.3f})")

        # Build context
        context = get_module_context(mod_name, graph, cochange, ownership, crit_data)

        # Generate guardrail
        t0 = time.time()
        guardrail_md = generate_guardrail(mod_name, context, client)
        elapsed = time.time() - t0
        print(f"    Generated in {elapsed:.1f}s")

        # Save individual guardrail file
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", mod_name)[:80]
        filename = f"{i:03d}_{safe_name}.md"
        (GUARDRAIL_DIR / filename).write_text(guardrail_md)

        guardrails_index.append({
            "rank": i,
            "module": mod_name,
            "score": crit_data["score"],
            "file": filename,
            "signals": crit_data.get("signals", {}),
        })

    # Save index
    index_path = ARTIFACT_DIR / "guardrails_index.json"
    with open(index_path, "w") as f:
        json.dump({
            "meta": {"total": len(guardrails_index), "model": MODEL},
            "guardrails": guardrails_index,
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(guardrails_index)} guardrails")
    print(f"  Directory: {GUARDRAIL_DIR}")
    print(f"  Index: {index_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
