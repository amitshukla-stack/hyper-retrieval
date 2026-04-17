"""
Stage 12 — Targeted Summarization for Critical Modules

Generates 1-2 line summaries for the top N critical modules by reading
their actual source code from body_store.json. Focuses on hand-written
modules (not src-generated) where documentation matters most.

Updates graph_with_summaries.json with the new summaries.

Output: Updates nodes in graph_with_summaries.json with 'summary' field
"""
import json, os, pathlib, time, re

ARTIFACT_DIR = pathlib.Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
OUTPUT_DIR = pathlib.Path(os.environ.get("OUTPUT_DIR", "output"))

API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://172.19.144.1:1234/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen3-coder-30b-a3b-instruct")

TOP_N = int(os.environ.get("SUMMARIZE_TOP_N", "200"))


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: {path}: {e}")
        return {}


def summarize_function(name, code, module, kind, client):
    """Generate a 1-2 line summary of a function/type from its source code."""
    # Truncate very long code
    code_snippet = code[:2000] if len(code) > 2000 else code

    prompt = f"""Read this {kind} from module {module} and write a 1-2 line summary of what it does.
Be specific. Don't be generic. If it's boilerplate, say "Boilerplate: [what kind]".

Name: {name}
Code:
```
{code_snippet}
```

Summary (1-2 lines only, no markdown):"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating summary: {e}]"


def main():
    from openai import OpenAI

    print("=" * 60)
    print(f"Stage 12: Targeted Summarization (top {TOP_N} critical modules)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    criticality = load_json(ARTIFACT_DIR / "criticality_index.json")
    body_store = load_json(OUTPUT_DIR / "body_store.json")
    graph_path = OUTPUT_DIR / "graph_clustered.json"
    graph = load_json(graph_path)

    if not graph.get("nodes"):
        # Try graph_with_summaries
        graph_path = ARTIFACT_DIR / "graph_with_summaries.json"
        graph = load_json(graph_path)

    nodes = graph.get("nodes", [])
    print(f"  {len(nodes)} nodes in graph")
    print(f"  {len(body_store)} entries in body_store")

    # Get critical modules (hand-written only)
    modules = criticality.get("modules", {})
    ranked = sorted(modules.items(), key=lambda x: x[1]["score"], reverse=True)
    hand_written = [(m, d) for m, d in ranked if "src-generated" not in m and "euler-x" not in m][:TOP_N]

    print(f"  Targeting {len(hand_written)} hand-written critical modules")

    # Build module -> node indices map
    module_to_nodes = {}
    for i, node in enumerate(nodes):
        mod = node.get("module", "")
        if mod not in module_to_nodes:
            module_to_nodes[mod] = []
        module_to_nodes[mod].append(i)

    # Find nodes that need summaries in critical modules
    nodes_to_summarize = []
    critical_module_names = set()
    for mod_name, _ in hand_written:
        # Extract the module part (after service::)
        parts = mod_name.split("::")
        # Try matching by the module field in graph nodes
        for node_idx_list_key in module_to_nodes:
            if any(p in node_idx_list_key for p in parts[-3:] if len(p) > 3):
                for idx in module_to_nodes[node_idx_list_key]:
                    node = nodes[idx]
                    if not node.get("summary") and node.get("kind") not in ("phantom", "import"):
                        nid = node.get("id", "")
                        if nid in body_store:
                            nodes_to_summarize.append((idx, nid, body_store[nid]))
                            critical_module_names.add(node_idx_list_key)

    # Deduplicate
    seen_ids = set()
    unique_nodes = []
    for idx, nid, code in nodes_to_summarize:
        if nid not in seen_ids:
            seen_ids.add(nid)
            unique_nodes.append((idx, nid, code))

    # Cap at reasonable number
    unique_nodes = unique_nodes[:TOP_N * 5]  # ~5 functions per module

    print(f"  Found {len(unique_nodes)} unsummarized functions in {len(critical_module_names)} critical modules")

    if not unique_nodes:
        print("  No functions to summarize!")
        return

    # Connect to LLM
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Generate summaries
    updated = 0
    t0 = time.time()
    for i, (node_idx, nid, code) in enumerate(unique_nodes):
        node = nodes[node_idx]
        name = node.get("name", nid)
        module = node.get("module", "")
        kind = node.get("kind", "function")

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(unique_nodes)}] ({rate:.1f}/s) {name[:50]}...")

        summary = summarize_function(name, code, module, kind, client)
        if summary and not summary.startswith("[Error"):
            nodes[node_idx]["summary"] = summary
            updated += 1

    elapsed = time.time() - t0
    print(f"\n  Summarized {updated}/{len(unique_nodes)} functions in {elapsed:.0f}s")

    # Save updated graph
    graph["nodes"] = nodes
    with open(graph_path, "w") as f:
        json.dump(graph, f)
    print(f"  Updated {graph_path}")

    # Also update the artifacts copy if different
    artifact_graph = ARTIFACT_DIR / "graph_with_summaries.json"
    if str(graph_path) != str(artifact_graph):
        import shutil
        shutil.copy2(graph_path, artifact_graph)
        print(f"  Copied to {artifact_graph}")

    print(f"\n{'=' * 60}")
    print(f"Stage 12 complete: {updated} function summaries generated")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
