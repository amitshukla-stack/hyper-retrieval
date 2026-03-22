"""
Rebuild vectors.lance from graph_with_summaries.json + embeddings.npy checkpoint.

Fixes the empty cluster_name/cluster_purpose fields caused by LanceDB being
built from graph_clustered.json (before stage 4 added cluster names).
"""
import json, pathlib, shutil
import numpy as np

PIPELINE = pathlib.Path(__file__).parent
OUT_DIR  = PIPELINE / "output"
ARTIFACT = PIPELINE / "demo_artifact"

EMBED_NPY  = OUT_DIR / "embeddings.npy"
EMBED_IDS  = OUT_DIR / "embed_ids.json"
GRAPH_PATH = ARTIFACT / "graph_with_summaries.json"
LANCE_PATH = ARTIFACT / "vectors.lance"


def node_to_text(n: dict) -> str:
    parts = [
        f"[{n.get('lang','?')}] service={n.get('service','?')}  module={n.get('module','')}",
        f"name={n.get('name','')}  kind={n.get('kind','')}",
    ]
    if n.get("type"):
        parts.append(f"signature: {n['type'][:200]}")
    if n.get("cluster_name"):
        parts.append(f"subsystem: {n['cluster_name']}")
    if n.get("cluster_purpose"):
        parts.append(f"purpose: {n['cluster_purpose'][:200]}")
    if n.get("docstring"):
        parts.append(f"doc: {n['docstring'][:400]}")
    if n.get("ghost_deps"):
        parts.append(f"external deps: {', '.join(n['ghost_deps'])}")
    if n.get("commit_history"):
        msgs = [c["msg"] for c in n["commit_history"][:3]]
        parts.append(f"recent changes: {'; '.join(msgs)}")
    return "\n".join(parts)


def main():
    import lancedb

    print("Loading embedding checkpoint...")
    embeddings = np.load(str(EMBED_NPY))
    embed_ids  = json.loads(EMBED_IDS.read_text())
    print(f"  {embeddings.shape}  ({len(embed_ids)} IDs)")

    print("Loading graph_with_summaries.json...")
    data  = json.loads(GRAPH_PATH.read_text())
    nodes = [n for n in data["nodes"] if n.get("kind") != "phantom"]
    print(f"  {len(nodes):,} non-phantom nodes")

    node_ids = [n["id"] for n in nodes]
    if node_ids != embed_ids:
        print("ERROR: node IDs do not match checkpoint IDs")
        return
    print("  Checkpoint alignment: OK")

    with_name = sum(1 for n in nodes if n.get("cluster_name"))
    print(f"  Nodes with cluster_name: {with_name:,} / {len(nodes):,}")

    print("Building text representations...")
    raw_texts = [node_to_text(n) for n in nodes]

    if LANCE_PATH.exists():
        print(f"Dropping old LanceDB...")
        shutil.rmtree(LANCE_PATH)

    print(f"Writing {len(nodes):,} vectors to LanceDB...")
    db = lancedb.connect(str(LANCE_PATH))

    records = [
        {
            "id":              n["id"],
            "text":            raw_texts[i],
            "vector":          embeddings[i].tolist(),
            "name":            n.get("name", ""),
            "service":         n.get("service", ""),
            "module":          n.get("module", ""),
            "kind":            n.get("kind", ""),
            "lang":            n.get("lang", ""),
            "cluster":         str(n.get("cluster", -1)),
            "cluster_name":    n.get("cluster_name", ""),
            "cluster_purpose": n.get("cluster_purpose", ""),
            "summary":         n.get("summary", ""),
            "file":            n.get("file", ""),
        }
        for i, n in enumerate(nodes)
    ]

    db.create_table("chunks", data=records)
    size_mb = sum(f.stat().st_size for f in LANCE_PATH.rglob("*") if f.is_file()) // (1024 * 1024)
    print(f"  Written: {LANCE_PATH}  ({size_mb}MB)")

    tbl = db.open_table("chunks")
    sample = tbl.search(embeddings[0].tolist()).limit(3).to_list()
    print(f"\nVerification sample:")
    for h in sample:
        print(f"  {h['name']:50s}  cluster={h['cluster_name']!r}")

    import pandas as pd
    all_rows = tbl.to_pandas()
    filled = (all_rows["cluster_name"] != "").sum()
    print(f"\ncluster_name filled: {filled:,} / {len(all_rows):,}")
    print("Done.")


if __name__ == "__main__":
    main()
