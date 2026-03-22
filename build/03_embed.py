"""
Stage 3 — GPU-batch embed all nodes with Qwen3-Embedding-8B → LanceDB.

Model: Qwen/Qwen3-Embedding-8B
  - 8B params, 4096d, 32K context, instruction-aware
  - Top MTEB as of mid-2025; handles mixed code+natural language well
  - Runs in fp16 on RTX 5090 (32GB VRAM), batch_size=32 safe

LanceDB note: LanceDB requires a native Linux ext4 filesystem (mmap).
  Writing to /mnt/d/... (NTFS via WSL2) fails at finalise time.
  Strategy: write to /tmp/vectors.lance, then rsync to demo_artifact/.

Checkpoint: embeddings saved to output/embeddings.npy + output/embed_ids.json
  so GPU work is not lost if the LanceDB step fails.

Input:  pipeline/output/graph_clustered.json
Output: pipeline/demo_artifact/vectors.lance/
        pipeline/output/graph_clustered.json  (embed_text added to each node)
"""
import json, pathlib, os, shutil

OUT_DIR      = pathlib.Path(__file__).parent / "output"
ARTIFACT_DIR = pathlib.Path(__file__).parent / "demo_artifact"
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_PATH = pathlib.Path(__file__).parent / "models" / "qwen3-embed-8b"
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

# Native Linux temp path for LanceDB (avoids NTFS mmap incompatibility)
LANCE_TMP  = pathlib.Path("/tmp/vectors.lance")
LANCE_DEST = ARTIFACT_DIR / "vectors.lance"

# Numpy checkpoint paths — survive LanceDB failures
EMBED_NPY  = OUT_DIR / "embeddings.npy"
EMBED_IDS  = OUT_DIR / "embed_ids.json"

INSTRUCTION = (
    "Instruct: Represent this code module for finding semantically similar "
    "components across microservices. "
    "Query: "
)


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


def run_embedding(nodes, device):
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model_id = str(MODEL_PATH) if MODEL_PATH.exists() else "Qwen/Qwen3-Embedding-8B"
    print(f"\nLoading model: {model_id}")
    model = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
    )
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    raw_texts = [node_to_text(n) for n in nodes]
    texts     = [INSTRUCTION + t for t in raw_texts]

    print(f"\nEncoding {len(texts):,} nodes in batches of {BATCH_SIZE}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Shape: {embeddings.shape}")

    # ── Checkpoint to disk immediately ───────────────────────────────────────
    print(f"\nSaving embedding checkpoint...")
    np.save(str(EMBED_NPY), embeddings)
    EMBED_IDS.write_text(json.dumps([n["id"] for n in nodes]))
    print(f"  Saved {EMBED_NPY.name} ({EMBED_NPY.stat().st_size // 1024 // 1024}MB)")

    return raw_texts, embeddings


def write_lancedb(nodes, raw_texts, embeddings):
    import lancedb

    # Clean tmp location
    if LANCE_TMP.exists():
        shutil.rmtree(LANCE_TMP)

    print(f"\nWriting {len(nodes):,} vectors to LanceDB...")
    print(f"  Temp path (native ext4): {LANCE_TMP}")

    db    = lancedb.connect(str(LANCE_TMP))
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

    table = db.create_table("chunks", data=records)
    print(f"  Written to {LANCE_TMP}")

    # Copy to Windows destination
    if LANCE_DEST.exists():
        shutil.rmtree(LANCE_DEST)
    shutil.copytree(LANCE_TMP, LANCE_DEST)
    print(f"  Copied → {LANCE_DEST}")

    size_mb = sum(f.stat().st_size for f in LANCE_DEST.rglob("*") if f.is_file()) // (1024*1024)
    print(f"  Final size: {size_mb}MB")
    return table


def main():
    import torch
    import numpy as np

    # ── Device setup ─────────────────────────────────────────────────────────
    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  CUDA: {torch.cuda.get_device_name(0)} ({vram_gb:.1f}GB VRAM)")
        except Exception as e:
            print(f"  CUDA failed ({e}), using CPU")
    print(f"  Device: {device}")

    # ── Load graph ───────────────────────────────────────────────────────────
    data  = json.loads((OUT_DIR / "graph_clustered.json").read_text())
    nodes = [n for n in data["nodes"] if n.get("kind") != "phantom"]
    print(f"  Nodes to embed: {len(nodes):,}")

    # ── Embeddings: use checkpoint if available ───────────────────────────────
    if EMBED_NPY.exists() and EMBED_IDS.exists():
        saved_ids = json.loads(EMBED_IDS.read_text())
        node_ids  = [n["id"] for n in nodes]
        if saved_ids == node_ids:
            print(f"\nCheckpoint found — loading embeddings from disk (skipping GPU step)")
            embeddings = np.load(str(EMBED_NPY))
            raw_texts  = [node_to_text(n) for n in nodes]
            print(f"  Loaded {embeddings.shape} from {EMBED_NPY.name}")
        else:
            print(f"\nCheckpoint node list mismatch — re-running embedding")
            raw_texts, embeddings = run_embedding(nodes, device)
    else:
        raw_texts, embeddings = run_embedding(nodes, device)

    # ── Write LanceDB ─────────────────────────────────────────────────────────
    write_lancedb(nodes, raw_texts, embeddings)

    print(f"\n✓ Wrote {len(nodes):,} vectors → {LANCE_DEST}")
    print(f"  Vector dim : {embeddings.shape[1]}")

    # ── Persist embed_text back to graph ─────────────────────────────────────
    text_map = {n["id"]: t for n, t in zip(nodes, raw_texts)}
    for n in data["nodes"]:
        n["embed_text"] = text_map.get(n["id"], "")
    (OUT_DIR / "graph_clustered.json").write_text(json.dumps(data, indent=2))
    print("  Updated graph_clustered.json with embed_text")


if __name__ == "__main__":
    main()
