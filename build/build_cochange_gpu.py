"""
build_cochange_gpu.py — GPU-accelerated co-change index builder

Strategy:
  Phase 1 (CPU, sequential): Stream all 30 split JSON files with ijson.
    For each commit, collect source file paths → build list of
    (commit_row_idx, module_col_idx) integer pairs.
    This replaces the old O(k²) Python dict updates with integer accumulation.

  Phase 2 (GPU): Build sparse commit×module matrix A.
    Compute co-change = A^T @ A in one matmul — all pair counts at once.

  Phase 3 (CPU): Threshold, sort, write cochange_index.json.

Speedup vs old CPU builder:
  Old: combinations(mods, 2) → O(k²) Python dict updates per commit
  New: GPU matmul A^T @ A replaces all pair counting in one kernel call
  Parsing is still sequential (ijson), but no longer the bottleneck.

Memory (30k modules, 300k commits, avg 5 files/commit):
  Pair arrays:       ~1.5M entries × 8 bytes = ~12 MB
  Sparse A on GPU:   ~12 MB
  Dense A^T @ A:     30k × 30k × 4 bytes = ~3.6 GB  (fits in 32GB VRAM)
"""

import json, pathlib, time, sys
import numpy as np

try:
    import ijson
except ImportError:
    raise SystemExit("pip install ijson")

# ── Fast line-based parser (replaces ijson for large files) ──────────────────

def parse_file_fast(json_path: pathlib.Path) -> tuple[str | None, list, int, int]:
    """
    Line-scanner: extracts repo name and (commit_idx, module) pairs without
    full JSON parsing. Scans for "path": lines and commit boundaries.
    ~10-20x faster than ijson for large files.
    """
    repo_name    = None
    pairs        = []
    commits      = 0
    skipped      = 0
    commit_files = []
    in_commits   = False
    commit_ctr   = 0
    file_idx_local = 0

    with open(json_path, "r", encoding="utf-8", errors="ignore", buffering=8*1024*1024) as f:
        for raw in f:
            # Repo name (appears before commits section)
            if not repo_name and '"repository"' in raw:
                try:
                    repo_name = raw.split('"repository"')[1].split('"')[1]
                    if repo_name not in INDEXED_REPOS:
                        return None, [], 0, 0
                except IndexError:
                    pass
                continue

            # Skip everything until the commits array
            if not in_commits:
                if '"commits"' in raw and '[' in raw:
                    in_commits = True
                continue

            # Commit start: exactly "    {\n" (4-space indent = top-level array item)
            if raw == "    {\n" or raw == "    {":
                commit_files = []
                continue

            # File path: "path" key inside files_changed (deeper indent, has "path")
            if '"path"' in raw:
                try:
                    path = raw.split('"path"')[1].split('"')[1]
                    if is_source(path):
                        commit_files.append(path)
                except IndexError:
                    pass
                continue

            # Commit end: "    }," or "    }" (4-space indent)
            if raw in ("    },\n", "    }\n", "    },", "    }"):
                commits += 1
                if 2 <= len(commit_files) <= MAX_FILES and repo_name:
                    uid = f"{repo_name}::{json_path.stem}::{commit_ctr}"
                    for fp in commit_files:
                        pairs.append((uid, to_module(repo_name, fp)))
                elif len(commit_files) > MAX_FILES:
                    skipped += 1
                commit_ctr += 1
                commit_files = []

    return repo_name, pairs, commits, skipped

try:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
except ImportError:
    raise SystemExit("pip install torch")

# ── Config ────────────────────────────────────────────────────────────────────
SPLIT_DIR = pathlib.Path("/home/beast/projects/mindmap/pipeline/git_history_split")
OUT_PATH  = pathlib.Path("/home/beast/projects/mindmap/pipeline/demo_artifact/cochange_index.json")

INDEXED_REPOS = {
    "euler-api-gateway", "euler-api-txns", "euler-db", "euler-api-order",
    "graphh", "euler-api-pre-txn", "euler-api-customer", "basilisk-v3",
    "euler-drainer", "token_issuer_portal_backend", "haskell-sequelize",
}

SRC_EXTS   = {".hs", ".rs", ".hs-boot", ".purs"}
SKIP_DIRS  = {"test", "tests", "spec", "mock", "node_modules", "__pycache__", ".git", ".stack-work"}
MIN_WEIGHT = 3
TOP_K      = 40
MAX_FILES  = 50   # skip mega-commits


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_source(path: str) -> bool:
    p = pathlib.PurePosixPath(path)
    return p.suffix in SRC_EXTS and not (set(p.parts) & SKIP_DIRS)


def to_module(repo: str, fpath: str) -> str:
    p = fpath
    for ext in SRC_EXTS:
        if p.endswith(ext):
            p = p[:-len(ext)]
            break
    return f"{repo}::{p.replace('/', '::')}"


# ── Phase 1: Sequential parse → integer arrays ────────────────────────────────

def parse_all_files(split_files: list) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Stream all split JSON files. Build vocabulary and integer COO arrays.
    Returns (row_arr, col_arr, commit_vocab, module_vocab).
    """
    commit_vocab: dict[str, int] = {}
    module_vocab: dict[str, int] = {}
    row_list: list[int] = []
    col_list: list[int] = []
    repo_stats: dict[str, int] = {}

    for file_idx, json_path in enumerate(split_files):
        t0 = time.time()
        repo_name    = None
        commits      = 0
        skipped      = 0
        commit_files = []
        in_commit    = False
        commit_ctr   = 0   # monotonic commit counter per file

        # Use fast line scanner for large files (>200MB), ijson for small ones
        use_fast = json_path.stat().st_size > 200 * 1024 * 1024

        if use_fast:
            repo_name, new_pairs, commits, skipped = parse_file_fast(json_path)
            if repo_name:
                repo_stats[repo_name] = repo_stats.get(repo_name, 0) + commits
                for uid, mod in new_pairs:
                    if uid not in commit_vocab:
                        commit_vocab[uid] = len(commit_vocab)
                    if mod not in module_vocab:
                        module_vocab[mod] = len(module_vocab)
                    row_list.append(commit_vocab[uid])
                    col_list.append(module_vocab[mod])
                elapsed = time.time() - t0
                print(f"  [{file_idx+1:02d}/{len(split_files)}] {json_path.name:<45s} "
                      f"{commits:>8,} commits  {len(row_list):>9,} pairs total  {elapsed:.1f}s  [fast]",
                      flush=True)
            else:
                print(f"  [{file_idx+1:02d}/{len(split_files)}] {json_path.name:<45s} SKIPPED", flush=True)
            continue

        # ijson path for small files
        repo_name    = None
        commits      = 0
        skipped      = 0
        commit_files = []
        in_commit    = False
        commit_ctr   = 0

        try:
            with open(json_path, "rb") as f:
                parser = ijson.parse(f, use_float=True)
                for prefix, event, value in parser:

                    if prefix == "export_metadata.repository" and event == "string":
                        repo_name = value
                        if repo_name not in INDEXED_REPOS:
                            repo_name = None
                            break

                    if prefix == "commits.item" and event == "start_map":
                        in_commit    = True
                        commit_files = []

                    if in_commit and prefix == "commits.item.files_changed.item.path" and event == "string":
                        if is_source(value):
                            commit_files.append(value)

                    if prefix == "commits.item" and event == "end_map":
                        in_commit = False
                        commits  += 1

                        if 2 <= len(commit_files) <= MAX_FILES and repo_name:
                            commit_uid = f"{repo_name}::{file_idx}::{commit_ctr}"
                            if commit_uid not in commit_vocab:
                                commit_vocab[commit_uid] = len(commit_vocab)
                            commit_row = commit_vocab[commit_uid]

                            for fp in commit_files:
                                mod = to_module(repo_name, fp)
                                if mod not in module_vocab:
                                    module_vocab[mod] = len(module_vocab)
                                row_list.append(commit_row)
                                col_list.append(module_vocab[mod])

                        elif len(commit_files) > MAX_FILES:
                            skipped += 1

                        commit_ctr += 1

        except Exception as exc:
            print(f"  WARNING: {json_path.name} stopped at {commits:,} commits: {exc}", flush=True)

        if repo_name:
            repo_stats[repo_name] = repo_stats.get(repo_name, 0) + commits
            elapsed = time.time() - t0
            n_pairs = sum(1 for r in row_list[-commits*MAX_FILES:])  # approx
            print(f"  [{file_idx+1:02d}/{len(split_files)}] {json_path.name:<45s} "
                  f"{commits:>8,} commits  {len(row_list):>9,} pairs total  {elapsed:.1f}s",
                  flush=True)
        else:
            print(f"  [{file_idx+1:02d}/{len(split_files)}] {json_path.name:<45s} SKIPPED",
                  flush=True)

    return (np.array(row_list, dtype=np.int32),
            np.array(col_list, dtype=np.int32),
            commit_vocab, module_vocab, repo_stats)


# ── Phase 2: GPU matmul ───────────────────────────────────────────────────────

def gpu_cochange(rows: np.ndarray, cols: np.ndarray,
                 n_commits: int, n_modules: int) -> np.ndarray:
    device = torch.device("cuda")
    free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
    dense_gb = n_modules * n_modules * 4 / 1e9
    print(f"\nGPU: {torch.cuda.get_device_name(0)}  |  VRAM free: {free_gb:.1f} GB", flush=True)
    print(f"Matrix size: {n_modules}×{n_modules} = {dense_gb:.2f} GB", flush=True)

    # Build sparse A [n_commits × n_modules]
    print("Building sparse A on GPU...", flush=True)
    t0 = time.time()
    idx = torch.tensor(
        np.stack([rows.astype(np.int64), cols.astype(np.int64)], axis=0),
        device=device
    )
    vals = torch.ones(len(rows), dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(idx, vals, (n_commits, n_modules), device=device).coalesce()
    del idx, vals
    print(f"  done in {time.time()-t0:.1f}s  ({A._nnz():,} non-zeros)", flush=True)

    # Compute A^T @ A
    print("Computing A^T @ A on GPU...", flush=True)
    t0 = time.time()

    if dense_gb <= free_gb * 0.8:
        A_dense = A.to_dense()   # [n_commits × n_modules]
        C = torch.mm(A_dense.t(), A_dense)   # [n_modules × n_modules]
        C_np = C.cpu().numpy().astype(np.float32)
        del A_dense, C
    else:
        # Chunked: compute in row blocks to stay within VRAM
        chunk = max(1, int(free_gb * 0.4 * 1e9 / (n_modules * 4)))
        print(f"  Chunked mode: {chunk} rows/chunk", flush=True)
        A_t = A.t().coalesce()
        C_np = np.zeros((n_modules, n_modules), dtype=np.float32)
        for start in range(0, n_modules, chunk):
            end = min(start + chunk, n_modules)
            rows_dense = A_t[start:end].to_dense()
            sub = torch.mm(rows_dense, A.to_dense()).cpu().numpy()
            C_np[start:end] = sub
            del rows_dense, sub

    torch.cuda.empty_cache()
    print(f"  Matmul done in {time.time()-t0:.1f}s", flush=True)

    np.fill_diagonal(C_np, 0)  # self-pairs are meaningless
    return C_np


# ── Phase 3: Build output ─────────────────────────────────────────────────────

def build_edges(C: np.ndarray, module_vocab: dict) -> tuple[dict, int]:
    idx_to_module = {v: k for k, v in module_vocab.items()}
    n = len(module_vocab)
    edges: dict = {}
    total_pairs = 0

    print(f"Building edge dict from {n}×{n} matrix...", flush=True)
    for i in range(n):
        row = C[i]
        candidates = np.where(row >= MIN_WEIGHT)[0]
        if len(candidates) == 0:
            continue
        top = candidates[np.argsort(-row[candidates])][:TOP_K]
        partners = [{"module": idx_to_module[j], "weight": int(row[j])} for j in top]
        if partners:
            edges[idx_to_module[i]] = partners
            total_pairs += len(partners)

    return edges, total_pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def build():
    t_start = time.time()

    split_files = sorted(SPLIT_DIR.glob("*.json"), key=lambda p: p.name)
    print(f"Found {len(split_files)} split files  |  GPU: {torch.cuda.get_device_name(0)}")
    print(f"Output → {OUT_PATH}\n")

    # Phase 1
    rows, cols, commit_vocab, module_vocab, repo_stats = parse_all_files(split_files)
    total_commits = sum(repo_stats.values())
    print(f"\nParsing done in {time.time()-t_start:.0f}s")
    print(f"  {total_commits:,} commits  |  {len(commit_vocab):,} unique commits  |  "
          f"{len(module_vocab):,} modules  |  {len(rows):,} non-zeros\n")

    # Phase 2
    C = gpu_cochange(rows, cols, len(commit_vocab), len(module_vocab))
    del rows, cols

    # Phase 3
    edges, total_pairs = build_edges(C, module_vocab)
    del C

    index = {
        "meta": {
            "total_commits":    total_commits,
            "repos_indexed":    sorted(repo_stats.keys()),
            "commits_per_repo": repo_stats,
            "total_modules":    len(edges),
            "total_pairs":      total_pairs,
            "min_weight":       MIN_WEIGHT,
            "top_k":            TOP_K,
            "builder":          "gpu",
        },
        "edges": edges,
    }

    OUT_PATH.write_text(json.dumps(index, separators=(",", ":")))
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    elapsed = time.time() - t_start

    print(f"\n{'─'*60}")
    print(f"Modules   : {len(edges):,}")
    print(f"Pairs     : {total_pairs:,}")
    print(f"Written   : {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"Total time: {elapsed:.0f}s")

    print("\nCommits per repo:")
    for repo, count in sorted(repo_stats.items()):
        print(f"  {repo:<40s} {count:>8,}")

    print("\nSample edges:")
    shown = 0
    for mod, partners in edges.items():
        if shown >= 5: break
        svc  = mod.split("::")[0]
        name = "::".join(mod.split("::")[1:])
        print(f"  [{svc}] {name}")
        for p in partners[:3]:
            p_svc  = p["module"].split("::")[0]
            p_name = "::".join(p["module"].split("::")[1:])
            print(f"    → [{p_svc}] {p_name}  (weight={p['weight']})")
        shown += 1


if __name__ == "__main__":
    build()
