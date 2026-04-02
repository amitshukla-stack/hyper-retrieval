"""
Stage 6 — Build evolutionary coupling graph from git history.

Streams one commit at a time (O(1) memory).
Handles truncated/incomplete JSON files gracefully — writes partial results.
"""
import argparse, json, pathlib, sys
from collections import defaultdict
from itertools import combinations

try:
    import ijson
except ImportError:
    raise SystemExit("pip install ijson")

parser = argparse.ArgumentParser(description="Build co-change index from git history")
parser.add_argument("--git-history", type=pathlib.Path,
                    default=pathlib.Path("/home/beast/projects/workspaces/juspay/git_history.json"))
parser.add_argument("--output", type=pathlib.Path, default=None,
                    help="Output path (default: <artifact-dir>/cochange_index.json)")
parser.add_argument("--artifact-dir", type=pathlib.Path, default=None)
parser.add_argument("--min-weight", type=int, default=None,
                    help="Min co-change weight (default: auto based on repo size)")
_args = parser.parse_args()

GIT_HISTORY = _args.git_history
if _args.output:
    OUT_PATH = _args.output
elif _args.artifact_dir:
    OUT_PATH = _args.artifact_dir / "cochange_index.json"
else:
    OUT_PATH = GIT_HISTORY.parent / "cochange_index.json"

SRC_EXTS   = {".hs", ".rs", ".hs-boot", ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java"}
SKIP_DIRS  = {".stack-work", "test", "tests", "spec", "mock", "node_modules", "__pycache__", ".git", "venv", ".venv"}
TOP_K      = 30
MAX_FILES  = 40  # skip mega-commits


def is_source(path: str) -> bool:
    p = pathlib.PurePosixPath(path)
    if p.suffix not in SRC_EXTS:
        return False
    return not (set(p.parts) & SKIP_DIRS)


def to_module(repo: str, fpath: str) -> str:
    p = fpath
    for ext in SRC_EXTS:
        p = p.replace(ext, "")
    return f"{repo}::{p.replace('/', '::')}"


def build():
    cochange = defaultdict(lambda: defaultdict(int))
    total_commits = 0
    skipped       = 0
    current_repo  = None
    truncated     = False

    print(f"Streaming {GIT_HISTORY}...", flush=True)

    with open(GIT_HISTORY, "rb") as f:
        parser = ijson.parse(f, use_float=True)
        in_commit    = False
        commit_files = []

        try:
            for prefix, event, value in parser:

                # Track current repo name
                if prefix == "repositories.item.name" and event == "string":
                    current_repo = value
                    continue

                # Detect commit start
                if prefix == "repositories.item.commits.item" and event == "start_map":
                    in_commit    = True
                    commit_files = []
                    continue

                # Collect only files_changed paths (ignore diff/body/other fields)
                if in_commit and event == "string" and "files_changed" in prefix and prefix.endswith(".path"):
                    if is_source(value):
                        commit_files.append(value)
                    continue

                # Commit ends — process it
                if prefix == "repositories.item.commits.item" and event == "end_map":
                    in_commit = False
                    total_commits += 1

                    if 2 <= len(commit_files) <= MAX_FILES and current_repo:
                        mods = [to_module(current_repo, fp) for fp in commit_files]
                        for a, b in combinations(mods, 2):
                            cochange[a][b] += 1
                            cochange[b][a] += 1
                    elif len(commit_files) > MAX_FILES:
                        skipped += 1

                    if total_commits % 5000 == 0:
                        print(f"  {total_commits:,} commits  {len(cochange):,} modules  "
                              f"repo={current_repo}", flush=True)
                    continue

        except Exception as exc:
            truncated = True
            print(f"\nWARNING: JSON parsing stopped at {total_commits:,} commits: {exc}", flush=True)
            print("Writing partial results...", flush=True)

    status = "PARTIAL (file truncated)" if truncated else "COMPLETE"
    print(f"\n{status}: {total_commits:,} commits processed  {skipped} mega-commits skipped",
          flush=True)
    print(f"Unique modules with co-change partners: {len(cochange):,}", flush=True)

    # Auto-scale MIN_WEIGHT based on repo size (small repos need lower threshold)
    if _args.min_weight is not None:
        MIN_WEIGHT = _args.min_weight
    elif total_commits < 200:
        MIN_WEIGHT = 2
    elif total_commits < 1000:
        MIN_WEIGHT = 2
    else:
        MIN_WEIGHT = 3

    # Filter: keep only pairs with weight >= MIN_WEIGHT, cap at TOP_K per module
    edges, total_pairs = {}, 0
    for mod, partners in cochange.items():
        filtered = sorted(
            [{"module": m, "weight": w} for m, w in partners.items() if w >= MIN_WEIGHT],
            key=lambda x: -x["weight"]
        )[:TOP_K]
        if filtered:
            edges[mod] = filtered
            total_pairs += len(filtered)

    print(f"After filter (weight>={MIN_WEIGHT}): {len(edges):,} modules  "
          f"{total_pairs:,} edges", flush=True)

    index = {
        "meta": {
            "total_commits":  total_commits,
            "truncated":      truncated,
            "total_modules":  len(edges),
            "total_pairs":    total_pairs,
            "min_weight":     MIN_WEIGHT,
        },
        "edges": edges,
    }

    OUT_PATH.write_text(json.dumps(index, separators=(",", ":")))
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nWritten: {OUT_PATH}  ({size_mb:.1f}MB)", flush=True)

    print("\nSample co-change edges:")
    for mod, partners in list(edges.items())[:5]:
        svc = mod.split("::")[0]
        name = "::".join(mod.split("::")[1:])
        print(f"  [{svc}] {name}")
        for p in partners[:3]:
            p_svc  = p["module"].split("::")[0]
            p_name = "::".join(p["module"].split("::")[1:])
            print(f"    -> [{p_svc}] {p_name}  (w={p['weight']})")


if __name__ == "__main__":
    build()
