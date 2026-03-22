"""
Stage 6 (v3) — Build co-change index from split git history files.

Reads all per-repo split JSON files from SPLIT_DIR, processes only repos
that are in our indexed services, streams one commit at a time (O(1) memory).

Split file format (per repo, may be multiple files):
  {
    "export_metadata": {"repository": "euler-api-gateway", ...},
    "commits": [
      {"files_changed": [{"path": "src/Foo.hs", "status": "M"}, ...], ...},
      ...
    ]
  }
"""
import json, pathlib, sys
from collections import defaultdict
from itertools import combinations

try:
    import ijson
except ImportError:
    raise SystemExit("pip install ijson")

# ── Config ────────────────────────────────────────────────────────────────────
SPLIT_DIR = pathlib.Path("/home/beast/projects/mindmap/pipeline/git_history_split")
OUT_PATH  = pathlib.Path("/home/beast/projects/mindmap/pipeline/demo_artifact/cochange_index.json")

# Only process repos that are indexed in the graph (skip dashboards, QA, PS, etc.)
INDEXED_REPOS = {
    "euler-api-gateway",
    "euler-api-txns",
    "euler-db",
    "euler-api-order",
    "graphh",
    "euler-api-pre-txn",
    "euler-api-customer",
    "basilisk-v3",
    "euler-drainer",
    "token_issuer_portal_backend",
    "haskell-sequelize",
    # UCS: no split file available — skipped
}

SRC_EXTS  = {".hs", ".rs", ".hs-boot", ".purs"}
SKIP_DIRS = {".stack-work", "test", "tests", "spec", "mock", "node_modules", "__pycache__", ".git"}
MIN_WEIGHT = 3    # ignore pairs that co-change fewer than 3 times
TOP_K      = 40   # max co-change partners per module
MAX_FILES  = 50   # skip mega-commits (bulk reformats, auto-generated changes)


def is_source(path: str) -> bool:
    p = pathlib.PurePosixPath(path)
    if p.suffix not in SRC_EXTS:
        return False
    return not (set(p.parts) & SKIP_DIRS)


def to_module(repo: str, fpath: str) -> str:
    """
    Convert a file path to a stable module identifier.
    Format: repo::path::without::extension (mirrors retrieval_engine naming)
    """
    p = fpath
    for ext in SRC_EXTS:
        if p.endswith(ext):
            p = p[:-len(ext)]
            break
    return f"{repo}::{p.replace('/', '::')}"


def process_file(json_path: pathlib.Path, cochange: dict, repo_stats: dict) -> str | None:
    """
    Stream one split JSON file. Returns repo name on success, None if skipped.
    Accumulates co-change counts into the shared `cochange` dict.
    """
    commits = 0
    skipped = 0
    repo_name = None

    with open(json_path, "rb") as f:
        parser = ijson.parse(f, use_float=True)
        in_commit    = False
        commit_files = []

        try:
            for prefix, event, value in parser:

                # Read repo name from metadata
                if prefix == "export_metadata.repository" and event == "string":
                    repo_name = value
                    if repo_name not in INDEXED_REPOS:
                        print(f"  Skipping {json_path.name} (repo={repo_name} not indexed)")
                        return None
                    continue

                # Commit starts
                if prefix == "commits.item" and event == "start_map":
                    in_commit    = True
                    commit_files = []
                    continue

                # Collect source file paths
                if in_commit and prefix == "commits.item.files_changed.item.path" and event == "string":
                    if is_source(value):
                        commit_files.append(value)
                    continue

                # Commit ends — process pairs
                if prefix == "commits.item" and event == "end_map":
                    in_commit = False
                    commits += 1

                    if 2 <= len(commit_files) <= MAX_FILES and repo_name:
                        mods = [to_module(repo_name, fp) for fp in commit_files]
                        for a, b in combinations(mods, 2):
                            cochange[a][b] += 1
                            cochange[b][a] += 1
                    elif len(commit_files) > MAX_FILES:
                        skipped += 1

        except Exception as exc:
            print(f"  WARNING: parse stopped at {commits:,} commits in {json_path.name}: {exc}")

    if repo_name:
        repo_stats[repo_name] = repo_stats.get(repo_name, 0) + commits
    return repo_name


def build():
    cochange: dict = defaultdict(lambda: defaultdict(int))
    repo_stats: dict = {}

    # Collect and sort split files: process repos in alphabetical order, files in numeric order
    split_files = sorted(
        [f for f in SPLIT_DIR.glob("*.json")],
        key=lambda p: p.name
    )

    print(f"Found {len(split_files)} split files in {SPLIT_DIR}")
    print(f"Processing {len(INDEXED_REPOS)} indexed repos...\n")

    total_files_processed = 0
    for json_path in split_files:
        print(f"→ {json_path.name}", end=" ", flush=True)
        repo = process_file(json_path, cochange, repo_stats)
        if repo:
            total_files_processed += 1
            commits_so_far = repo_stats.get(repo, 0)
            modules_so_far = len(cochange)
            print(f"[{repo}] {commits_so_far:,} commits so far | {modules_so_far:,} modules total")
        print()

    print(f"\n{'─'*60}")
    print(f"Files processed : {total_files_processed}")
    print(f"Repos covered   : {len(repo_stats)}")
    print("\nCommits per repo:")
    for repo, count in sorted(repo_stats.items()):
        print(f"  {repo:<40s} {count:>8,} commits")

    total_commits = sum(repo_stats.values())
    print(f"\nTotal commits   : {total_commits:,}")
    print(f"Raw modules     : {len(cochange):,}")

    # Filter: keep pairs with weight >= MIN_WEIGHT, cap at TOP_K per module
    edges: dict = {}
    total_pairs = 0
    for mod, partners in cochange.items():
        filtered = sorted(
            [{"module": m, "weight": w} for m, w in partners.items() if w >= MIN_WEIGHT],
            key=lambda x: -x["weight"],
        )[:TOP_K]
        if filtered:
            edges[mod] = filtered
            total_pairs += len(filtered)

    print(f"After filter (weight>={MIN_WEIGHT}, top {TOP_K}): {len(edges):,} modules  {total_pairs:,} edges")

    index = {
        "meta": {
            "total_commits":  total_commits,
            "repos_indexed":  sorted(repo_stats.keys()),
            "commits_per_repo": repo_stats,
            "total_modules":  len(edges),
            "total_pairs":    total_pairs,
            "min_weight":     MIN_WEIGHT,
            "top_k":          TOP_K,
        },
        "edges": edges,
    }

    OUT_PATH.write_text(json.dumps(index, separators=(",", ":")))
    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nWritten: {OUT_PATH}  ({size_mb:.1f} MB)")

    # Sample output
    print("\nSample co-change edges:")
    shown = 0
    for mod, partners in edges.items():
        if shown >= 5:
            break
        svc  = mod.split("::")[0]
        name = "::".join(mod.split("::")[1:])
        print(f"  [{svc}] {name}")
        for p in partners[:3]:
            p_svc  = p["module"].split("::")[0]
            p_name = "::".join(p["module"].split("::")[1:])
            print(f"      -> [{p_svc}] {p_name}  (weight={p['weight']})")
        shown += 1


if __name__ == "__main__":
    build()
