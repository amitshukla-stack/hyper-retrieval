"""
Stage 6b — Build cross-repo co-change index from git history.

Detects files in DIFFERENT repos that frequently change within a temporal
window of each other. This fills the gap where intra-repo co-change (06)
cannot see cross-service coupling.

Algorithm:
  1. Load all commits with timestamps from git_history.json
  2. Sort all commits chronologically
  3. For each commit, find commits in OTHER repos within ±T hours
  4. Build cross-repo co-change pairs (files from different repos)
  5. Output cross_cochange_index.json (same schema as cochange_index.json)

Optimizations:
  - Sliding window on sorted commits (avoids O(C²))
  - Only pairs source files (SRC_EXTS filter)
  - Skips mega-commits (>MAX_FILES)
  - Same-author changes get boosted weight

Usage:
  python3 build/06b_build_cross_cochange.py [--git-history FILE] [--artifact-dir DIR]
  python3 build/06b_build_cross_cochange.py --time-window 24 --min-weight 3
"""
import argparse
import json
import pathlib
import sys
from collections import defaultdict
from datetime import datetime, timezone

SRC_EXTS = {".hs", ".rs", ".hs-boot", ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java"}
SKIP_DIRS = {".stack-work", "test", "tests", "spec", "mock", "node_modules",
             "__pycache__", ".git", "venv", ".venv"}
MAX_FILES = 40
TOP_K = 30


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


def parse_date(date_str: str) -> float:
    """Parse ISO date string to Unix timestamp."""
    try:
        # Handle ISO 8601 with timezone (e.g., 2024-01-15T10:30:00+05:30)
        dt = datetime.fromisoformat(date_str)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def load_commits(git_history_path: pathlib.Path) -> list[dict]:
    """Load all commits with timestamps, grouped by repo."""
    try:
        import ijson
    except ImportError:
        raise SystemExit("pip install ijson")

    commits = []
    current_repo = None
    total = 0

    print(f"Loading commits from {git_history_path}...", flush=True)

    with open(git_history_path, "rb") as f:
        parser = ijson.parse(f, use_float=True)
        in_commit = False
        commit_date = ""
        commit_author = ""
        commit_files = []

        try:
            for prefix, event, value in parser:
                if prefix == "repositories.item.name" and event == "string":
                    current_repo = value
                    continue

                if prefix == "repositories.item.commits.item" and event == "start_map":
                    in_commit = True
                    commit_files = []
                    commit_date = ""
                    commit_author = ""
                    continue

                if in_commit:
                    if prefix == "repositories.item.commits.item.date" and event == "string":
                        commit_date = value
                        continue
                    if prefix == "repositories.item.commits.item.author_email" and event == "string":
                        commit_author = value.lower()
                        continue

                if in_commit and event == "string" and "files_changed" in prefix and prefix.endswith(".path"):
                    if is_source(value):
                        commit_files.append(value)
                    continue

                if prefix == "repositories.item.commits.item" and event == "end_map":
                    in_commit = False
                    total += 1

                    if commit_files and len(commit_files) <= MAX_FILES and current_repo:
                        ts = parse_date(commit_date)
                        if ts > 0:
                            modules = [to_module(current_repo, fp) for fp in commit_files]
                            commits.append({
                                "repo": current_repo,
                                "ts": ts,
                                "author": commit_author,
                                "modules": modules,
                            })

                    if total % 10000 == 0:
                        print(f"  {total:,} commits scanned...", flush=True)
                    continue

        except Exception as exc:
            print(f"WARNING: JSON parsing stopped at {total:,} commits: {exc}", flush=True)

    print(f"Loaded {len(commits):,} usable commits from {total:,} total", flush=True)
    return commits


def build_cross_cochange(commits: list[dict], time_window_hours: float,
                         author_boost: float) -> dict:
    """Build cross-repo co-change pairs using sliding temporal window."""
    window_secs = time_window_hours * 3600
    cochange = defaultdict(lambda: defaultdict(float))

    # Sort by timestamp
    commits.sort(key=lambda c: c["ts"])
    n = len(commits)

    print(f"Building cross-repo pairs (window=±{time_window_hours}h, "
          f"author_boost={author_boost}x)...", flush=True)

    cross_pairs_found = 0

    for i in range(n):
        ci = commits[i]

        # Look forward from i until outside window
        j = i + 1
        while j < n and (commits[j]["ts"] - ci["ts"]) <= window_secs:
            cj = commits[j]

            # Only cross-repo pairs
            if ci["repo"] != cj["repo"]:
                # Weight: 1.0 base, boosted if same author
                weight = 1.0
                if ci["author"] and ci["author"] == cj["author"]:
                    weight = author_boost

                # Create pairs between all modules in ci and cj
                for mod_a in ci["modules"]:
                    for mod_b in cj["modules"]:
                        cochange[mod_a][mod_b] += weight
                        cochange[mod_b][mod_a] += weight
                        cross_pairs_found += 1

            j += 1

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1:,}/{n:,} commits processed, "
                  f"{cross_pairs_found:,} raw pairs...", flush=True)

    print(f"Raw cross-repo pairs: {cross_pairs_found:,}", flush=True)
    print(f"Unique modules with cross-repo partners: {len(cochange):,}", flush=True)
    return cochange


def write_index(cochange: dict, min_weight: int, out_path: pathlib.Path,
                total_commits: int, time_window_hours: float):
    """Filter and write the cross-repo co-change index."""
    edges = {}
    total_pairs = 0

    for mod, partners in cochange.items():
        filtered = sorted(
            [{"module": m, "weight": round(w, 1)}
             for m, w in partners.items() if w >= min_weight],
            key=lambda x: -x["weight"]
        )[:TOP_K]
        if filtered:
            edges[mod] = filtered
            total_pairs += len(filtered)

    # Count unique repo pairs
    repo_pairs = set()
    for mod, partners in edges.items():
        repo_a = mod.split("::")[0]
        for p in partners:
            repo_b = p["module"].split("::")[0]
            if repo_a != repo_b:
                pair = tuple(sorted([repo_a, repo_b]))
                repo_pairs.add(pair)

    print(f"\nAfter filter (weight>={min_weight}): {len(edges):,} modules, "
          f"{total_pairs:,} edges, {len(repo_pairs)} repo pairs", flush=True)

    index = {
        "meta": {
            "type": "cross_repo_cochange",
            "total_commits": total_commits,
            "total_modules": len(edges),
            "total_pairs": total_pairs,
            "min_weight": min_weight,
            "time_window_hours": time_window_hours,
            "repo_pairs": len(repo_pairs),
        },
        "edges": edges,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, separators=(",", ":")))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Written: {out_path}  ({size_mb:.1f}MB)", flush=True)

    # Sample output
    print("\nTop cross-repo co-change edges:")
    all_edges = []
    for mod, partners in edges.items():
        for p in partners:
            repo_a = mod.split("::")[0]
            repo_b = p["module"].split("::")[0]
            if repo_a != repo_b:
                all_edges.append((mod, p["module"], p["weight"]))

    all_edges.sort(key=lambda x: -x[2])
    for mod_a, mod_b, weight in all_edges[:10]:
        svc_a = mod_a.split("::")[0]
        name_a = "::".join(mod_a.split("::")[1:])
        svc_b = mod_b.split("::")[0]
        name_b = "::".join(mod_b.split("::")[1:])
        print(f"  [{svc_a}] {name_a[:40]}  <->  [{svc_b}] {name_b[:40]}  (w={weight})")

    return len(edges), total_pairs


def main():
    parser = argparse.ArgumentParser(description="Build cross-repo co-change index")
    parser.add_argument("--git-history", type=pathlib.Path,
                        default=pathlib.Path("/home/beast/projects/workspaces/juspay/git_history.json"))
    parser.add_argument("--artifact-dir", type=pathlib.Path, default=None)
    parser.add_argument("--time-window", type=float, default=24,
                        help="Time window in hours (default: 24)")
    parser.add_argument("--min-weight", type=int, default=3,
                        help="Minimum co-change weight to keep (default: 3)")
    parser.add_argument("--author-boost", type=float, default=2.0,
                        help="Weight multiplier for same-author changes (default: 2.0)")
    args = parser.parse_args()

    if args.artifact_dir:
        out_path = args.artifact_dir / "cross_cochange_index.json"
    else:
        out_path = args.git_history.parent / "cross_cochange_index.json"

    commits = load_commits(args.git_history)

    if not commits:
        print("ERROR: No commits loaded. Check git_history.json path.", flush=True)
        return

    cochange = build_cross_cochange(commits, args.time_window, args.author_boost)

    total_commits = len(commits)
    modules, pairs = write_index(cochange, args.min_weight, out_path,
                                 total_commits, args.time_window)

    # Summary
    print(f"\n{'='*60}")
    print(f"Cross-Repo Co-Change Summary")
    print(f"{'='*60}")
    print(f"  Time window:    ±{args.time_window}h")
    print(f"  Min weight:     {args.min_weight}")
    print(f"  Author boost:   {args.author_boost}x")
    print(f"  Modules:        {modules:,}")
    print(f"  Pairs:          {pairs:,}")


if __name__ == "__main__":
    main()
