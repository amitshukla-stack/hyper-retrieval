"""
10_build_activity.py — Build per-module activity index from git history.

For each module, computes:
  - activity_50:  number of commits touching this module in the last 50 commits (per repo)
  - activity_200: number of commits touching this module in the last 200 commits (per repo)
  - last_commit_idx: index of the most recent commit touching this module

The activity index enables blast_radius v2 to rerank by recent activity,
which exp_003 showed is the dominant signal for K=3 change prediction
(79-84% feature importance across codebases).

Input:  git_history.json (from 00_export_git_history.py)
Output: activity_index.json → artifact_dir
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def file_to_module(filepath: str) -> str | None:
    """Convert file path to module name (dot-separated, no extension)."""
    if not filepath.endswith(".hs"):
        return None
    base = filepath[:-3]
    parts = base.split("/")
    # Strip common build prefixes
    prefixes = {"src", "lib", "app", "common", "test", "tests", "bench"}
    while parts and parts[0].lower() in prefixes:
        parts = parts[1:]
    if not parts:
        return None
    return ".".join(parts)


def build_activity_index(git_history_path: str, artifact_dir: str):
    with open(git_history_path) as f:
        data = json.load(f)

    repos = data.get("repositories", [])
    print(f"[load] {len(repos)} repos")

    # Per-module activity counters
    module_activity = defaultdict(lambda: {
        "activity_50": 0, "activity_200": 0, "last_commit_idx": -1,
        "total_commits": 0, "repo": "",
    })

    total_commits = 0
    for repo in repos:
        rname = repo["name"]
        commits = sorted(repo["commits"], key=lambda c: c["date"])
        n = len(commits)
        total_commits += n

        for local_idx, commit in enumerate(commits):
            files = [fc["path"] for fc in commit.get("files_changed", []) if fc.get("path")]
            modules_in_commit = set()
            for f in files:
                mod = file_to_module(f)
                if mod:
                    modules_in_commit.add(mod)

            for mod in modules_in_commit:
                entry = module_activity[mod]
                entry["repo"] = rname
                entry["total_commits"] += 1
                entry["last_commit_idx"] = local_idx
                if local_idx >= n - 50:
                    entry["activity_50"] += 1
                if local_idx >= n - 200:
                    entry["activity_200"] += 1

    # Normalize activity scores to [0, 1]
    max_a50 = max((e["activity_50"] for e in module_activity.values()), default=1)
    max_a200 = max((e["activity_200"] for e in module_activity.values()), default=1)

    output = {}
    for mod, entry in module_activity.items():
        output[mod] = {
            "activity_50": entry["activity_50"],
            "activity_200": entry["activity_200"],
            "activity_score": round(
                0.6 * entry["activity_50"] / max(max_a50, 1) +
                0.4 * entry["activity_200"] / max(max_a200, 1),
                4
            ),
            "total_commits": entry["total_commits"],
            "repo": entry["repo"],
        }

    out_path = os.path.join(artifact_dir, "activity_index.json")
    with open(out_path, "w") as f:
        json.dump(output, f)

    print(f"[write] {out_path}")
    print(f"  {len(output)} modules indexed from {total_commits} commits")
    print(f"  max activity_50: {max_a50}, max activity_200: {max_a200}")

    # Show top 10 most active
    top = sorted(output.items(), key=lambda x: -x[1]["activity_score"])[:10]
    print(f"\n  Top 10 most active modules:")
    for mod, info in top:
        print(f"    {mod}: score={info['activity_score']} a50={info['activity_50']} a200={info['activity_200']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-history", default=None,
                        help="Path to git_history.json")
    parser.add_argument("--artifact-dir", default=None,
                        help="Output directory for activity_index.json")
    args = parser.parse_args()

    git_history = args.git_history or os.environ.get(
        "GIT_HISTORY", str(Path.home() / "projects/workspaces/juspay/git_history.json"))
    artifact_dir = args.artifact_dir or os.environ.get(
        "ARTIFACT_DIR", str(Path.home() / "projects/workspaces/juspay/artifacts"))

    build_activity_index(git_history, artifact_dir)
