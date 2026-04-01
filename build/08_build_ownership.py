"""
Stage 8 — Build module ownership index from git history.

For each module, tracks which authors have changed it and how often.
Used by Guardian Mode "Suggested Reviewers" to recommend PR reviewers
based on who has the most context on affected modules.

Streams commits from git_history.json using ijson (O(1) memory).
Handles truncated/incomplete JSON files gracefully — writes partial results.
Output: ownership_index.json in artifact dir.
"""
import json, pathlib, sys
from collections import defaultdict

try:
    import ijson
except ImportError:
    raise SystemExit("pip install ijson")

try:
    import yaml
    CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.yaml"
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
except ImportError:
    cfg = {}

# Resolve paths from config or defaults
GIT_HISTORY = pathlib.Path(cfg.get("git_history_path",
    "/home/beast/projects/workspaces/juspay/git_history.json"))
ARTIFACT_DIR = pathlib.Path(cfg.get("artifact_dir",
    "/home/beast/projects/workspaces/juspay/artifacts"))
OUT_PATH = ARTIFACT_DIR / "ownership_index.json"

SRC_EXTS  = {".hs", ".rs", ".hs-boot", ".py", ".ts", ".js", ".tsx", ".jsx", ".groovy"}
SKIP_DIRS = {".stack-work", "node_modules", "__pycache__", ".git", "dist", "build"}
MAX_FILES = 40  # skip mega-commits (same as cochange)


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
    # module -> author_email -> commit_count
    ownership = defaultdict(lambda: defaultdict(int))
    # author_email -> author_name (keep latest name)
    author_names = {}
    total_commits = 0
    skipped = 0
    current_repo = None
    truncated = False

    print(f"Streaming {GIT_HISTORY}...", flush=True)

    with open(GIT_HISTORY, "rb") as f:
        parser = ijson.parse(f, use_float=True)
        in_commit = False
        commit_files = []
        commit_author_email = ""
        commit_author_name = ""

        try:
            for prefix, event, value in parser:
                # Track current repo name
                if prefix == "repositories.item.name" and event == "string":
                    current_repo = value
                    continue

                # Detect commit start
                if prefix == "repositories.item.commits.item" and event == "start_map":
                    in_commit = True
                    commit_files = []
                    commit_author_email = ""
                    commit_author_name = ""
                    continue

                # Collect author info
                if in_commit:
                    if prefix == "repositories.item.commits.item.author_email" and event == "string":
                        commit_author_email = value.lower()
                        continue
                    if prefix == "repositories.item.commits.item.author_name" and event == "string":
                        commit_author_name = value
                        continue

                # Collect file paths
                if in_commit and event == "string" and "files_changed" in prefix and prefix.endswith(".path"):
                    if is_source(value):
                        commit_files.append(value)
                    continue

                # Commit ends — process it
                if prefix == "repositories.item.commits.item" and event == "end_map":
                    in_commit = False
                    total_commits += 1

                    if commit_author_email:
                        author_names[commit_author_email] = commit_author_name

                    if (commit_author_email and current_repo
                            and 1 <= len(commit_files) <= MAX_FILES):
                        for fp in commit_files:
                            mod = to_module(current_repo, fp)
                            ownership[mod][commit_author_email] += 1
                    elif len(commit_files) > MAX_FILES:
                        skipped += 1

                    if total_commits % 5000 == 0:
                        print(f"  {total_commits:,} commits  {len(ownership):,} modules  "
                              f"{len(author_names):,} authors  repo={current_repo}",
                              flush=True)
                    continue

        except Exception as exc:
            truncated = True
            print(f"\nWARNING: JSON parsing stopped at {total_commits:,} commits: {exc}",
                  flush=True)
            print("Writing partial results...", flush=True)

    status = "PARTIAL (file truncated)" if truncated else "COMPLETE"
    print(f"\n{status}: {total_commits:,} commits processed  {skipped} mega-commits skipped",
          flush=True)
    print(f"Unique modules: {len(ownership):,}  Unique authors: {len(author_names):,}",
          flush=True)

    # Build final index: for each module, top 10 authors sorted by commit count
    TOP_AUTHORS = 10
    index = {}
    for mod, authors in ownership.items():
        sorted_authors = sorted(authors.items(), key=lambda x: -x[1])[:TOP_AUTHORS]
        index[mod] = [
            {
                "email": email,
                "name": author_names.get(email, email.split("@")[0]),
                "commits": count,
            }
            for email, count in sorted_authors
        ]

    output = {
        "meta": {
            "total_commits": total_commits,
            "truncated": truncated,
            "total_modules": len(index),
            "total_unique_authors": len(author_names),
        },
        "authors": dict(author_names),
        "modules": index,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, separators=(",", ":")))
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nWritten: {OUT_PATH}  ({size_mb:.1f}MB)", flush=True)

    # Sample output
    print("\nSample module ownership:")
    for mod, authors in list(index.items())[:5]:
        print(f"  {mod}")
        for a in authors[:3]:
            print(f"    {a['name']} ({a['email']}): {a['commits']} commits")


if __name__ == "__main__":
    build()
