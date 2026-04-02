"""
Stage 0 — Export git history from all source repos into a single JSON file.

Produces the format consumed by 06_build_cochange.py and 09_build_granger.py:
{
  "repositories": [
    {
      "name": "repo-name",
      "commits": [
        {
          "hash": "abc123",
          "date": "2024-01-15",
          "files_changed": [{"path": "src/Module/File.hs"}]
        }
      ]
    }
  ]
}

Streams git log per repo. No external dependencies.

Usage:
  python3 build/00_export_git_history.py [source_dir] [output_file]
  python3 build/00_export_git_history.py /path/to/source /path/to/git_history.json
"""
import json, os, pathlib, subprocess, sys, time

SOURCE_DIR = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1
    else "/home/beast/projects/workspaces/juspay/source"
)
OUTPUT = pathlib.Path(
    sys.argv[2] if len(sys.argv) > 2
    else "/home/beast/projects/workspaces/juspay/git_history.json"
)

# Separator unlikely to appear in commit data
SEP = "<<<COMMIT_SEP>>>"
FIELD_SEP = "<<<FIELD>>>"


def export_repo(repo_path: pathlib.Path) -> list[dict]:
    """Export all commits from a repo using git log."""
    # Use git log with a custom format to get hash, date, and files
    # --name-only gives us changed file paths
    # --diff-filter=AMRC excludes deleted files
    result = subprocess.run(
        ["git", "log", "--all", "--pretty=format:" + SEP + "%H" + FIELD_SEP + "%aI" + FIELD_SEP + "%aN" + FIELD_SEP + "%aE",
         "--name-only", "--diff-filter=AMRC"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        print(f"  WARNING: git log failed for {repo_path.name}: {result.stderr[:200]}", flush=True)
        return []

    commits = []
    raw = result.stdout

    for chunk in raw.split(SEP):
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = chunk.split("\n")
        header = lines[0]

        if FIELD_SEP not in header:
            continue

        parts = header.split(FIELD_SEP)
        commit_hash = parts[0].strip()
        date = parts[1].strip() if len(parts) > 1 else ""
        author_name = parts[2].strip() if len(parts) > 2 else ""
        author_email = parts[3].strip() if len(parts) > 3 else ""

        files = []
        for line in lines[1:]:
            line = line.strip()
            if line and not line.startswith("commit "):
                files.append({"path": line})

        if files:
            commits.append({
                "hash": commit_hash,
                "date": date,
                "author_name": author_name,
                "author_email": author_email,
                "files_changed": files,
            })

    return commits


def main():
    print(f"Source directory: {SOURCE_DIR}", flush=True)
    print(f"Output file: {OUTPUT}", flush=True)

    # Find all git repos in source dir (subdirectories with .git)
    repos = sorted([
        d for d in SOURCE_DIR.iterdir()
        if d.is_dir() and (d / ".git").exists()
    ])

    # If source_dir itself is a git repo and has no sub-repos, use it directly
    if not repos and (SOURCE_DIR / ".git").exists():
        repos = [SOURCE_DIR]

    print(f"Found {len(repos)} git repositories\n", flush=True)

    repositories = []
    total_commits = 0
    t0 = time.time()

    for repo_path in repos:
        print(f"Exporting {repo_path.name}...", end=" ", flush=True)
        commits = export_repo(repo_path)
        total_commits += len(commits)
        print(f"{len(commits):,} commits", flush=True)

        repositories.append({
            "name": repo_path.name,
            "commits": commits,
        })

    elapsed = time.time() - t0

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({"repositories": repositories}, f)

    file_size = OUTPUT.stat().st_size / (1024 * 1024)
    print(f"\nDone in {elapsed:.1f}s", flush=True)
    print(f"Total: {len(repos)} repos, {total_commits:,} commits", flush=True)
    print(f"Output: {OUTPUT} ({file_size:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
