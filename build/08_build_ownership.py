"""
Stage 8 — Build module ownership index from git history.

For each module, tracks which authors have changed it and how often.
Used by Guardian Mode "Suggested Reviewers" to recommend PR reviewers
based on who has the most context on affected modules.

Two modes:
  1. --from-repos <dir>  : scan git repos directly (preferred, complete data)
  2. --from-json <file>  : stream git_history.json with ijson (handles truncation)

Output: ownership_index.json in artifact dir.
"""
import argparse, json, os, pathlib, subprocess, sys
from collections import defaultdict

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

SRC_EXTS  = {".hs", ".rs", ".hs-boot", ".py", ".ts", ".js", ".tsx", ".jsx", ".groovy"}
SKIP_DIRS = {".stack-work", "node_modules", "__pycache__", ".git", "dist", "venv", ".venv"}
MAX_FILES = 40


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


def build_from_repos(source_dir: pathlib.Path):
    """Build ownership by running git log on each repo."""
    ownership = defaultdict(lambda: defaultdict(int))
    author_names = {}
    total_commits = 0

    repos = sorted([d for d in source_dir.iterdir() if d.is_dir() and (d / ".git").exists()])
    print(f"Found {len(repos)} git repos in {source_dir}", flush=True)

    for repo_path in repos:
        repo_name = repo_path.name
        print(f"\n  Processing {repo_name}...", flush=True)
        repo_commits = 0

        try:
            result = subprocess.run(
                ["git", "log", "--all", "--format=COMMIT_SEP%n%ae%n%an", "--name-only"],
                cwd=str(repo_path),
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                print(f"    git log failed: {result.stderr[:200]}", flush=True)
                continue

            current_email = ""
            current_name = ""
            current_files = []
            in_commit = False

            for line in result.stdout.split("\n"):
                if line.startswith("COMMIT_SEP"):
                    # Process previous commit
                    if current_email and 1 <= len(current_files) <= MAX_FILES:
                        for fp in current_files:
                            if is_source(fp):
                                mod = to_module(repo_name, fp)
                                ownership[mod][current_email] += 1
                        repo_commits += 1
                    elif current_email:
                        repo_commits += 1

                    current_files = []
                    current_email = ""
                    current_name = ""
                    in_commit = True
                    continue

                if in_commit and not current_email:
                    current_email = line.strip().lower()
                    continue
                if in_commit and not current_name:
                    current_name = line.strip()
                    if current_email:
                        author_names[current_email] = current_name
                    continue

                stripped = line.strip()
                if stripped:
                    current_files.append(stripped)

            # Process last commit
            if current_email and 1 <= len(current_files) <= MAX_FILES:
                for fp in current_files:
                    if is_source(fp):
                        mod = to_module(repo_name, fp)
                        ownership[mod][current_email] += 1
                repo_commits += 1

            total_commits += repo_commits
            repo_mods = sum(1 for m in ownership if m.startswith(f"{repo_name}::"))
            print(f"    {repo_commits:,} commits  {repo_mods:,} modules", flush=True)

        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT on {repo_name} (skipped)", flush=True)
        except Exception as e:
            print(f"    ERROR on {repo_name}: {e}", flush=True)

    return ownership, author_names, total_commits


def build_from_json(json_path: pathlib.Path):
    """Build ownership by streaming git_history.json with ijson."""
    try:
        import ijson
    except ImportError:
        raise SystemExit("pip install ijson (needed for --from-json mode)")

    ownership = defaultdict(lambda: defaultdict(int))
    author_names = {}
    total_commits = 0
    current_repo = None

    print(f"Streaming {json_path}...", flush=True)

    with open(json_path, "rb") as f:
        parser = ijson.parse(f, use_float=True)
        in_commit = False
        commit_files = []
        commit_author_email = ""
        commit_author_name = ""

        try:
            for prefix, event, value in parser:
                if prefix == "repositories.item.name" and event == "string":
                    current_repo = value
                    continue
                if prefix == "repositories.item.commits.item" and event == "start_map":
                    in_commit = True
                    commit_files = []
                    commit_author_email = ""
                    commit_author_name = ""
                    continue
                if in_commit:
                    if prefix == "repositories.item.commits.item.author_email" and event == "string":
                        commit_author_email = value.lower()
                        continue
                    if prefix == "repositories.item.commits.item.author_name" and event == "string":
                        commit_author_name = value
                        continue
                if in_commit and event == "string" and "files_changed" in prefix and prefix.endswith(".path"):
                    if is_source(value):
                        commit_files.append(value)
                    continue
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
                    if total_commits % 5000 == 0:
                        print(f"  {total_commits:,} commits  {len(ownership):,} modules  "
                              f"repo={current_repo}", flush=True)
                    continue
        except Exception as exc:
            print(f"\nWARNING: JSON truncated at {total_commits:,} commits: {exc}", flush=True)

    return ownership, author_names, total_commits


def write_index(ownership, author_names, total_commits, out_path):
    """Write the ownership index to disk."""
    TOP_AUTHORS = 10
    index = {}
    for mod, authors in ownership.items():
        sorted_authors = sorted(authors.items(), key=lambda x: -x[1])[:TOP_AUTHORS]
        index[mod] = [
            {"email": email, "name": author_names.get(email, email.split("@")[0]),
             "commits": count}
            for email, count in sorted_authors
        ]

    output = {
        "meta": {
            "total_commits": total_commits,
            "total_modules": len(index),
            "total_unique_authors": len(author_names),
        },
        "authors": dict(author_names),
        "modules": index,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, separators=(",", ":")))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWritten: {out_path}  ({size_mb:.1f}MB)", flush=True)
    print(f"Total: {total_commits:,} commits  {len(index):,} modules  "
          f"{len(author_names):,} authors", flush=True)

    repos = sorted(set(k.split("::")[0] for k in index))
    print(f"Repos: {', '.join(repos)}", flush=True)

    print("\nSample module ownership:")
    for mod, authors in list(index.items())[:5]:
        print(f"  {mod}")
        for a in authors[:2]:
            print(f"    {a['name']} ({a['email']}): {a['commits']} commits")


def main():
    parser = argparse.ArgumentParser(description="Build module ownership index")
    parser.add_argument("--from-repos", default=None,
                        help="Path to directory containing git repos")
    parser.add_argument("--from-json", default=None,
                        help="Path to git_history.json (ijson streaming)")
    parser.add_argument("--artifact-dir", default=None,
                        help="Output directory for ownership_index.json")
    args = parser.parse_args()

    artifact_dir = pathlib.Path(args.artifact_dir or cfg.get("artifact_dir",
        "/home/beast/projects/workspaces/juspay/artifacts"))
    out_path = artifact_dir / "ownership_index.json"

    if args.from_repos:
        ownership, author_names, total = build_from_repos(pathlib.Path(args.from_repos))
    elif args.from_json:
        ownership, author_names, total = build_from_json(pathlib.Path(args.from_json))
    else:
        # Auto-detect: prefer repos if available
        source_dir = pathlib.Path(cfg.get("source_dir",
            "/home/beast/projects/workspaces/juspay/source"))
        if source_dir.exists():
            print(f"Auto-detected source repos at {source_dir}", flush=True)
            ownership, author_names, total = build_from_repos(source_dir)
        else:
            json_path = pathlib.Path(cfg.get("git_history_path",
                "/home/beast/projects/workspaces/juspay/git_history.json"))
            ownership, author_names, total = build_from_json(json_path)

    write_index(ownership, author_names, total, out_path)


if __name__ == "__main__":
    main()
