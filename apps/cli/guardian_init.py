#!/usr/bin/env python3
"""
guardian_init.py — Zero-config Guardian setup for any git repo.

Exports git history, builds co-change and ownership indexes.
No AST parsing, no GPU, no LLM, no config file needed.
Works with ANY programming language.

Usage:
  # Initialize Guardian in the current repo
  python3 guardian_init.py

  # Initialize for a specific repo
  python3 guardian_init.py --repo /path/to/repo

  # Custom artifact directory
  python3 guardian_init.py --artifact-dir .guardian/

  # Then run Guardian analysis:
  python3 pr_analyzer.py --mode guardian --files changed_file.py --artifact-dir .ripple/artifacts
"""
import argparse, json, os, pathlib, subprocess, sys, time

_REPO = pathlib.Path(__file__).parent.parent.parent  # apps/cli → apps → repo root
sys.path.insert(0, str(_REPO / "build"))

def main():
    parser = argparse.ArgumentParser(description="Zero-config Guardian setup")
    parser.add_argument("--repo", type=pathlib.Path, default=".",
                        help="Path to git repo (default: current directory)")
    parser.add_argument("--artifact-dir", type=pathlib.Path, default=None,
                        help="Where to store indexes (default: <repo>/.ripple/artifacts)")
    parser.add_argument("--min-weight", type=int, default=None,
                        help="Min co-change weight (default: auto)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild even if indexes exist")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        print(f"Error: {repo} is not a git repository", file=sys.stderr)
        sys.exit(1)

    artifact_dir = args.artifact_dir or (repo / ".ripple" / "artifacts")
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    git_history_path = artifact_dir.parent / "git_history.json"
    cochange_path = artifact_dir / "cochange_index.json"
    ownership_path = artifact_dir / "ownership_index.json"

    print(f"Guardian Init — {repo.name}")
    print(f"Artifacts: {artifact_dir}\n")
    t0 = time.time()

    # Step 1: Export git history
    if git_history_path.exists() and not args.rebuild:
        print(f"[skip] Git history exists: {git_history_path}")
    else:
        print("[1/3] Exporting git history...")
        _run_build_script("00_export_git_history.py", [str(repo), str(git_history_path)])

    # Step 2: Build co-change index
    if cochange_path.exists() and not args.rebuild:
        print(f"[skip] Co-change index exists: {cochange_path}")
    else:
        print("[2/3] Building co-change index...")
        cmd = ["--git-history", str(git_history_path), "--artifact-dir", str(artifact_dir)]
        if args.min_weight is not None:
            cmd += ["--min-weight", str(args.min_weight)]
        _run_build_script("06_build_cochange.py", cmd)

    # Step 3: Build ownership index
    if ownership_path.exists() and not args.rebuild:
        print(f"[skip] Ownership index exists: {ownership_path}")
    else:
        print("[3/3] Building ownership index...")
        _run_build_script("08_build_ownership.py",
                          ["--from-json", str(git_history_path),
                           "--artifact-dir", str(artifact_dir)])

    elapsed = time.time() - t0
    print(f"\nGuardian initialized in {elapsed:.1f}s")

    # Summary
    for name, path in [("Co-change", cochange_path), ("Ownership", ownership_path)]:
        if path.exists():
            data = json.loads(path.read_text())
            meta = data.get("meta", {})
            modules = meta.get("total_modules", len(data.get("edges", data.get("modules", {}))))
            print(f"  {name}: {modules} modules")

    # Create a minimal graph stub so retrieval_engine can initialize
    graph_path = artifact_dir / "graph_with_summaries.json"
    if not graph_path.exists():
        stub = {
            "nodes": [], "edges": [], "clusters": {},
            "networkx": {"directed": True, "multigraph": False,
                         "graph": {}, "nodes": [], "edges": []},
            "stats": {"total_nodes": 0, "total_edges": 0},
        }
        graph_path.write_text(json.dumps(stub))
        print("  Graph: stub (run full pipeline for import analysis)")

    print(f"\nReady! Run Guardian with:")
    print(f"  python3 {_REPO}/apps/cli/pr_analyzer.py \\")
    print(f"    --mode guardian --artifact-dir {artifact_dir} \\")
    print(f"    --files <changed_file1> <changed_file2>")

    # Check for Lore decision-record trailers (arXiv 2603.15566)
    # If none found in recent commits, suggest adoption for richer get_why_context output
    try:
        lore_check = subprocess.run(
            ["git", "log", "--max-count=100", "--format=%B"],
            cwd=str(repo), capture_output=True, text=True, timeout=5
        )
        has_lore = any(
            trailer in lore_check.stdout
            for trailer in ("Lore-Constraint:", "Lore-Directive:", "Lore-Rejected:", "Lore-Verify:")
        )
        if not has_lore:
            print("\nTip: No Lore decision-record trailers found in recent commits.")
            print("  Add 'Lore-Constraint: <why>' to commit messages to enrich get_why_context.")
            print("  See arXiv 2603.15566 or set HR_LORE_PATH to enable in the MCP server.")
    except Exception:
        pass


def _run_build_script(script: str, args: list):
    """Run a build script with the appropriate Python."""
    script_path = _REPO / "build" / script
    if not script_path.exists():
        print(f"  Error: {script_path} not found", file=sys.stderr)
        return

    python = sys.executable
    result = subprocess.run(
        [python, str(script_path)] + args,
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  Error running {script}:\n{result.stderr[:500]}", file=sys.stderr)
    # Print last few lines of output for progress
    lines = result.stdout.strip().split("\n")
    for line in lines[-3:]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
