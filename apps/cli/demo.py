#!/usr/bin/env python3
"""
hr demo <github_url_or_local_path>

One-command demo: clone a repo, analyze git history, render an HTML blast-radius
report. No GPU, no embedding server, no config. Works on any public GitHub repo.

Output: hr-demo-report.html (open in browser)
Runtime: ~30-60 seconds on a typical repo
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def git_log_cochange(repo_path: str, max_commits: int = 2000) -> dict:
    """
    Parse git log and compute co-change pairs.
    Returns: {file_a: {file_b: weight}} where weight = # commits where both changed.
    """
    result = subprocess.run(
        ["git", "log", f"--max-count={max_commits}", "--name-only", "--pretty=format:COMMIT"],
        cwd=repo_path, capture_output=True, text=True, timeout=60
    )
    cochange: dict = collections.defaultdict(lambda: collections.defaultdict(int))
    current_files: list = []
    for line in result.stdout.splitlines():
        if line == "COMMIT":
            for i, fa in enumerate(current_files):
                for fb in current_files[i+1:]:
                    cochange[fa][fb] += 1
                    cochange[fb][fa] += 1
            current_files = []
        elif line.strip():
            current_files.append(line.strip())
    return {k: dict(v) for k, v in cochange.items()}


def blast_radius_score(file: str, cochange: dict) -> float:
    """Sum of co-change weights for a file = its blast radius score."""
    neighbors = cochange.get(file, {})
    return sum(neighbors.values())


def top_files(cochange: dict, n: int = 15) -> list:
    files = list(cochange.keys())
    scored = [(f, blast_radius_score(f, cochange)) for f in files]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:n]


def render_html(repo_name: str, top: list, cochange: dict, commit_count: int) -> str:
    rows = ""
    for rank, (file, score) in enumerate(top, 1):
        neighbors = sorted(cochange.get(file, {}).items(), key=lambda x: x[1], reverse=True)[:5]
        nbr_html = " ".join(
            f'<span class="nbr">{n} <span class="w">×{w}</span></span>'
            for n, w in neighbors
        )
        rows += f"""
        <tr>
          <td class="rank">{rank}</td>
          <td class="file">{file}</td>
          <td class="score">{score}</td>
          <td class="nbrs">{nbr_html}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>HyperRetrieval Demo — {repo_name}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; }}
    h1 {{ font-size: 1.6rem; margin-bottom: 4px; }}
    .sub {{ color: #666; font-size: 0.9rem; margin-bottom: 24px; }}
    .cta {{ background: #0070f3; color: white; padding: 10px 20px; border-radius: 6px;
            text-decoration: none; font-size: 0.9rem; display: inline-block; margin-bottom: 32px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid #e5e7eb;
          color: #6b7280; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #f3f4f6; vertical-align: top; }}
    tr:hover td {{ background: #f9fafb; }}
    .rank {{ color: #9ca3af; width: 32px; }}
    .file {{ font-family: 'SF Mono', 'Monaco', monospace; color: #111; font-size: 0.8rem; }}
    .score {{ color: #dc2626; font-weight: 700; width: 70px; }}
    .nbr {{ display: inline-block; background: #eff6ff; border: 1px solid #bfdbfe;
            border-radius: 4px; padding: 2px 6px; margin: 2px; font-family: monospace;
            font-size: 0.75rem; color: #1d4ed8; }}
    .w {{ color: #6b7280; font-size: 0.7rem; }}
    .legend {{ font-size: 0.8rem; color: #6b7280; margin-top: 24px; }}
    .badge {{ display: inline-block; background: #ecfdf5; color: #059669;
              border: 1px solid #a7f3d0; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; }}
  </style>
</head>
<body>
  <h1>🔍 HyperRetrieval — Blast Radius Report</h1>
  <div class="sub">
    <strong>{repo_name}</strong> &nbsp;·&nbsp;
    {commit_count} commits analyzed &nbsp;·&nbsp;
    <span class="badge">zero GPU · no config</span>
  </div>
  <a class="cta" href="https://github.com/Amitshukla2308/Index-the-code">
    Get full HyperRetrieval (15 MCP tools) →
  </a>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>File</th>
        <th>Blast Score</th>
        <th>Top co-change neighbors (if you change this, these also changed historically)</th>
      </tr>
    </thead>
    <tbody>{rows}
    </tbody>
  </table>
  <p class="legend">
    <strong>Blast score</strong> = sum of co-change weights across {commit_count} commits.
    Higher = more central to the codebase. Change these files carefully.
    <br>Full HyperRetrieval adds: criticality scoring, Guard static checks, cross-repo signals,
    semantic search, and 15 MCP tools for AI coding agents.
  </p>
</body>
</html>"""


def clone_repo(url: str, target: str) -> str:
    print(f"Cloning {url}...")
    subprocess.run(["git", "clone", "--depth=500", url, target],
                   check=True, capture_output=True, timeout=120)
    return target


def commit_count(repo_path: str) -> int:
    r = subprocess.run(["git", "rev-list", "--count", "HEAD"],
                       cwd=repo_path, capture_output=True, text=True, timeout=30)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="HyperRetrieval demo — instant blast-radius report from git history"
    )
    parser.add_argument("target", help="GitHub URL or local repo path")
    parser.add_argument("--output", default="hr-demo-report.html", help="Output HTML file")
    parser.add_argument("--commits", type=int, default=2000, help="Max commits to analyze")
    args = parser.parse_args()

    tmpdir = None
    repo_path = args.target

    if args.target.startswith("http"):
        tmpdir = tempfile.mkdtemp(prefix="hr-demo-")
        repo_path = os.path.join(tmpdir, "repo")
        clone_repo(args.target, repo_path)
        repo_name = args.target.rstrip("/").split("/")[-1].replace(".git", "")
    else:
        repo_path = os.path.abspath(args.target)
        repo_name = Path(repo_path).name

    print(f"Analyzing git history ({args.commits} commits)...")
    cochange = git_log_cochange(repo_path, max_commits=args.commits)
    n_commits = min(commit_count(repo_path), args.commits)

    print(f"Computing blast radius for {len(cochange)} files...")
    top = top_files(cochange, n=15)

    html = render_html(repo_name, top, cochange, n_commits)
    with open(args.output, "w") as f:
        f.write(html)

    print(f"\n✅ Report written to: {args.output}")
    print(f"   Top blast-radius file: {top[0][0]} (score {top[0][1]})" if top else "")
    print(f"   Open in browser: file://{os.path.abspath(args.output)}")

    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
