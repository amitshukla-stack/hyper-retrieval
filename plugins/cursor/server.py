#!/usr/bin/env python3
"""
HyperRetrieval MCP Server — Cursor Plugin Edition

Lightweight MCP server exposing code impact intelligence tools.
Requires only git-history-based artifacts (co-change + ownership indexes).
No GPU, no embeddings, no AST parsing needed.

Tools exposed:
  - get_blast_radius: tiered impact analysis (will_break/may_break/review)
  - predict_missing_changes: detect files likely missing from a changeset
  - suggest_reviewers: module ownership from git history
"""
import json
import os
import sys
import pathlib

# Add parent paths for imports
_PLUGIN_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _PLUGIN_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "serve"))
sys.path.insert(0, str(_REPO_ROOT))

ARTIFACT_DIR = os.environ.get(
    "HR_ARTIFACT_DIR",
    str(pathlib.Path.cwd() / ".hyperretrieval" / "artifacts")
)
os.environ["ARTIFACT_DIR"] = ARTIFACT_DIR

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: 'mcp' package not installed. Run: pip install mcp[cli]", file=sys.stderr)
    sys.exit(1)

import retrieval_engine as RE

mcp = FastMCP(
    "HyperRetrieval",
    description="Code impact intelligence — blast radius, missing changes, reviewers"
)


def _init():
    """Initialize retrieval engine with available artifacts."""
    artifact_path = pathlib.Path(ARTIFACT_DIR)
    if not artifact_path.exists():
        return False
    try:
        RE.initialize(load_embedder=False)
        return True
    except Exception as e:
        print(f"Init warning: {e}", file=sys.stderr)
        return False


_ready = _init()


@mcp.tool()
def get_blast_radius(files_or_modules: list[str], max_hops: int = 2) -> str:
    """
    Compute blast radius for changed files or modules.

    Returns tiered impact analysis:
    - will_break: direct dependencies (high confidence)
    - may_break: transitive deps or Granger-causal co-changes
    - review: historically co-changed, no static link

    Pass git diff --name-only output directly.

    Args:
        files_or_modules: File paths or module names
        max_hops: Import graph traversal depth (default: 2)
    """
    if not _ready:
        return "HyperRetrieval not initialized. Run the init hook or: python3 init.py --repo ."

    resolved = RE.resolve_files_to_modules(files_or_modules)
    seed_mods = []
    for f, mods in resolved.items():
        if mods:
            seed_mods.extend(mods)
        elif "." in f or "::" in f:
            seed_mods.append(f)

    if not seed_mods:
        return f"Could not resolve inputs to known modules: {files_or_modules}"

    result = RE.get_blast_radius(seed_mods, max_hops=max_hops)

    lines = [f"## Blast Radius Analysis"]
    lines.append(f"**Seed:** {', '.join(result['seed_modules'][:5])}")
    lines.append(f"**Services affected:** {', '.join(result['affected_services']) or 'none'}")

    tiered = result.get("tiered_impact", [])
    if tiered:
        for tier_name, marker in [("will_break", "!"), ("may_break", "?"), ("review", "~")]:
            items = [t for t in tiered if t["tier"] == tier_name]
            if items:
                lines.append(f"\n[{marker}] **{tier_name.upper()}** ({len(items)} modules)")
                for t in items[:8]:
                    sigs = t.get("signals", {})
                    parts = [f"conf={t['confidence']:.2f}"]
                    if "cochange_weight" in sigs:
                        parts.append(f"cc={sigs['cochange_weight']}")
                    if "granger" in sigs:
                        parts.append(f"granger(lag={sigs['granger']['lag']})")
                    svc = f" [{t['service']}]" if t.get("service") else ""
                    lines.append(f"  {t['module']}{svc} ({', '.join(parts)})")
                if len(items) > 8:
                    lines.append(f"  ... and {len(items) - 8} more")

    return "\n".join(lines)


@mcp.tool()
def predict_missing_changes(changed_files: list[str], min_confidence: float = 0.1) -> str:
    """
    Predict files likely MISSING from a changeset.

    Uses co-change history to find modules that typically change together
    with your modifications but aren't in the changeset.

    Args:
        changed_files: Files in the PR/changeset
        min_confidence: Minimum confidence threshold 0-1
    """
    if not _ready:
        return "HyperRetrieval not initialized."

    resolved = RE.resolve_files_to_modules(changed_files)
    mods = []
    for f, m in resolved.items():
        if m:
            mods.extend(m)
        elif "." in f or "::" in f:
            mods.append(f)

    if not mods:
        return f"Could not resolve inputs: {changed_files}"

    result = RE.predict_missing_changes(mods)
    preds = [p for p in result.get("predictions", []) if p["confidence"] >= min_confidence]
    coverage = result.get("coverage_score", 1.0)

    lines = [f"## Missing Change Predictions"]
    lines.append(f"**Coverage score:** {coverage:.0%}")
    lines.append(f"**Predictions:** {len(preds)}")

    for p in preds[:10]:
        causal = f" [causal: {p['causal']['direction']}]" if p.get("causal") else ""
        lines.append(f"  - **{p['module']}** ({p['confidence']:.0%}) — {p['reason']}{causal}")

    if not preds:
        lines.append("No missing changes detected — changeset looks complete.")

    return "\n".join(lines)


@mcp.tool()
def suggest_reviewers(changed_files: list[str], top_k: int = 5) -> str:
    """
    Suggest PR reviewers based on module ownership from git history.

    Returns ranked reviewers with the most context on affected modules.

    Args:
        changed_files: Files being changed
        top_k: Number of reviewers to suggest
    """
    if not _ready:
        return "HyperRetrieval not initialized."

    resolved = RE.resolve_files_to_modules(changed_files)
    mods = []
    for f, m in resolved.items():
        if m:
            mods.extend(m)
        elif "." in f or "::" in f:
            mods.append(f)

    if not mods:
        return f"Could not resolve inputs: {changed_files}"

    result = RE.suggest_reviewers(mods, top_k=top_k)
    reviewers = result.get("reviewers", [])

    lines = ["## Suggested Reviewers"]
    for r in reviewers:
        modules = ", ".join(r["modules"][:3])
        lines.append(f"  - **{r['name']}** ({r['commits']} commits) — {modules}")

    if not reviewers:
        lines.append("No reviewer data available for these modules.")

    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
