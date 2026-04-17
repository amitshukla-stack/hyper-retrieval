"""
Stage 10 — Criticality Scoring

Computes a criticality score (0-1) for every indexed module from signals
that already exist in the build artifacts. No domain knowledge required.

Signals (all derived from git history + code structure):
  1. Blast radius: co-change coupling count (more coupled = more critical)
  2. Cross-repo coupling: modules that co-change across repos
  3. Change frequency: how often this module changes (high churn = fragile)
  4. Author concentration: few authors = bus factor risk
  5. Recency: recently changed code is more likely to have fresh bugs
  6. Granger causal influence: modules that CAUSE changes in others
  7. Revert signals: code that has been reverted (from commit messages)

Output: ARTIFACT_DIR/criticality_index.json
  {
    "meta": {...},
    "modules": {
      "module_name": {
        "score": 0.92,
        "rank": 1,
        "signals": {
          "blast_radius": 0.95,
          "cross_repo_coupling": 0.88,
          "change_frequency": 0.72,
          "author_concentration": 0.91,
          "recency": 0.65,
          "granger_influence": 0.80,
          "revert_risk": 0.30
        },
        "reasons": ["High blast radius (47 co-change neighbors)", ...]
      }
    }
  }

Input: cochange_index.json, cross_cochange_index.json, ownership_index.json,
       granger_index.json, git_history.json, graph_with_summaries.json
"""
import json, os, pathlib, math, re
from collections import defaultdict
from datetime import datetime

ARTIFACT_DIR = pathlib.Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
OUTPUT_DIR = pathlib.Path(os.environ.get("OUTPUT_DIR", "output"))
GIT_HISTORY = pathlib.Path(os.environ.get("GIT_HISTORY", "git_history.json"))

# Signal weights (tunable)
WEIGHTS = {
    "blast_radius": 0.25,
    "cross_repo_coupling": 0.15,
    "change_frequency": 0.15,
    "author_concentration": 0.15,
    "recency": 0.10,
    "granger_influence": 0.10,
    "revert_risk": 0.10,
}


def normalize_scores(values: dict) -> dict:
    """Min-max normalize a dict of {key: float} to [0, 1]."""
    if not values:
        return {}
    vals = list(values.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def load_json(path):
    """Load JSON, return empty dict on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  WARNING: Could not load {path}: {e}")
        return {}


def compute_blast_radius_scores(cochange, cross_cochange):
    """Score based on number of co-change neighbors (within + cross repo)."""
    print("  Computing blast radius scores...")
    neighbor_counts = defaultdict(int)

    # Within-repo co-change
    edges = cochange.get("edges", cochange) if isinstance(cochange, dict) else {}
    for mod, neighbors in edges.items():
        if isinstance(neighbors, list):
            neighbor_counts[mod] += len(neighbors)
        elif isinstance(neighbors, dict):
            neighbor_counts[mod] += len(neighbors)

    # Cross-repo co-change
    if isinstance(cross_cochange, dict):
        cc_edges = cross_cochange.get("edges", cross_cochange)
        for mod, neighbors in cc_edges.items():
            if isinstance(neighbors, (list, dict)):
                neighbor_counts[mod] += len(neighbors) if isinstance(neighbors, list) else len(neighbors)

    print(f"    {len(neighbor_counts)} modules with co-change neighbors")
    return normalize_scores(dict(neighbor_counts))


def compute_cross_repo_scores(cross_cochange):
    """Score based on cross-repo coupling specifically."""
    print("  Computing cross-repo coupling scores...")
    coupling = defaultdict(float)

    if isinstance(cross_cochange, dict):
        edges = cross_cochange.get("edges", cross_cochange)
        for mod, neighbors in edges.items():
            if isinstance(neighbors, dict):
                coupling[mod] = sum(neighbors.values())
            elif isinstance(neighbors, list):
                coupling[mod] = len(neighbors)

    print(f"    {len(coupling)} modules with cross-repo coupling")
    return normalize_scores(dict(coupling))


def compute_change_frequency(git_history):
    """Score based on how often files in each module change."""
    print("  Computing change frequency scores...")
    change_counts = defaultdict(int)

    repos = git_history.get("repositories", [])
    for repo in repos:
        repo_name = repo.get("name", "")
        for commit in repo.get("commits", []):
            files = commit.get("files_changed", commit.get("files", []))
            for fobj in files:
                fp = fobj if isinstance(fobj, str) else fobj.get("path", fobj.get("file", ""))
                if fp:
                    # Convert to module-like key
                    mod_key = f"{repo_name}::{fp.replace('/', '::').rsplit('.', 1)[0]}"
                    change_counts[mod_key] += 1

    print(f"    {len(change_counts)} file-modules with change data")
    return normalize_scores(dict(change_counts))


def compute_author_concentration(ownership):
    """Score based on bus factor — fewer authors = higher risk."""
    print("  Computing author concentration scores...")
    concentration = {}

    modules = ownership.get("modules", ownership)
    for mod, data in modules.items():
        if isinstance(data, list):
            authors = data
        elif isinstance(data, dict):
            authors = data.get("authors", [])
        else:
            continue

        n_authors = len(authors) if authors else 1
        # Inverse: fewer authors = higher concentration = higher risk
        # 1 author = 1.0, 2 = 0.5, 5 = 0.2, 10+ = ~0.1
        concentration[mod] = 1.0 / n_authors

    print(f"    {len(concentration)} modules with ownership data")
    return normalize_scores(concentration)


def compute_recency(git_history):
    """Score based on how recently the module was changed."""
    print("  Computing recency scores...")
    last_change = {}

    repos = git_history.get("repositories", [])
    for repo in repos:
        repo_name = repo.get("name", "")
        for commit in repo.get("commits", []):
            date_str = commit.get("date", "")
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                ts = dt.timestamp()
            except (ValueError, AttributeError):
                continue

            files = commit.get("files_changed", commit.get("files", []))
            for fobj in files:
                fp = fobj if isinstance(fobj, str) else fobj.get("path", fobj.get("file", ""))
                if fp:
                    mod_key = f"{repo_name}::{fp.replace('/', '::').rsplit('.', 1)[0]}"
                    if mod_key not in last_change or ts > last_change[mod_key]:
                        last_change[mod_key] = ts

    print(f"    {len(last_change)} file-modules with recency data")
    return normalize_scores(last_change)


def compute_granger_influence(granger):
    """Score based on causal influence — modules that cause changes in many others."""
    print("  Computing Granger influence scores...")
    influence = defaultdict(int)

    pairs = granger.get("causal_pairs", granger)
    if isinstance(pairs, dict):
        for pair_key, data in pairs.items():
            source = data.get("source", pair_key.split("→")[0] if "→" in pair_key else "")
            if source:
                influence[source] += 1
    elif isinstance(pairs, list):
        for pair in pairs:
            source = pair.get("source", "")
            if source:
                influence[source] += 1

    print(f"    {len(influence)} modules with causal influence")
    return normalize_scores(dict(influence))


def compute_revert_risk(git_history):
    """Score based on revert signals in commit messages."""
    print("  Computing revert risk scores...")
    revert_counts = defaultdict(int)
    revert_pattern = re.compile(r"\brevert\b|\brollback\b|\bundo\b|\bbacked out\b", re.IGNORECASE)

    repos = git_history.get("repositories", [])
    for repo in repos:
        repo_name = repo.get("name", "")
        for commit in repo.get("commits", []):
            msg = commit.get("message", commit.get("msg", ""))
            if revert_pattern.search(msg):
                files = commit.get("files_changed", commit.get("files", []))
                for fobj in files:
                    fp = fobj if isinstance(fobj, str) else fobj.get("path", fobj.get("file", ""))
                    if fp:
                        mod_key = f"{repo_name}::{fp.replace('/', '::').rsplit('.', 1)[0]}"
                        revert_counts[mod_key] += 1

    print(f"    {len(revert_counts)} file-modules with revert history")
    return normalize_scores(dict(revert_counts))


def main():
    print("=" * 60)
    print("Stage 10: Criticality Scoring")
    print("=" * 60)

    # Load all signal sources
    print("\nLoading signal sources...")
    cochange = load_json(ARTIFACT_DIR / "cochange_index.json")
    cross_cochange = load_json(ARTIFACT_DIR / "cross_cochange_index.json")
    ownership = load_json(ARTIFACT_DIR / "ownership_index.json")
    granger = load_json(ARTIFACT_DIR / "granger_index.json")

    print(f"Loading git history ({GIT_HISTORY})...")
    git_history = load_json(GIT_HISTORY)

    # Compute individual signal scores
    print("\nComputing signal scores...")
    signals = {
        "blast_radius": compute_blast_radius_scores(cochange, cross_cochange),
        "cross_repo_coupling": compute_cross_repo_scores(cross_cochange),
        "change_frequency": compute_change_frequency(git_history),
        "author_concentration": compute_author_concentration(ownership),
        "recency": compute_recency(git_history),
        "granger_influence": compute_granger_influence(granger),
        "revert_risk": compute_revert_risk(git_history),
    }

    # Collect all module names
    all_modules = set()
    for sig_scores in signals.values():
        all_modules.update(sig_scores.keys())
    print(f"\n  Total unique modules across all signals: {len(all_modules)}")

    # Compute composite criticality score
    print("\nComputing composite scores...")
    modules = {}
    for mod in all_modules:
        mod_signals = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for sig_name, sig_scores in signals.items():
            score = sig_scores.get(mod, 0.0)
            mod_signals[sig_name] = round(score, 4)
            weighted_sum += score * WEIGHTS[sig_name]
            total_weight += WEIGHTS[sig_name]

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Generate human-readable reasons for top signals
        reasons = []
        for sig_name in sorted(mod_signals, key=lambda s: mod_signals[s], reverse=True):
            if mod_signals[sig_name] > 0.5:
                label = sig_name.replace("_", " ").title()
                reasons.append(f"{label}: {mod_signals[sig_name]:.2f}")

        modules[mod] = {
            "score": round(composite, 4),
            "signals": mod_signals,
            "reasons": reasons[:5],
        }

    # Rank modules
    ranked = sorted(modules.keys(), key=lambda m: modules[m]["score"], reverse=True)
    for rank, mod in enumerate(ranked, 1):
        modules[mod]["rank"] = rank

    # Output
    output = {
        "meta": {
            "total_modules": len(modules),
            "signal_weights": WEIGHTS,
            "top_10": [{"module": m, "score": modules[m]["score"]} for m in ranked[:10]],
        },
        "modules": modules,
    }

    out_path = ARTIFACT_DIR / "criticality_index.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    size_mb = out_path.stat().st_size / 1e6
    print(f"\n{'=' * 60}")
    print(f"Criticality Index: {len(modules)} modules scored")
    print(f"Output: {out_path} ({size_mb:.1f} MB)")
    print(f"\nTop 10 critical modules:")
    for i, mod in enumerate(ranked[:10], 1):
        m = modules[mod]
        print(f"  {i:2d}. [{m['score']:.3f}] {mod[:70]}")
        if m["reasons"]:
            print(f"      Reasons: {', '.join(m['reasons'][:3])}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
