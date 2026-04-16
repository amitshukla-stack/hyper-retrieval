"""
bench_blast_radius_recall.py — Measure blast_radius recall against real commits.

Ground truth: multi-file commits from git_history.json. For each commit,
pick the first file as "the change", run blast_radius, check if the other
files from that commit appear in the results.

Metrics:
  - recall@10: fraction of actually-changed files in top-10 blast_radius results
  - recall@20: same at K=20
  - MRR: mean reciprocal rank of first hit
  - tier_accuracy: fraction of hits in "will_break" or "may_break" (vs "review")

Runs keyword-only (no GPU required).

Usage:
    ARTIFACT_DIR=/home/beast/projects/workspaces/juspay/artifacts \
    python3 tests/bench_blast_radius_recall.py [--commits 100] [--output results.json]
"""
import sys, os, pathlib, json, time, argparse

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "serve"))

os.environ["EMBED_SERVER_URL"] = ""  # keyword-only mode

parser = argparse.ArgumentParser()
parser.add_argument("--commits", type=int, default=100, help="Number of test commits")
parser.add_argument("--min-files", type=int, default=3, help="Min files per commit")
parser.add_argument("--max-files", type=int, default=30, help="Max files per commit")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--label", type=str, default="unlabeled", help="Label for this run (e.g. main, v2)")
args = parser.parse_args()

# Load git history for ground truth
GIT_HISTORY = pathlib.Path(os.environ.get(
    "GIT_HISTORY", "/home/beast/projects/workspaces/juspay/git_history.json"))

print(f"Loading git history from {GIT_HISTORY}...")
with open(GIT_HISTORY) as f:
    raw_data = json.load(f)

# Handle nested format: {"repositories": [{"name": ..., "commits": [...]}]}
if isinstance(raw_data, dict) and "repositories" in raw_data:
    raw_commits = []
    for repo in raw_data["repositories"]:
        repo_name = repo.get("name", "")
        for c in repo.get("commits", []):
            c["_repo"] = repo_name
            raw_commits.append(c)
elif isinstance(raw_data, list):
    raw_commits = raw_data
else:
    raw_commits = []
print(f"  {len(raw_commits)} total commits")

# Filter to multi-file commits (the interesting ones for blast radius)
multi_commits = []
for c in raw_commits:
    files = c.get("files_changed", c.get("files", []))
    if args.min_files <= len(files) <= args.max_files:
        repo_name = c.get("_repo", "")
        modules = []
        for fobj in files:
            fp = fobj if isinstance(fobj, str) else fobj.get("path", fobj.get("file", ""))
            if fp and not fp.startswith("."):
                # Store repo_name + path so we can build module keys later
                modules.append({"repo": repo_name, "path": fp})
        if len(modules) >= args.min_files:
            multi_commits.append({
                "hash": c.get("hash", c.get("commit", ""))[:8],
                "date": c.get("date", ""),
                "repo": repo_name,
                "files": modules,
            })

print(f"  {len(multi_commits)} commits with {args.min_files}-{args.max_files} files")

# Sort by date (most recent first) to get temporal mix across repos
multi_commits.sort(key=lambda c: c.get("date", ""), reverse=True)

# Take the first N (most recent)
test_commits = multi_commits[:args.commits]
print(f"  Using {len(test_commits)} most recent commits for evaluation")

# Now load retrieval engine
print("\nLoading retrieval engine (keyword-only)...")
import retrieval_engine as RE
RE.initialize(load_embedder=False)

# Build a reverse lookup: file path -> module name (for matching)
# blast_radius returns module names, git history has file paths
path_to_module = {}
for mod_name, node_ids in RE.file_to_nodes.items():
    path_to_module[mod_name] = mod_name
# Also use filepath_to_module if available
for fp, mod in RE.filepath_to_module.items():
    path_to_module[fp] = mod

print(f"  {len(path_to_module)} path→module mappings")


def resolve_module(file_info) -> str:
    """Resolve a {repo, path} dict to an MG module name.

    blast_radius output uses MG names (dotted Haskell: Euler.API.Gateway.App.Routes).
    We ONLY return MG-format names to ensure seeds and targets are comparable
    to blast_radius output.
    """
    if isinstance(file_info, str):
        repo, path = "", file_info
    else:
        repo, path = file_info.get("repo", ""), file_info.get("path", "")

    # Only handle .hs files — MG coverage is Haskell only
    if not path.endswith(".hs"):
        return None

    # Convert path to dotted module name
    mod = path.replace("/", ".").replace(".hs", "")
    for prefix in ["src.", "app.", "src-generated.", "lib.", "common.src.",
                    "common.src-generated.", "dbTypes.src-generated.",
                    "euler-x.src-generated.", "euler-x.src.",
                    "oltp.src-generated.", "oltp.src.",
                    "euler-api-decider.src.", "euler-api-decider.src-generated."]:
        if mod.startswith(prefix):
            mod = mod[len(prefix):]
            break

    if RE.MG is not None and mod in RE.MG.nodes:
        return mod
    return None


def get_blast_radius_modules(seed_module: str, top_k: int = 20):
    """Run blast_radius and return ranked list of (module, confidence, tier).

    Handles both v1 (import_neighbors/cochange_neighbors lists) and
    v2 (tiered_impact dict with confidence scores) formats.
    """
    try:
        result = RE.get_blast_radius([seed_module])
    except Exception as e:
        return []

    # v2 format: tiered_impact (list of dicts or dict)
    tiered = result.get("tiered_impact", None)
    if tiered:
        ranked = []
        if isinstance(tiered, list):
            for item in tiered:
                ranked.append((item.get("module", ""), item.get("confidence", 0), item.get("tier", "review")))
        elif isinstance(tiered, dict):
            for mod, info in tiered.items():
                ranked.append((mod, info.get("confidence", 0), info.get("tier", "review")))
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]

    # v1 format: flat lists of neighbors
    ranked = []
    seen = set()
    # Import neighbors first (higher signal)
    for nb in result.get("import_neighbors", []):
        mod = nb if isinstance(nb, str) else nb.get("module", nb.get("name", ""))
        if mod and mod not in seen:
            seen.add(mod)
            ranked.append((mod, 0.8, "will_break"))
    # Co-change neighbors second
    for nb in result.get("cochange_neighbors", []):
        mod = nb if isinstance(nb, str) else nb.get("module", nb.get("name", ""))
        if mod and mod not in seen:
            seen.add(mod)
            ranked.append((mod, 0.5, "may_break"))
    return ranked[:top_k]


# Run benchmark
print(f"\n{'='*60}")
print(f"BENCHMARK: Blast Radius Recall [{args.label}]")
print(f"{'='*60}")

results = []
skipped = 0
no_resolve = 0
no_br = 0
t0 = time.time()

# Debug: resolution stats across all test commits
resolve_counts = []
for commit in test_commits:
    n_resolved = sum(1 for f in commit["files"] if resolve_module(f))
    resolve_counts.append(n_resolved)
n_with_2plus = sum(1 for c in resolve_counts if c >= 2)
print(f"  Resolution: {n_with_2plus}/{len(test_commits)} commits have >=2 resolved modules")
print(f"  Avg resolved per commit: {sum(resolve_counts)/len(resolve_counts):.1f}")

# Show first 3 commits with >=2 resolved and their blast_radius output
shown = 0
for commit in test_commits:
    resolved_ok = [(f, resolve_module(f)) for f in commit["files"]]
    resolved_ok = [(f, m) for f, m in resolved_ok if m]
    if len(resolved_ok) >= 2 and shown < 3:
        seed_mod = resolved_ok[0][1]
        br = get_blast_radius_modules(seed_mod, top_k=5)
        targets = set(m for _, m in resolved_ok[1:])
        print(f"  Sample {commit['hash']} ({commit.get('repo','')}):")
        print(f"    seed: {seed_mod}, targets: {list(targets)[:3]}")
        print(f"    blast_radius: {len(br)} results, top: {[m for m,_,_ in br[:3]]}")
        print(f"    hits: {targets & set(m for m,_,_ in br)}")
        shown += 1

for i, commit in enumerate(test_commits):
    files = commit["files"]

    # Resolve all files to modules
    resolved = []
    for f in files:
        mod = resolve_module(f)
        if mod:
            resolved.append((f, mod))

    if len(resolved) < 2:
        skipped += 1
        no_resolve += 1
        continue

    # Seed = first resolved module, targets = rest
    seed_file, seed_mod = resolved[0]
    target_mods = set(mod for _, mod in resolved[1:])

    # Run blast radius
    br_results = get_blast_radius_modules(seed_mod, top_k=20)
    if not br_results:
        skipped += 1
        no_br += 1
        continue
    br_modules = [m for m, _, _ in br_results]
    br_set = set(br_modules)

    # Recall@K
    hits_at_10 = len(target_mods & set(br_modules[:10]))
    hits_at_20 = len(target_mods & br_set)
    recall_10 = hits_at_10 / len(target_mods) if target_mods else 0
    recall_20 = hits_at_20 / len(target_mods) if target_mods else 0

    # MRR: reciprocal rank of first target hit
    mrr = 0
    for rank, mod in enumerate(br_modules):
        if mod in target_mods:
            mrr = 1.0 / (rank + 1)
            break

    # Tier accuracy: hits in will_break or may_break
    tier_hits = 0
    for mod, conf, tier in br_results:
        if mod in target_mods and tier in ("will_break", "may_break"):
            tier_hits += 1
    tier_acc = tier_hits / len(target_mods) if target_mods else 0

    # Confidence stats for hits vs non-hits
    hit_confs = [c for m, c, _ in br_results if m in target_mods]
    miss_confs = [c for m, c, _ in br_results if m not in target_mods]

    results.append({
        "commit": commit["hash"],
        "seed": seed_mod,
        "n_targets": len(target_mods),
        "n_br_results": len(br_results),
        "recall_10": recall_10,
        "recall_20": recall_20,
        "mrr": mrr,
        "tier_accuracy": tier_acc,
        "hits_at_10": hits_at_10,
        "hits_at_20": hits_at_20,
        "avg_hit_conf": sum(hit_confs) / len(hit_confs) if hit_confs else 0,
        "avg_miss_conf": sum(miss_confs) / len(miss_confs) if miss_confs else 0,
    })

    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(test_commits)}] ...")

elapsed = time.time() - t0

# Aggregate
n = len(results)
if n == 0:
    print("No valid test commits found! Check module resolution.")
    sys.exit(1)

avg = lambda key: sum(r[key] for r in results) / n

print(f"\n{'-'*60}")
print(f"RESULTS: {args.label} ({n} evaluated, {skipped} skipped [{no_resolve} no-resolve, {no_br} no-br], {elapsed:.1f}s)")
print(f"{'-'*60}")
print(f"  recall@10:        {avg('recall_10'):.4f}")
print(f"  recall@20:        {avg('recall_20'):.4f}")
print(f"  MRR:              {avg('mrr'):.4f}")
print(f"  tier_accuracy:    {avg('tier_accuracy'):.4f}")
print(f"  avg hits@10:      {avg('hits_at_10'):.2f} / {avg('n_targets'):.1f} targets")
print(f"  avg hits@20:      {avg('hits_at_20'):.2f} / {avg('n_targets'):.1f} targets")
print(f"  avg hit conf:     {avg('avg_hit_conf'):.4f}")
print(f"  avg miss conf:    {avg('avg_miss_conf'):.4f}")
print(f"  conf separation:  {avg('avg_hit_conf') - avg('avg_miss_conf'):.4f}")
print(f"{'-'*60}")

# Save
out_path = args.output or str(pathlib.Path(__file__).parent / "generated" / f"blast_recall_{args.label}.json")
pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
output = {
    "label": args.label,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "n_commits": n,
    "skipped": skipped,
    "elapsed_s": round(elapsed, 1),
    "metrics": {
        "recall_10": round(avg("recall_10"), 4),
        "recall_20": round(avg("recall_20"), 4),
        "mrr": round(avg("mrr"), 4),
        "tier_accuracy": round(avg("tier_accuracy"), 4),
        "avg_hits_10": round(avg("hits_at_10"), 2),
        "avg_hits_20": round(avg("hits_at_20"), 2),
        "avg_targets": round(avg("n_targets"), 1),
        "conf_separation": round(avg("avg_hit_conf") - avg("avg_miss_conf"), 4),
    },
    "per_commit": results,
}
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}")
