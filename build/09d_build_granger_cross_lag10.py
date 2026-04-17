#!/usr/bin/env python3
"""
09c_build_granger_cross_fast.py — Cross-service Granger, day-bucket time series.

Uses 24h calendar buckets (not raw commit sequences) for fast Granger tests.
With 188 calendar days, each pair's time series is 188 points — 400x faster
than the 67K-point approach in 09b.

Partial run (09b) showed 38% significance rate at 2K tested pairs — signal is real.
This script tests all 60K top pairs in ~2 minutes instead of 3.4 hours.
"""
import json, pathlib, sys, time, re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

try:
    import ijson
except ImportError:
    raise SystemExit("pip install ijson")
try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:
    raise SystemExit("pip install statsmodels")
import warnings
warnings.filterwarnings("ignore")

GIT_HISTORY  = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
    "/home/beast/projects/workspaces/juspay/git_history.json")
ARTIFACT     = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else
    "/home/beast/projects/workspaces/juspay/artifacts")
CROSS_COCHANGE = ARTIFACT / "cross_cochange_index.json"
OUT_PATH       = ARTIFACT / "granger_cross_lag10_index.json"

SRC_EXTS     = {".hs", ".rs", ".hs-boot", ".py", ".ts", ".js", ".go"}
SKIP_DIRS    = {".stack-work", "test", "tests", "spec", "mock", "node_modules", "__pycache__"}
MAX_FILES    = 40
MAX_LAG      = 10
P_THRESHOLD  = 0.05
MIN_ACTIVE_DAYS = 5   # module must change on ≥5 distinct calendar days
MIN_COCHANGE_WEIGHT = 5
MAX_PAIRS    = 60_000


def is_source(path: str) -> bool:
    p = pathlib.PurePosixPath(path)
    return p.suffix in SRC_EXTS and not (set(p.parts) & SKIP_DIRS)


def to_module(repo: str, fpath: str) -> str:
    p = fpath
    for ext in SRC_EXTS:
        p = p.replace(ext, "")
    return f"{repo}::{p.replace('/', '::')}"


def parse_ts(date_str: str) -> float:
    s = date_str.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        pass
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})([+-]\d{2}):(\d{2})$", s)
    if m:
        dt = datetime.fromisoformat(m.group(1))
        h, mins = int(m.group(2)[1:]), int(m.group(3))
        offset = (h * 60 + mins) * 60 * (-1 if m.group(2)[0] == "-" else 1)
        return dt.timestamp() - offset
    return datetime.fromisoformat(s[:10]).timestamp()


def build():
    # ── Step 1: Load cross-repo pairs ─────────────────────────────────────
    print("Loading cross co-change index...", flush=True)
    with open(CROSS_COCHANGE) as f:
        xcc = json.load(f)
    edges = xcc.get("edges", xcc)
    pair_weights: dict = {}
    for mod_a, neighbors in edges.items():
        if mod_a == "meta" or not isinstance(neighbors, list):
            continue
        repo_a = mod_a.split("::")[0]
        for nb in neighbors:
            mod_b = nb.get("module", "")
            if mod_a.split("::")[0] == mod_b.split("::")[0]:
                continue
            w = nb.get("weight", 0)
            if w >= MIN_COCHANGE_WEIGHT:
                pair = tuple(sorted([mod_a, mod_b]))
                pair_weights[pair] = max(pair_weights.get(pair, 0), w)

    top_pairs = sorted(pair_weights, key=lambda p: pair_weights[p], reverse=True)[:MAX_PAIRS]
    mods_needed = {m for p in top_pairs for m in p}
    print(f"  Top {len(top_pairs):,} cross-repo pairs, {len(mods_needed):,} modules", flush=True)

    # ── Step 2: Stream git history, build day-bucket time series ──────────
    print("Building calendar day-bucket time series...", flush=True)
    # module_days[mod] = set of day indices (UTC day since epoch)
    module_days: dict[str, set] = defaultdict(set)
    ts_min = float("inf"); ts_max = float("-inf")
    current_repo = None; current_ts = None; commit_files = []; in_commit = False

    with open(GIT_HISTORY, "rb") as f:
        parser = ijson.parse(f, use_float=True)
        try:
            for prefix, event, value in parser:
                if prefix == "repositories.item.name" and event == "string":
                    current_repo = value
                elif prefix == "repositories.item.commits.item" and event == "start_map":
                    in_commit = True; commit_files = []; current_ts = None
                elif in_commit and prefix.endswith(".date") and event == "string":
                    try: current_ts = parse_ts(value)
                    except Exception: pass
                elif in_commit and "files_changed" in prefix and prefix.endswith(".path") and event == "string":
                    if is_source(value): commit_files.append(value)
                elif prefix == "repositories.item.commits.item" and event == "end_map":
                    in_commit = False
                    if current_ts and 1 <= len(commit_files) <= MAX_FILES and current_repo:
                        day_idx = int(current_ts // 86400)
                        ts_min = min(ts_min, current_ts); ts_max = max(ts_max, current_ts)
                        mods = {to_module(current_repo, fp) for fp in commit_files}
                        for m in mods & mods_needed:
                            module_days[m].add(day_idx)
        except Exception as exc:
            print(f"Warning: {exc}", flush=True)

    if ts_min == float("inf"):
        raise SystemExit("No timestamp data found")

    day_start = int(ts_min // 86400)
    day_end   = int(ts_max // 86400)
    n_days    = day_end - day_start + 1
    print(f"  Calendar span: {n_days} days  |  Modules tracked: {len(module_days):,}", flush=True)

    # ── Step 3: Run Granger tests ─────────────────────────────────────────
    print(f"Running Granger tests ({len(top_pairs):,} pairs, {n_days}-point series)...", flush=True)
    granger_results = {}; tested = significant = skipped = 0
    t0 = time.time()

    for pair_idx, (mod_a, mod_b) in enumerate(top_pairs):
        if pair_idx % 5000 == 0 and pair_idx > 0:
            elapsed = time.time() - t0
            rate = pair_idx / elapsed
            eta = (len(top_pairs) - pair_idx) / rate
            print(f"  {pair_idx:,}/{len(top_pairs):,}  sig={significant}  ETA={eta:.0f}s", flush=True)

        days_a = module_days.get(mod_a, set())
        days_b = module_days.get(mod_b, set())
        if len(days_a) < MIN_ACTIVE_DAYS or len(days_b) < MIN_ACTIVE_DAYS:
            skipped += 1; continue

        # Dense binary series over the full calendar window
        ts_a = np.zeros(n_days, dtype=np.float32)
        ts_b = np.zeros(n_days, dtype=np.float32)
        for d in days_a: ts_a[d - day_start] = 1.0
        for d in days_b: ts_b[d - day_start] = 1.0

        tested += 1
        try:
            def best_p(y, x):
                data = np.column_stack([y, x])
                res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
                best_lag = min(range(1, MAX_LAG + 1), key=lambda l: res[l][0]["ssr_chi2test"][1])
                return res[best_lag][0]["ssr_chi2test"][1], best_lag

            p_a2b, lag_a2b = best_p(ts_b, ts_a)
            p_b2a, lag_b2a = best_p(ts_a, ts_b)

            w = pair_weights.get((mod_a, mod_b), pair_weights.get((mod_b, mod_a), 0))

            if p_a2b < P_THRESHOLD:
                granger_results[f"{mod_a}→{mod_b}"] = {
                    "source": mod_a, "target": mod_b,
                    "best_lag": lag_a2b, "p_value": round(p_a2b, 6), "weight": w,
                }
                significant += 1

            if p_b2a < P_THRESHOLD:
                granger_results[f"{mod_b}→{mod_a}"] = {
                    "source": mod_b, "target": mod_a,
                    "best_lag": lag_b2a, "p_value": round(p_b2a, 6), "weight": w,
                }
                significant += 1

        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s", flush=True)
    print(f"Tested: {tested:,}  Skipped: {skipped:,}  Significant: {significant:,}", flush=True)

    out = {
        "metadata": {
            "calendar_days": n_days, "pairs_tested": tested,
            "significant_results": significant,
            "p_threshold": P_THRESHOLD, "max_lag": MAX_LAG,
            "min_active_days": MIN_ACTIVE_DAYS, "mode": "cross-service-day-bucket",
        },
        "causal_pairs": granger_results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f)
    print(f"Written: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    build()
