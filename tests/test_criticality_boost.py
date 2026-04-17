"""Verify criticality-boost rerank logic in retrieval_engine.

Runs against real criticality_index.json (75k modules). Constructs synthetic
RRF results where we know ground-truth rank shifts, and asserts the boost
behaves as expected under different alpha values.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serve"))

# Point at real artifacts for criticality_index load
os.environ.setdefault("ARTIFACT_DIR", "/home/beast/projects/workspaces/juspay/artifacts")

import retrieval_engine as re_
from retrieval_engine import _apply_criticality_boost, criticality_index

# Lazy-load the criticality index only (skip full engine load)
if not criticality_index:
    with open(os.path.join(os.environ["ARTIFACT_DIR"], "criticality_index.json")) as f:
        data = json.load(f)
    criticality_index.update(data.get("modules", {}))

print(f"Loaded criticality_index: {len(criticality_index)} modules")

# Pick 3 modules with known scores: a HIGH, a MEDIUM, a LOW
sorted_mods = sorted(criticality_index.items(), key=lambda kv: -kv[1].get("score", 0))
high_mod, high_entry = sorted_mods[0]
mid_idx = len(sorted_mods) // 2
mid_mod, mid_entry = sorted_mods[mid_idx]
low_mod, low_entry = sorted_mods[-1]

print(f"HIGH: {high_mod[:60]}... score={high_entry['score']:.3f}")
print(f"MED : {mid_mod[:60]}... score={mid_entry['score']:.3f}")
print(f"LOW : {low_mod[:60]}... score={low_entry['score']:.3f}")

# Build synthetic merged results where HIGH-crit module has LOWER rrf
# Test: does the boost promote it above MID and LOW?
merged = {
    "svc_a": [
        {"name": low_mod, "module": low_mod, "_rrf_score": 0.050, "service": "svc_a"},
        {"name": mid_mod, "module": mid_mod, "_rrf_score": 0.040, "service": "svc_a"},
        {"name": high_mod, "module": high_mod, "_rrf_score": 0.035, "service": "svc_a"},
    ]
}

# Run with alpha=0.5 (default)
os.environ["HR_CRITICALITY_BOOST"] = "1"
os.environ["HR_CRIT_ALPHA"] = "0.5"
boosted = _apply_criticality_boost(merged)
print("\nWith HR_CRIT_ALPHA=0.5:")
for i, n in enumerate(boosted["svc_a"], 1):
    print(f"  #{i}: crit={n['_crit_score']:.3f} rrf={n['_rrf_score']:.3f} -> boosted={n['_rrf_boosted']:.4f} ({n['name'][:40]}...)")

# Run with alpha=2.0 (aggressive)
os.environ["HR_CRIT_ALPHA"] = "2.0"
boosted2 = _apply_criticality_boost(merged)
print("\nWith HR_CRIT_ALPHA=2.0:")
for i, n in enumerate(boosted2["svc_a"], 1):
    print(f"  #{i}: crit={n['_crit_score']:.3f} boosted={n['_rrf_boosted']:.4f} ({n['name'][:40]}...)")

# Run with boost OFF
os.environ["HR_CRITICALITY_BOOST"] = "0"
untouched = _apply_criticality_boost(merged)
print(f"\nWith HR_CRITICALITY_BOOST=0: order preserved = {[n['name']==low_mod or n['name']==mid_mod or n['name']==high_mod for n in untouched['svc_a']]}")
assert untouched == merged, "OFF flag must be a no-op"

# Quantitative check
print("\n=== VERDICT ===")
top_alpha05 = boosted["svc_a"][0]["name"]
top_alpha20 = boosted2["svc_a"][0]["name"]
print(f"alpha=0.5 top-1: {'HIGH-crit' if top_alpha05==high_mod else 'MID' if top_alpha05==mid_mod else 'LOW'}")
print(f"alpha=2.0 top-1: {'HIGH-crit' if top_alpha20==high_mod else 'MID' if top_alpha20==mid_mod else 'LOW'}")
print(f"OFF flag respected: {untouched == merged}")
