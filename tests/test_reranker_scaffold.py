"""Verify pluggable reranker interface without downloading any model.

Uses a fake Reranker subclass that inverts the input order. This proves:
- `apply_reranker` plumbs correctly
- OFF flag (HR_RERANKER unset) is a no-op
- `alpha` knob blends correctly
- Nodes get `_rerank_score` + `_final_rank_score` stamped
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from serve.reranker import Reranker, apply_reranker, get_reranker


class InvertingReranker(Reranker):
    """Scores candidates in reverse-index order (first input → lowest score)."""
    name = "invert"
    def available(self): return True
    def rerank(self, query, candidates):
        scored = [(c, float(i)) for i, c in enumerate(candidates)]
        scored.sort(key=lambda cs: -cs[1])
        return scored


fake = InvertingReranker()

# Build a merged bucket with explicit RRF order: nodes 0,1,2,3,4
svc = [
    {"name": f"mod_{i}", "module": f"mod_{i}", "_rrf_score": 0.1 - 0.01 * i}
    for i in range(5)
]
merged = {"svc_a": svc}

# ─── TEST 1: fake reranker with alpha=1.0 → order is fully inverted ───
os.environ["HR_RERANKER_TOPK"] = "5"
os.environ["HR_RERANKER_ALPHA"] = "1.0"
out1 = apply_reranker("any query", merged, reranker=fake)
names1 = [n["name"] for n in out1["svc_a"]]
print(f"TEST 1 alpha=1.0: input order=['mod_0'..'mod_4'], output={names1}")
assert names1 == ["mod_4", "mod_3", "mod_2", "mod_1", "mod_0"], f"Got {names1}"
for n in out1["svc_a"]:
    assert "_rerank_score" in n and "_final_rank_score" in n
print("  PASS\n")

# ─── TEST 2: alpha=0.0 → RRF wins, order unchanged ───
os.environ["HR_RERANKER_ALPHA"] = "0.0"
out2 = apply_reranker("any query", merged, reranker=fake)
names2 = [n["name"] for n in out2["svc_a"]]
print(f"TEST 2 alpha=0.0: output={names2}")
assert names2 == ["mod_0", "mod_1", "mod_2", "mod_3", "mod_4"], f"Got {names2}"
print("  PASS\n")

# ─── TEST 3: HR_RERANKER unset → get_reranker returns None → no-op ───
os.environ.pop("HR_RERANKER", None)
rr = get_reranker()
print(f"TEST 3 HR_RERANKER unset: get_reranker()={rr}")
assert rr is None
out3 = apply_reranker("any query", merged)  # no explicit reranker
assert out3 == merged, "OFF path must be bit-exact"
print("  PASS\n")

# ─── TEST 4: HR_RERANKER=noop returns NoopReranker ───
os.environ["HR_RERANKER"] = "noop"
rr4 = get_reranker()
print(f"TEST 4 HR_RERANKER=noop: rr={rr4.name if rr4 else None}")
assert rr4 is not None and rr4.name == "noop"
print("  PASS\n")

# ─── TEST 5: HR_RERANKER=bge returns BGEReranker (not loaded yet) ───
os.environ["HR_RERANKER"] = "bge"
# Force a fresh instance
import serve.reranker as _r
_r._RERANKER_INSTANCE = None
rr5 = get_reranker()
print(f"TEST 5 HR_RERANKER=bge: rr={rr5.name if rr5 else None}")
assert rr5 is not None and rr5.name == "bge"
print("  PASS (model not yet downloaded — .available() would return False)")

print("\n=== reranker scaffold VERIFIED ===")
