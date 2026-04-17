"""Guard bundled-path test: verify HR ships Guard out of the box.

The existing `test_guard_integration.py` test pins `HR_GUARD_PATH` to the
external prototype location for dev parity. This test covers the *installed
user* path — no env vars, Guard loads from the bundled `hyperretrieval/
guardrails/comment_code_checker.py` that ships with the package.
"""
from __future__ import annotations
import os
import pathlib
import sys
import tempfile

# Remove any dev override so we exercise the bundled path
os.environ.pop("HR_GUARD_PATH", None)
os.environ.pop("HR_GUARD_DISABLE", None)

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from serve import guard_integration as gi  # noqa: E402

print("=== Guard bundled-path tests ===")

print("TEST 1: bundled checker exists at expected location")
expected = REPO / "guardrails" / "comment_code_checker.py"
assert expected.exists(), f"bundled checker missing: {expected}"
assert gi._BUNDLED_CHECKER == expected, (
    f"_BUNDLED_CHECKER={gi._BUNDLED_CHECKER} != {expected}"
)
print(f"  path: {expected}")
print("  PASS")

print("\nTEST 2: available() returns True with no env vars set")
assert gi.available(), "Guard should load from bundled path"
print(f"  loaded module from: {gi._GUARD_MOD.__file__}")
assert str(expected) in gi._GUARD_MOD.__file__, (
    "loaded module must come from bundled path"
)
print("  PASS")

print("\nTEST 3: imperative-directive TP fires through the bundled loader")
bad = '''
def process_payment(pid):
    # Acquire lock for payment processing
    amount = pid * 2
    return amount
'''
with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
    f.write(bad)
    tp_path = f.name
findings = gi.run_guard_on_files([tp_path])
assert len(findings) >= 1, f"expected >=1 finding, got {len(findings)}"
patterns = {f["pattern"] for f in findings}
assert "comment-action-mismatch" in patterns, f"expected comment-action-mismatch, got {patterns}"
print(f"  findings: {len(findings)}, patterns: {sorted(patterns)}")
pathlib.Path(tp_path).unlink()
print("  PASS")

print("\nTEST 4: narrative-text FP does NOT fire (imperative filter)")
narrative = '''
def helper(x):
    # Note: security checks happen in the middleware layer
    return x * 2
'''
with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
    f.write(narrative)
    fp_path = f.name
findings2 = gi.run_guard_on_files([fp_path])
cam = [x for x in findings2 if x["pattern"] == "comment-action-mismatch"]
assert len(cam) == 0, f"narrative FP should be filtered, got {cam}"
print(f"  comment-action-mismatch findings: {len(cam)} (expect 0)")
pathlib.Path(fp_path).unlink()
print("  PASS")

print("\nTEST 5: summarize_findings aggregates correctly")
summary = gi.summarize_findings([
    {"severity": "CRITICAL", "pattern": "lock-with-pass"},
    {"severity": "WARNING", "pattern": "comment-action-mismatch"},
    {"severity": "WARNING", "pattern": "error-swallowed"},
])
assert summary["count"] == 3
assert summary["critical"] == 1
assert summary["warning"] == 2
assert set(summary["patterns"]) == {"lock-with-pass", "comment-action-mismatch", "error-swallowed"}
print(f"  summary: {summary}")
print("  PASS")

print("\n=== Guard bundled-path VERIFIED ===")
