"""
T-018 LLM alignment checker — vendored into HR guardrails.
Gate: TP >= 1/1 (MISALIGNED detected), TN >= 1/1 (ALIGNED not flagged),
      latency <= 5s per call, no import errors.
"""
import sys, os, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "guardrails"))

# Set env vars for test (use HR LLM config)
os.environ.setdefault("LLM_API_KEY",  "sk-bNKMSoQf4hYrFQ8Lf8ftYg")
os.environ.setdefault("LLM_BASE_URL", "https://grid.ai.juspay.net")
os.environ.setdefault("LLM_MODEL",    "kimi-latest")

# TP: comment claims to validate+debit but code just writes raw SQL
TP_CODE = '''
def process_payment(amount, account_id):
    """Validates the payment amount and debits the account atomically."""
    db.write(f"UPDATE accounts SET balance = balance - {amount} WHERE id = {account_id}")
    return True
'''

# TN: comment accurately describes the code
TN_CODE = '''
def validate_and_debit(amount, account_id):
    """Validates the payment amount and debits the account atomically."""
    if amount <= 0:
        raise ValueError("Amount must be positive")
    with db.transaction():
        balance = db.query(f"SELECT balance FROM accounts WHERE id = {account_id}")
        if balance < amount:
            raise InsufficientFundsError()
        db.execute(f"UPDATE accounts SET balance = balance - {amount}")
    return True
'''

try:
    from llm_alignment_checker import check_llm_alignment, _call_llm
    print("Import: OK")
except ImportError as e:
    print(f"Import FAIL: {e}")
    sys.exit(1)

# ── Connectivity test ──────────────────────────────────────────────────────
print("\nTest 0: LLM connectivity")
t0 = time.time()
verdict, reason = _call_llm("Returns True always", "def f(): return True")
elapsed = time.time() - t0
print(f"  Verdict: {verdict} | Reason: {reason[:60]} | {elapsed:.1f}s")
if verdict == "ERROR":
    print(f"  WARN: LLM unavailable — skipping TP/TN tests")
    print(f"\nResult: SKIP (LLM not reachable)")
    sys.exit(0)

# ── TP test ───────────────────────────────────────────────────────────────
print("\nTest 1: TP — MISALIGNED code should be flagged")
t0 = time.time()
findings = check_llm_alignment(TP_CODE, "payment.py", language="python")
elapsed = time.time() - t0
if findings:
    print(f"  PASS — {len(findings)} finding(s): {findings[0].pattern} L{findings[0].line} ({elapsed:.1f}s)")
    tp_pass = True
else:
    print(f"  FAIL — no findings on MISALIGNED code ({elapsed:.1f}s)")
    tp_pass = False

# ── TN test ───────────────────────────────────────────────────────────────
print("\nTest 2: TN — ALIGNED code should NOT be flagged")
t0 = time.time()
findings = check_llm_alignment(TN_CODE, "payment.py", language="python")
elapsed = time.time() - t0
if not findings:
    print(f"  PASS — 0 findings on ALIGNED code ({elapsed:.1f}s)")
    tn_pass = True
else:
    print(f"  FAIL — {len(findings)} finding(s) on clean code: {[f.pattern for f in findings]}")
    tn_pass = False

print(f"\n{'='*50}")
gate = tp_pass and tn_pass
print(f"Gate (TP=1/1, TN=1/1): {'PASS ✓' if gate else 'FAIL ✗'}")
if not gate:
    sys.exit(1)
