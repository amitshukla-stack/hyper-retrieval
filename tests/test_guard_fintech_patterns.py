"""
T-020 Guard fintech patterns — 5 new patterns, each with TP + TN fixtures.
Gate: 5/5 TP detected, 5/5 TN clean.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "guardrails"))
from comment_code_checker import (
    check_float_for_money,
    check_timeout_without_backoff,
    check_missing_idempotency_key,
    check_unbounded_retry,
    check_silent_state_mutation,
)

tp_pass = tn_pass = 0
tp_fail = tn_fail = 0

def assert_tp(name, findings):
    global tp_pass, tp_fail
    if findings:
        print(f"  TP PASS — {name}: {findings[0].pattern} L{findings[0].line}")
        tp_pass += 1
    else:
        print(f"  TP FAIL — {name}: no finding")
        tp_fail += 1

def assert_tn(name, findings):
    global tn_pass, tn_fail
    if not findings:
        print(f"  TN PASS — {name}")
        tn_pass += 1
    else:
        print(f"  TN FAIL — {name}: got {[f.pattern for f in findings]}")
        tn_fail += 1

# ── Pattern 1: float-for-money ────────────────────────────────────────────
print("\n[1] float-for-money")
assert_tp("amount: float annotation",
    check_float_for_money("def charge(amount: float): pass"))
assert_tp("float() conversion",
    check_float_for_money("total = float(amount)"))
assert_tn("Decimal amount",
    check_float_for_money("from decimal import Decimal\namount: Decimal = Decimal('9.99')"))
assert_tn("int cents",
    check_float_for_money("amount_cents: int = 999"))

# ── Pattern 2: timeout-without-backoff ───────────────────────────────────
print("\n[2] timeout-without-backoff")
assert_tp("bare timeout on requests.post",
    check_timeout_without_backoff('r = requests.post(url, json=data, timeout=30)'))
assert_tp("bare timeout on httpx.get",
    check_timeout_without_backoff('resp = httpx.get(url, timeout=5)'))
assert_tn("timeout + tenacity retry",
    check_timeout_without_backoff(
        'from tenacity import retry\n@retry\ndef call(): r = requests.post(url, timeout=30)'))
assert_tn("timeout + backoff sleep",
    check_timeout_without_backoff(
        'for i in range(3):\n  time.sleep(2**i)\n  r = requests.post(url, timeout=30)'))

# ── Pattern 3: missing-idempotency-key ───────────────────────────────────
print("\n[3] missing-idempotency-key")
assert_tp("stripe charge without key",
    check_missing_idempotency_key("stripe.Charge.create(amount=100, currency='usd')"))
assert_tp("requests.post to payment endpoint",
    check_missing_idempotency_key("resp = requests.post('/api/payment', json=payload)"))
assert_tn("stripe charge with idempotency_key",
    check_missing_idempotency_key(
        "stripe.Charge.create(amount=100, idempotency_key=uid)"))
assert_tn("non-payment POST",
    check_missing_idempotency_key("requests.post('/api/users', json={'name': 'test'})"))

# ── Pattern 4: unbounded-retry ───────────────────────────────────────────
print("\n[4] unbounded-retry")
assert_tp("while True no ceiling",
    check_unbounded_retry("while True:\n    result = send_to_gateway()\n    if result: return result\n"))
assert_tp("while retry: no ceiling",
    check_unbounded_retry("while retry:\n    do_something()\n    if ok: continue\n"))
assert_tn("while True with max_retries",
    check_unbounded_retry(
        "while True:\n    attempts += 1\n    if attempts > max_retries:\n        raise\n"))
assert_tn("while True with break",
    check_unbounded_retry("while True:\n    r = get()\n    if r.ok:\n        break\n"))

# ── Pattern 5: silent-state-mutation ─────────────────────────────────────
print("\n[5] silent-state-mutation")
assert_tp("account.status = without log",
    check_silent_state_mutation("account.status = 'frozen'", "serve/accounts.py"))
assert_tp("update() without audit",
    check_silent_state_mutation("update(user_id, {'status': 'disabled'})", "serve/admin.py"))
assert_tn("mutation with logger",
    check_silent_state_mutation(
        "account.status = 'frozen'\nlogger.info('account frozen')", "serve/accounts.py"))
assert_tn("test file excluded",
    check_silent_state_mutation("account.status = 'frozen'", "tests/test_accounts.py"))

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"TP: {tp_pass}/{tp_pass+tp_fail} PASS  |  TN: {tn_pass}/{tn_pass+tn_fail} PASS")
gate_pass = tp_fail == 0 and tn_fail == 0
print(f"Gate (5/5 TP, 5/5 TN): {'PASS ✓' if gate_pass else 'FAIL ✗'}")
if not gate_pass:
    sys.exit(1)
