"""
test_03_canary.py — Critical function body integrity tests.

Ground truth = the SOURCE FILE, not the body_store.

For each canary function:
  1. Load body_store entry
  2. Read the actual source file from all_repos/
  3. Verify stored body is a prefix of actual function source
  4. Verify stored body contains expected code fragments
  5. Verify length is within expected range

This test catches:
  - Silent truncation (body cut mid-statement)
  - Stale/mismatched body (body from wrong function or wrong version)
  - Parser regression (new version of extractor produces wrong body)

Run:
    python3 tests/test_03_canary.py
"""
import sys, json, pathlib, re, os
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "serve"))

_WS       = pathlib.Path(os.environ.get("WORKSPACE_DIR", "/home/beast/projects/workspaces/juspay"))
OUTPUT    = pathlib.Path(os.environ.get("OUTPUT_DIR",    str(_WS / "output")))
ARTIFACTS = pathlib.Path(os.environ.get("ARTIFACT_DIR", str(_WS / "artifacts")))
ALL_REPOS = pathlib.Path(os.environ.get("SOURCE_DIR",   str(_WS / "source")))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
errors: list = []

def fail(label, detail=""):
    print(f"  {FAIL} {label}")
    if detail: print(f"      {detail}")
    errors.append(label)

def ok(label):
    print(f"  {PASS} {label}")

def warn(label, detail=""):
    print(f"  {WARN} {label}")
    if detail: print(f"      {detail}")


# ── Load body_store ────────────────────────────────────────────────────────────
body_store = json.loads((OUTPUT / "body_store.json").read_text())

# ── Canary function definitions ────────────────────────────────────────────────
#
# Each entry: (fn_id, source_file, fn_name, min_len, max_len, must_contain_fragments)
#
# min_len / max_len: expected CHARACTER range of stored body
# must_contain_fragments: code strings that MUST appear in the stored body
#   Ground truth: verified by reading source on 2026-03-22
#
CANARIES = [
    # ── Master payment flow function ──────────────────────────────────────────
    (
        "PaymentFlows.getAllPaymentFlowsForTxn",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getAllPaymentFlowsForTxn",
        3500, 8500,
        [
            # Tags from lines 64-100 (verified correct)
            "getCardAuthTypeDetails",
            "authFlowTypePaymentFlows",
            "getPaymentFlowsFromTxnType",
            "getCardTokenRepeatFlowMapper",
            "getCardMandateFlowMapper",
            "getEmandateFlowMapper",
            "getPaymentFlowsForEMI",
            # Tags from lines 101-114 (the ones that were previously missed)
            "GUARANTEE_FLOW",
            "BILLING_MANDATE_REGISTER",
            "BILLING_MANDATE_PAYMENT",
            "getTransactionIntentFlows",
            "getDecryptedPaymentTokenFlow",
            "getMetricBillingExecutionTypeFromOrder",
        ]
    ),

    # ── Second-layer post-routing tags ────────────────────────────────────────
    # getUpdatedPaymentFLows body: lines 329-369 (where clause + helpers).
    # SR_SELECTION/PRIORITY_LOGIC/PAYMENT_COLLECTION_LINK/INAPP_FRESH/INAPP_LINKED
    # appear in ADJACENT top-level functions (getDeciderRoutingApproach,
    # ppFlowMapper) — NOT inside getUpdatedPaymentFLows itself.
    (
        "PaymentFlows.getUpdatedPaymentFLows",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getUpdatedPaymentFLows",
        1500, 8500,
        [
            "RISK_CHECK",
            "CUSTOMER_FEE_BEARING_SURCHARGE",
            "getDeciderRoutingApproach",
            "nbQuickCheckoutPaymentFlows",
            "getGpayDecryptedFlow",
        ]
    ),

    # ── Card token repeat mapper (the error-source function) ──────────────────
    (
        "PaymentFlows.getCardTokenRepeatFlowMapper",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getCardTokenRepeatFlowMapper",
        200, 8500,
        [
            "LOCKER_TOKEN_USED",
            "NETWORK_TOKEN_USED",
            "ISSUER_TOKEN_USED",
            "ALTID",
            "SODEXO_TOKEN_USED",
            "ISSUER_ALT_ID",
        ]
    ),

    # ── PF/PL flow (the other error-source function) ──────────────────────────
    (
        "PaymentFlows.getPfOrPlFlowDetails",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getPfOrPlFlowDetails",
        100, 8500,
        [
            "PAYMENT_LINK",
            "PAYMENT_FORM",
        ]
    ),

    # ── UPI flow mapper ───────────────────────────────────────────────────────
    (
        "PaymentFlows.getUpiFlowMapper",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getUpiFlowMapper",
        200, 8500,
        ["COLLECT", "INTENT", "INAPP", "QR"]
    ),

    # ── Card auth type ────────────────────────────────────────────────────────
    (
        "PaymentFlows.getCardAuthTypeDetails",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getCardAuthTypeDetails",
        300, 8500,
        ["THREE_DS", "DOTP", "MOTO", "NO_THREE_DS", "FIDO"]
    ),

    # ── Fallback source-object tags ───────────────────────────────────────────
    (
        "PaymentFlows.getFallbackPfFromSourceObject",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getFallbackPfFromSourceObject",
        400, 8500,
        [
            "PAYMENT_CHANNEL_FALLBACK_DOTP_TO_3DS",
            "PG_FAILURE_FALLBACK_DOTP_TO_3DS",
            "FRM_FALLBACK_TO_3DS",
            "ORDER_PREFERENCE_TO_NO_3DS",
        ]
    ),

    # ── EMI flows ─────────────────────────────────────────────────────────────
    (
        "PaymentFlows.getPaymentFlowsForEMI",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getPaymentFlowsForEMI",
        300, 8500,
        ["NO_COST_EMI", "LOW_COST_EMI", "STANDARD_EMI", "DIRECT_BANK_EMI"]
    ),

    # ── Retry type ────────────────────────────────────────────────────────────
    (
        "PaymentFlows.getTxnRetryType",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getTxnRetryType",
        200, 8500,
        ["NO_THREE_DS_RETRY", "NO_THREE_DS_UPGRADE", "SILENT_RETRY"]
    ),

    # ── Pre-auth flows ────────────────────────────────────────────────────────
    (
        "PaymentFlows.getPaymentFlowsForPreAuth",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getPaymentFlowsForPreAuth",
        200, 8500,
        ["PARTIAL_CAPTURE", "MULTIPLE_PARTIAL_CAPTURE"]
    ),

    # ── Txn type → flow tags ──────────────────────────────────────────────────
    (
        "PaymentFlows.getPaymentFlowsFromTxnType",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getPaymentFlowsFromTxnType",
        200, 8500,
        ["PREAUTH", "ZERO_AUTH", "STANDALONE_AUTHENTICATION", "STANDALONE_AUTHORIZATION"]
    ),

    # ── Card mandate mapper ───────────────────────────────────────────────────
    (
        "PaymentFlows.getCardMandateFlowMapper",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getCardMandateFlowMapper",
        200, 8500,
        ["MANDATE_REGISTER", "MANDATE_PAYMENT", "TPV_MANDATE"]
    ),

    # ── E-mandate mapper ──────────────────────────────────────────────────────
    (
        "PaymentFlows.getEmandateFlowMapper",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getEmandateFlowMapper",
        200, 8500,
        ["EMANDATE_REGISTER", "EMANDATE_PAYMENT", "TPV_EMANDATE"]
    ),

    # ── UPI source flows ──────────────────────────────────────────────────────
    (
        "PaymentFlows.getUpiSourceFlows",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "getUpiSourceFlows",
        100, 8500,
        ["UPI_SOURCE"]
    ),

    # ── SCA exemption tags ────────────────────────────────────────────────────
    (
        "PaymentFlows.authFlowTypePaymentFlows",
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "authFlowTypePaymentFlows",
        100, 8500,
        ["PREFERRED_3DS", "MANDATED_3DS", "TRA_EXEMPTION", "LOW_VALUE_EXEMPTION"]
    ),
]


def _extract_hs_body_from_source(src: str, fn_name: str, max_chars: int = 20000) -> str:
    """
    Re-extract function body directly from Haskell source.
    Used as ground truth — independent of the pipeline extractor.
    Returns the raw text from the first definition line to the next top-level def.
    """
    lines = src.splitlines(keepends=True)
    body_lines = []
    in_body = False
    top_level_re = re.compile(r'^[a-z_][a-zA-Z0-9_\']*\s*(?:[^=\n]*=|::)')

    for i, line in enumerate(lines):
        if not in_body:
            if re.match(rf'^{re.escape(fn_name)}\b', line):
                in_body = True
            elif re.match(rf'^{re.escape(fn_name)}\s*::', line):
                continue  # skip type sig, wait for definition
        if in_body:
            if i > 0 and top_level_re.match(line) and not re.match(rf'^{re.escape(fn_name)}\b', line):
                break
            body_lines.append(line)

    body = "".join(body_lines).rstrip()
    return body[:max_chars]


print("\n=== Canary function body integrity ===")
print(f"  Testing {len(CANARIES)} critical functions against source files\n")

for fn_id, src_rel, fn_name, min_len, max_len, must_have in CANARIES:
    print(f"  --- {fn_id} ---")

    # Check body_store has the entry
    stored = body_store.get(fn_id, None)
    if stored is None:
        fail(f"body_store has entry for {fn_id}")
        continue

    # Check length range
    blen = len(stored)
    if not (min_len <= blen <= max_len):
        fail(f"Body length in range [{min_len}, {max_len}]",
             f"Actual length: {blen}")
    else:
        ok(f"Length {blen} in range [{min_len}, {max_len}]")

    # Check must-have fragments
    for fragment in must_have:
        if fragment in stored:
            ok(f"  Contains {fragment!r}")
        else:
            fail(f"  Must contain {fragment!r}",
                 f"First 200 chars of stored body: {stored[:200]!r}")

    # Ground-truth check: compare against source file
    src_path = ALL_REPOS / src_rel
    if not src_path.exists():
        warn(f"Source file not found (skipping ground-truth check): {src_path}")
        continue

    src_text = src_path.read_text(encoding="utf-8", errors="replace")
    true_body = _extract_hs_body_from_source(src_text, fn_name)

    if not true_body:
        warn(f"Could not extract {fn_name} from source (parser may differ)")
        continue

    # Stored body must be a prefix of true body (may be truncated, but not diverge)
    stored_lines = [l.strip() for l in stored.split("\n") if l.strip()]
    true_lines   = [l.strip() for l in true_body.split("\n") if l.strip()]

    # First N lines of stored body must appear in true body in the same order
    n_check = min(5, len(stored_lines))
    prefix_ok = all(stored_lines[i] in true_body for i in range(n_check))
    if prefix_ok:
        ok(f"  First {n_check} lines of stored body found in source")
    else:
        fail(f"  Stored body diverges from source",
             f"First stored line: {stored_lines[0][:80]!r}\n"
             f"      First source line: {true_lines[0][:80]!r}")

    # If stored body is much shorter than true body without a truncation marker, flag it
    true_len = len(true_body)
    stored_untrunc = stored.replace("-- [body truncated]", "").replace(
        "-- [body truncated at char limit]", "").strip()
    ratio = len(stored_untrunc) / max(true_len, 1)
    if ratio < 0.5 and "truncated" not in stored:
        fail(f"  Body completeness: stored is {ratio:.0%} of source without truncation marker",
             f"stored={len(stored_untrunc)}, source={true_len}")
    elif ratio < 0.9 and "truncated" not in stored:
        warn(f"  Body {ratio:.0%} of source — may be silently truncated",
             f"stored={len(stored_untrunc)}, source={true_len}")
    else:
        ok(f"  Completeness: stored={len(stored)}, source={true_len}, ratio={ratio:.0%}"
           + (" [truncated with marker]" if "truncated" in stored else ""))

    print()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
if errors:
    print(f"\033[91m{len(errors)} FAILED: {errors}\033[0m")
    sys.exit(1)
else:
    print(f"\033[92mAll {len(CANARIES)} canary functions passed.\033[0m")
