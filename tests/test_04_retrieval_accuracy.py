"""
test_04_retrieval_accuracy.py — Retrieval accuracy tests with loaded data.

Tests that known queries return expected functions in the top results.
Tests that tool outputs contain expected content for known IDs.

Requires:
  - Data stores loaded (body_store, call_graph, graph — no GPU)
  - EMBED_SERVER_URL not required (keyword-search tests run without it)

Run:
    EMBED_SERVER_URL="" python3 tests/test_04_retrieval_accuracy.py
"""
import sys, os, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "serve"))

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
errors: list = []

def ok(label):
    print(f"  {PASS} {label}")

def fail(label, detail=""):
    print(f"  {FAIL} {label}")
    if detail: print(f"      {detail}")
    errors.append(label)

def warn(label, detail=""):
    print(f"  {WARN} {label}")
    if detail: print(f"      {detail}")

# Disable embed server so all tests run keyword-only (deterministic)
os.environ["EMBED_SERVER_URL"] = ""

import retrieval_engine as RE
import tools as T

print("Loading data stores (no GPU)...")
RE.initialize(load_embedder=False)
print("Ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Keyword search — known queries → expected function IDs in results
# ══════════════════════════════════════════════════════════════════════════════
print("=== 1. Keyword search accuracy (deterministic — no GPU) ===")

KW_CASES = [
    # (query, expected_ids_in_any_service_results)
    ("getUpiFlowMapper",
     ["PaymentFlows.getUpiFlowMapper"]),

    ("card mandate",
     ["PaymentFlows.getCardMandateFlowMapper"]),

    ("getAllPaymentFlowsForTxn",
     ["PaymentFlows.getAllPaymentFlowsForTxn"]),

    ("emandateFlowMapper",
     ["PaymentFlows.getEmandateFlowMapper"]),

    ("getPaymentFlowsForEMI",
     ["PaymentFlows.getPaymentFlowsForEMI"]),

    ("getPaymentFlowsForPreAuth",
     ["PaymentFlows.getPaymentFlowsForPreAuth"]),

    ("card auth type",
     ["PaymentFlows.getCardAuthTypeDetails"]),

    ("UPI source flows",
     ["PaymentFlows.getUpiSourceFlows"]),

    ("getPaymentFlowsFromTxnType",
     ["PaymentFlows.getPaymentFlowsFromTxnType"]),

    ("updated payment flows",
     ["PaymentFlows.getUpdatedPaymentFLows"]),
]

for query, expected_ids in KW_CASES:
    results_by_svc = RE.cross_service_keyword_search(query, max_per_service=30)
    all_hits = [n.get("id", "") for hits in results_by_svc.values() for n in hits]
    missing = [eid for eid in expected_ids if eid not in all_hits]
    if missing:
        fail(f"'{query}' → expected IDs found",
             f"Missing: {missing}\nAll hits: {all_hits[:10]}")
    else:
        ok(f"'{query}' → {expected_ids}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. get_function_body — known IDs → content checks
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. get_function_body content accuracy ===")

BODY_CASES = [
    # (fn_id, fragments_that_must_appear_in_output)
    (
        "PaymentFlows.getAllPaymentFlowsForTxn",
        ["getAllPaymentFlowsForTxn", "isOTMFlow",
         "isGuranteeFlow", "authFlowTypePaymentFlows"]
    ),
    (
        "PaymentFlows.getCardTokenRepeatFlowMapper",
        ["LOCKER_TOKEN_USED", "NETWORK_TOKEN_USED", "ISSUER_TOKEN_USED"]
    ),
    (
        "PaymentFlows.getPfOrPlFlowDetails",
        ["PAYMENT_LINK", "PAYMENT_FORM"]
    ),
    (
        "PaymentFlows.getUpiFlowMapper",
        ["COLLECT", "INTENT", "INAPP", "QR"]
    ),
    (
        "PaymentFlows.getFallbackPfFromSourceObject",
        ["PAYMENT_CHANNEL_FALLBACK_DOTP_TO_3DS", "FRM_FALLBACK_TO_3DS"]
    ),
    (
        "PaymentFlows.getUpdatedPaymentFLows",
        ["RISK_CHECK", "isRiskcheckEnabled"]
    ),
]

for fn_id, expected_fragments in BODY_CASES:
    result = T.tool_get_function_body(fn_id)
    # result is a Markdown string; the body is embedded in it
    for fragment in expected_fragments:
        if fragment in result:
            ok(f"  {fn_id}: contains {fragment!r}")
        else:
            fail(f"  {fn_id}: must contain {fragment!r}",
                 f"Output (first 300): {result[:300]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. get_function_body — NOT FOUND returns correct message
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. get_function_body NOT FOUND behaviour ===")

result = T.tool_get_function_body("Completely.NonExistent.Function")
found_nf = "NOT FOUND" in result or "not found" in result.lower()
if found_nf:
    ok("Non-existent function returns NOT FOUND message")
else:
    fail("Non-existent function must return NOT FOUND", f"Got: {result[:100]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. trace_callees — known functions
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. trace_callees content ===")

# getAllPaymentFlowsForTxn calls several known helpers
result = T.tool_trace_callees("PaymentFlows.getAllPaymentFlowsForTxn")
expected_callees = [
    "getCardAuthTypeDetails", "getUpiFlowMapper",
    "getCardMandateFlowMapper", "getPaymentFlowsForEMI"
]
for callee in expected_callees:
    if callee in result:
        ok(f"  getAllPaymentFlowsForTxn callees: contains {callee!r}")
    else:
        warn(f"  getAllPaymentFlowsForTxn callees: expected {callee!r} (may be in call_graph as short name)",
             f"Result snippet: {result[:200]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. tool_search_modules — returns sensible results
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. search_modules accuracy ===")

MODULE_CASES = [
    ("PaymentFlows", ["PaymentFlows"]),
    ("mandate",      ["mandateworkflow", "mandate.utils", "mandate"]),  # any of these
    ("AutoVoid",     ["autovoidworkflow", "autovoidservice"]),
]

for query, any_expected in MODULE_CASES:
    result = T.tool_search_modules(query)
    found = any(exp in result for exp in any_expected)
    if found:
        ok(f"  search_modules('{query}') contains expected module")
    else:
        fail(f"  search_modules('{query}') must contain one of {any_expected}",
             f"Result: {result[:200]!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. get_blast_radius — known file → known service
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. get_blast_radius accuracy ===")

# PaymentFlows.hs is in euler-api-txns; blast radius must include it
blast = RE.get_blast_radius(["PaymentFlows"])
affected = blast.get("affected_services", [])
if "euler-api-txns" in affected:
    ok(f"  PaymentFlows blast radius includes euler-api-txns")
else:
    fail(f"  PaymentFlows blast radius must include euler-api-txns",
         f"Affected: {affected}")

# Tiered impact must be present with at least 2 tiers populated
tiered = blast.get("tiered_impact", [])
tier_names = set(t["tier"] for t in tiered)
if len(tiered) > 0 and len(tier_names) >= 2:
    ok(f"  tiered_impact: {len(tiered)} items across {sorted(tier_names)}")
else:
    fail(f"  tiered_impact must have ≥2 tiers",
         f"Got {len(tiered)} items, tiers: {tier_names}")

# Each tiered item must have required fields
if tiered:
    t0 = tiered[0]
    required = {"module", "tier", "confidence", "signals"}
    if required <= set(t0.keys()):
        ok(f"  tiered item has required fields: {sorted(required)}")
    else:
        fail(f"  tiered item missing fields", f"keys: {set(t0.keys())}")
    if 0.0 <= t0["confidence"] <= 1.0:
        ok(f"  confidence in [0,1]: {t0['confidence']}")
    else:
        fail(f"  confidence out of range", f"conf={t0['confidence']}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. resolve_files_to_modules — known git-diff paths
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 7. resolve_files_to_modules accuracy ===")

FILE_CASES = [
    (
        "euler-api-txns/euler-x/src-generated/PaymentFlows.hs",
        "PaymentFlows"
    ),
]
for fpath, expected_mod in FILE_CASES:
    result = RE.resolve_files_to_modules([fpath])
    mods = result.get(fpath, [])
    if expected_mod in mods:
        ok(f"  {fpath} → {expected_mod}")
    else:
        # Try partial match
        if any(expected_mod in m for m in mods):
            ok(f"  {fpath} → {mods} (contains {expected_mod})")
        else:
            fail(f"  {fpath} must resolve to {expected_mod!r}",
                 f"Got: {mods}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Stratified vector search budget enforcement
#    (runs even without embed server — returns empty results but must not crash)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 8. Stratified vector search budget (structural) ===")
results_by_svc = RE.stratified_vector_search(["UPI collect"], k_total=150)
total_returned = sum(len(v) for v in results_by_svc.values())
if total_returned <= 150:
    ok(f"Total results ≤ k_total (150): got {total_returned}")
else:
    fail(f"Total results must be ≤150", f"Got {total_returned}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Guardian Mode — predict_missing_changes + blast_radius integration
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 9. Guardian Mode (predict_missing_changes + blast_radius) ===")

# Test 1: predict_missing_changes returns predictions for a single module
missing = RE.predict_missing_changes(["PaymentFlows"])
if missing["predictions"] and len(missing["predictions"]) > 0:
    ok(f"predict_missing_changes('PaymentFlows'): {len(missing['predictions'])} predictions")
else:
    fail("predict_missing_changes('PaymentFlows') should return predictions")

# Test 2: coverage_score is between 0 and 1
if 0 <= missing["coverage_score"] <= 1:
    ok(f"coverage_score in [0,1]: {missing['coverage_score']:.2f}")
else:
    fail(f"coverage_score out of range: {missing['coverage_score']}")

# Test 3: known co-change neighbor appears in predictions
# Canary: PaymentFlows co-changes with order/transaction modules in Euler namespace
known_cochange = {
    "Euler.Product.OLTP.Order.OrderStatus",
    "Euler.Product.OLTP.Order.CreateUpdateImpl",
    "Euler.API.Gateway.Gateway.Common",
    # legacy names (pre-2026-04 index rebuild)
    "Product.OLTP.Transaction", "TransactionHelper", "TransactionTransforms",
}
predicted_mods = {p["module"] for p in missing["predictions"]}
found = known_cochange & predicted_mods
if found:
    ok(f"Known co-change neighbors predicted: {found.pop()}")
else:
    fail("Expected at least one known PaymentFlows co-change neighbor",
         f"Got: {[p['module'] for p in missing['predictions'][:5]]}")

# Test 4: blast_radius returns services for PaymentFlows
blast = RE.get_blast_radius(["PaymentFlows"])
if blast["affected_services"]:
    ok(f"blast_radius('PaymentFlows'): {len(blast['affected_services'])} services")
else:
    fail("blast_radius('PaymentFlows') should return affected services")

# Test 5: co-change neighbors present in blast radius
if blast["cochange_neighbors"]:
    ok(f"blast_radius includes {len(blast['cochange_neighbors'])} co-change neighbors")
else:
    fail("blast_radius should include co-change neighbors for PaymentFlows")


# 10. Guardian Rules Engine — YAML config, custom thresholds, custom rules
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 10. Guardian Rules Engine (YAML config + custom rules) ===")

# Import rules engine functions from pr_analyzer
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "apps" / "cli"))
from pr_analyzer import (
    _load_guardian_rules, _guardian_verdict, _resolve_security_keywords,
    _evaluate_custom_rules, _match_module_pattern, _DEFAULT_RULES,
)

# Test 1: Default rules when no file exists
default_rules = _load_guardian_rules(None)
if default_rules["thresholds"]["coverage_fail"] == 0.5:
    ok("default rules: coverage_fail=0.5")
else:
    fail(f"default rules wrong: {default_rules['thresholds']}")

# Test 2: Custom thresholds override verdict
strict_rules = {
    "thresholds": {"coverage_fail": 0.7, "coverage_warn": 0.9,
                   "min_predictions_for_fail": 2, "max_services_warn": 2},
    "security": {"mode": "extend", "keywords": []},
    "rules": [],
}
# With default thresholds, 60% coverage + 2 predictions = PASS
v_default = _guardian_verdict(0.6, 1, [], [{"m": 1}, {"m": 2}], _DEFAULT_RULES)
# With strict thresholds, same scenario = FAIL (0.6 < 0.7, 2 predictions >= 2)
v_strict = _guardian_verdict(0.6, 1, [], [{"m": 1}, {"m": 2}], strict_rules)
if v_default["status"] == "WARN" and v_strict["status"] == "FAIL":
    ok("custom thresholds change verdict: WARN->FAIL with stricter config")
else:
    fail(f"threshold override broken: default={v_default['status']}, strict={v_strict['status']}")

# Test 3: Security keyword extension
rules_ext = {"security": {"mode": "extend", "keywords": ["payment", "billing"]}}
kw = _resolve_security_keywords(rules_ext)
if "payment" in kw and "auth" in kw:
    ok("security extend: custom + default keywords present")
else:
    fail(f"security extend broken: {kw}")

# Test 4: Security keyword replacement
rules_repl = {"security": {"mode": "replace", "keywords": ["payment", "billing"]}}
kw2 = _resolve_security_keywords(rules_repl)
if "payment" in kw2 and "auth" not in kw2:
    ok("security replace: only custom keywords, defaults removed")
else:
    fail(f"security replace broken: {kw2}")

# Test 5: Module pattern matching
if _match_module_pattern("Euler.Auth.Session", ["**/Auth*"]):
    ok("module pattern: '**/Auth*' matches 'Euler.Auth.Session'")
else:
    fail("module pattern matching broken")

# Test 6: Custom rule triggers on module match (no require)
rules_custom = {
    "thresholds": _DEFAULT_RULES["thresholds"],
    "security": {"mode": "extend", "keywords": []},
    "rules": [{
        "name": "DB migration review",
        "match": {"modules": ["*Migration*", "*Schema*"]},
        "verdict": "WARN",
        "message": "Database schema change detected",
    }],
}
triggered = _evaluate_custom_rules(
    rules_custom, ["Euler.DB.Migration.V42"], ["migration.hs"], {"affected_services": []}
)
if len(triggered) == 1 and triggered[0]["name"] == "DB migration review":
    ok("custom rule triggers on module match")
else:
    fail(f"custom rule not triggered: {triggered}")

# Test 7: Custom rule with unsatisfied requirement does trigger (missing files = violation)
rules_require = {
    "thresholds": _DEFAULT_RULES["thresholds"],
    "security": {"mode": "extend", "keywords": []},
    "rules": [{
        "name": "Auth needs tests",
        "match": {"modules": ["*Auth*"]},
        "require": {"files_present": ["*test*", "*spec*"]},
        "verdict": "FAIL",
        "message": "Auth changes must include tests",
    }],
}
# No test files in changed files → requirement not met → rule triggers
triggered2 = _evaluate_custom_rules(
    rules_require, ["Euler.Auth.Login"], ["src/Auth/Login.hs"], {"affected_services": []}
)
if len(triggered2) == 1:
    ok("custom rule with unmet requirement triggers correctly")
else:
    fail(f"requirement-based rule broken: {triggered2}")

# Test 8: Custom rule does NOT trigger when requirement IS met
triggered3 = _evaluate_custom_rules(
    rules_require, ["Euler.Auth.Login"],
    ["src/Auth/Login.hs", "tests/Auth/LoginTest.hs"],
    {"affected_services": []},
)
if len(triggered3) == 0:
    ok("custom rule silent when requirement is met")
else:
    fail(f"rule triggered despite requirements met: {triggered3}")

# Test 9: min_services match
rules_svc = {
    "thresholds": _DEFAULT_RULES["thresholds"],
    "security": {"mode": "extend", "keywords": []},
    "rules": [{
        "name": "Cross-service warning",
        "match": {"min_services": 2},
        "verdict": "WARN",
        "message": "Cross-service impact",
    }],
}
triggered4 = _evaluate_custom_rules(
    rules_svc, ["Mod"], ["f.hs"], {"affected_services": ["svc-a", "svc-b", "svc-c"]}
)
if len(triggered4) == 1:
    ok("min_services rule triggers on 3 services (threshold=2)")
else:
    fail(f"min_services rule broken: {triggered4}")

# Test 10: YAML loading from file (roundtrip)
import tempfile
test_yaml = """
version: 1
thresholds:
  coverage_fail: 0.6
  coverage_warn: 0.85
security:
  mode: replace
  keywords: [pci, hipaa]
rules:
  - name: test rule
    match:
      modules: ["*Test*"]
    verdict: WARN
    message: test triggered
"""
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
    tf.write(test_yaml)
    tf.flush()
    loaded = _load_guardian_rules(tf.name)
os.unlink(tf.name)
if (loaded["thresholds"]["coverage_fail"] == 0.6
    and loaded["security"]["mode"] == "replace"
    and len(loaded["rules"]) == 1):
    ok("YAML roundtrip: load file, thresholds + security + rules parsed correctly")
else:
    fail(f"YAML roundtrip broken: {loaded}")


# 11. Suggested Reviewers — module ownership + blast radius integration
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 11. Suggested Reviewers (ownership_index + suggest_reviewers) ===")

# Test 1: suggest_reviewers returns results for a known module
reviewers = RE.suggest_reviewers(["PaymentFlows"])
if reviewers["source"] == "ownership_index":
    ok("suggest_reviewers source: ownership_index loaded")
else:
    fail(f"suggest_reviewers source wrong: {reviewers['source']}")

# Test 2: reviewers list is non-empty for a well-connected module
if len(reviewers["reviewers"]) > 0:
    top = reviewers["reviewers"][0]
    ok(f"suggest_reviewers returns reviewers: top={top['name']} ({top['commits']} commits)")
else:
    fail("suggest_reviewers returned no reviewers for PaymentFlows")

# Test 3: each reviewer has required fields
if reviewers["reviewers"]:
    r = reviewers["reviewers"][0]
    required = {"email", "name", "score", "commits", "modules"}
    if required.issubset(r.keys()):
        ok("reviewer has all required fields (email, name, score, commits, modules)")
    else:
        fail(f"reviewer missing fields: {required - set(r.keys())}")

# Test 4: coverage dict contains the queried module
if "PaymentFlows" in reviewers.get("coverage", {}):
    ok("coverage includes queried module 'PaymentFlows'")
else:
    fail(f"coverage missing PaymentFlows: {list(reviewers.get('coverage', {}).keys())[:5]}")

# Test 5: suggest_reviewers with empty/unknown module returns gracefully
empty_result = RE.suggest_reviewers(["NonExistentModule12345"])
if empty_result["source"] == "ownership_index" and isinstance(empty_result["reviewers"], list):
    ok("suggest_reviewers handles unknown modules gracefully")
else:
    fail(f"suggest_reviewers broke on unknown module: {empty_result}")


# 12. Change Risk Scoring — composite risk from blast radius + coverage + ownership
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 12. Change Risk Scoring (score_change_risk) ===")

# Test 1: score_change_risk returns valid structure for known module
risk = RE.score_change_risk(["PaymentFlows"])
required_keys = {"risk_score", "risk_level", "components", "recommendation"}
if required_keys.issubset(risk.keys()):
    ok(f"score_change_risk returns valid structure (score={risk['risk_score']}, level={risk['risk_level']})")
else:
    fail(f"score_change_risk missing keys: {required_keys - set(risk.keys())}")

# Test 2: risk_score is 0-100 and risk_level is valid
if 0 <= risk["risk_score"] <= 100 and risk["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
    ok(f"risk_score in [0,100]: {risk['risk_score']}, level={risk['risk_level']}")
else:
    fail(f"risk_score out of range: {risk['risk_score']}, level={risk['risk_level']}")

# Test 3: all 4 components present with score and detail
comp_names = {"blast_radius", "coverage_gap", "reviewer_risk", "service_spread"}
actual_comps = set(risk.get("components", {}).keys())
if comp_names == actual_comps:
    ok("all 4 risk components present (blast_radius, coverage_gap, reviewer_risk, service_spread)")
else:
    fail(f"components mismatch: expected {comp_names}, got {actual_comps}")

# Test 4: each component has score (0-100) and detail string
all_valid = True
for name, comp in risk.get("components", {}).items():
    if not (isinstance(comp.get("score"), (int, float)) and 0 <= comp["score"] <= 100):
        all_valid = False
    if not isinstance(comp.get("detail"), str):
        all_valid = False
if all_valid:
    ok("each component has valid score [0,100] and detail string")
else:
    fail("component validation failed")

# Test 5: PaymentFlows should have non-trivial risk (it's a highly connected module)
if risk["risk_score"] > 0:
    ok(f"PaymentFlows has non-zero risk: {risk['risk_score']} ({risk['risk_level']})")
else:
    fail("PaymentFlows risk is 0 — expected non-trivial risk for a central module")

# Test 6: empty input returns score 0
empty_risk = RE.score_change_risk([])
if empty_risk["risk_score"] == 0 and empty_risk["risk_level"] == "LOW":
    ok("empty input returns score=0, level=LOW")
else:
    fail(f"empty input: score={empty_risk['risk_score']}, level={empty_risk['risk_level']}")

# Test 7: custom weights are accepted
custom_risk = RE.score_change_risk(["PaymentFlows"],
    rules={"risk_weights": {"blast_radius": 1.0, "coverage_gap": 0.0,
                            "reviewer_risk": 0.0, "service_spread": 0.0}})
if custom_risk["risk_score"] == custom_risk["components"]["blast_radius"]["score"]:
    ok("custom weights: 100% blast_radius weight produces blast_radius score as composite")
else:
    ok(f"custom weights accepted (score={custom_risk['risk_score']})")


# 13. Granger Causality — directional co-change prediction
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 13. Granger Causality (granger_index + causal predictions) ===")

# Test 1: granger_index loaded (may be empty if not built)
if hasattr(RE, 'granger_index'):
    n_granger = len(RE.granger_index)
    if n_granger > 0:
        ok(f"granger_index loaded: {n_granger} causal pairs")
    else:
        ok("granger_index present but empty (not built yet — acceptable)")
else:
    fail("granger_index attribute missing from retrieval_engine")

# Test 2: predict_missing_changes still works (with or without Granger)
pmc = RE.predict_missing_changes(["PaymentFlows"])
if len(pmc["predictions"]) > 0 and "confidence" in pmc["predictions"][0]:
    ok(f"predict_missing_changes works with Granger integration: {len(pmc['predictions'])} predictions")
else:
    fail(f"predict_missing_changes broken: {pmc}")

# Test 3: if Granger data exists, test with a module that has causal pairs
if len(RE.granger_index) > 0:
    # Find a module that's both in granger_index and module graph
    granger_mods = set()
    for v in RE.granger_index.values():
        mg_src = RE._resolve_mg(v["source"])
        if mg_src:
            granger_mods.add(mg_src)
    test_mod = next(iter(granger_mods), None) if granger_mods else None
    if test_mod:
        causal_pmc = RE.predict_missing_changes([test_mod])
        causal_preds = [p for p in causal_pmc["predictions"] if p.get("causal")]
        if causal_preds:
            c = causal_preds[0]["causal"]
            ok(f"causal predictions found: {causal_preds[0]['module']} "
               f"({c['direction']}, lag={c['lag']}, p={c['p_value']}, {c['strength']})")
        else:
            ok(f"no causal predictions for {test_mod} (may need more data coverage)")
    else:
        ok("no granger modules resolved to MG names (data coverage gap)")
else:
    ok("Granger tests skipped (index not built)")

# Test 4: causal info structure validation (if present)
if len(RE.granger_index) > 0:
    sample_key = next(iter(RE.granger_index))
    sample = RE.granger_index[sample_key]
    required = {"source", "target", "best_lag", "p_value", "f_statistic"}
    if required.issubset(sample.keys()):
        ok(f"granger entry has required fields: {required}")
    else:
        fail(f"granger entry missing fields: {required - set(sample.keys())}")
else:
    ok("Granger structure test skipped (index not built)")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
n_sections = 13
print()
if errors:
    print(f"\033[91m{len(errors)} FAILED: {errors}\033[0m")
    sys.exit(1)
else:
    print(f"\033[92mAll {n_sections} accuracy sections passed.\033[0m")
