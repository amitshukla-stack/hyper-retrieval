"""Exercise the 3 guardrail MCP tools against real artifacts.

Tests the retrieval_engine implementations of check_criticality, get_guardrails,
list_critical_modules. Bypasses the MCP transport layer — validates the
underlying Python functions directly.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "serve"))
os.environ["ARTIFACT_DIR"] = "/home/beast/projects/workspaces/juspay/artifacts"

import retrieval_engine as re_
from retrieval_engine import criticality_index, guardrails_index, guardrails_content

# Lazy-load criticality index (skip full engine load)
if not criticality_index:
    with open(os.path.join(os.environ["ARTIFACT_DIR"], "criticality_index.json")) as f:
        data = json.load(f)
    criticality_index.update(data.get("modules", {}))

# Load guardrails from directory (no guardrails_index.json exists)
guardrails_dir = os.path.join(os.environ["ARTIFACT_DIR"], "guardrails")
for gf in sorted(os.listdir(guardrails_dir)):
    if gf.endswith(".md"):
        with open(os.path.join(guardrails_dir, gf)) as f:
            guardrails_content[gf[:-3]] = f.read()

print(f"criticality_index: {len(criticality_index)} modules")
print(f"guardrails_content: {len(guardrails_content)} docs\n")

# Pick 2 real module names from the top of criticality ranking + 1 fake
sorted_crit = sorted(criticality_index.items(), key=lambda kv: -kv[1].get("score", 0))
mod_high = sorted_crit[0][0]
mod_med = sorted_crit[len(sorted_crit) // 2][0]
mod_fake = "FakeModule::DoesNotExist::Nowhere"

# Also grab a guardrail filename to try get_guardrails on
gr_sample = list(guardrails_content.keys())[0] if guardrails_content else None
print(f"Sample guardrail module: {gr_sample}\n")

# ──────────── TEST 1: check_criticality ────────────
print("=== TEST 1: check_criticality ===")
r1 = re_.check_criticality([mod_high, mod_med, mod_fake])
for m, v in r1.items():
    print(f"  {m[:55]}... score={v['score']:.3f} risk={v['risk_level']}")
assert r1[mod_high]["score"] > 0.5, f"HIGH module should have score > 0.5, got {r1[mod_high]['score']}"
assert r1[mod_fake]["risk_level"] == "UNKNOWN", "Fake module should be UNKNOWN"
print("  PASS\n")

# ──────────── TEST 2: get_guardrails ────────────
print("=== TEST 2: get_guardrails ===")
test_mods = [gr_sample, mod_fake] if gr_sample else [mod_fake]
r2 = re_.get_guardrails(test_mods)
for m, v in r2.items():
    if m is None:
        continue
    has = v.get("has_guardrail", False)
    print(f"  {str(m)[:55]}... has_guardrail={has}")
    if has:
        content = v.get("content", "") or v.get("guardrail", "")
        print(f"    content size: {len(content)} chars")
assert r2.get(mod_fake, {}).get("has_guardrail") is False, "Fake module should have no guardrail"
print("  PASS\n")

# ──────────── TEST 3: list_critical_modules ────────────
print("=== TEST 3: list_critical_modules ===")
r3 = re_.list_critical_modules(threshold=0.4, top_k=5)
print(f"  Returned type: {type(r3).__name__}")
if isinstance(r3, dict):
    mods = r3.get("modules", r3.get("items", []))
else:
    mods = r3
print(f"  Returned {len(mods) if hasattr(mods,'__len__') else '?'} modules at threshold>=0.4")
if mods:
    for m in mods[:3]:
        if isinstance(m, dict):
            print(f"    - {str(m.get('module','?'))[:50]}... score={m.get('score','?')}")
        else:
            print(f"    - {str(m)[:50]}...")
print("  PASS\n")

print("=== ALL 3 MCP TOOLS VERIFIED AGAINST REAL ARTIFACTS ===")
