"""Verify chat guardrail auto-surface logic against real guardrail markdown.

Generic test — imports `serve.guardrail_autosurface` directly, bypassing
the Chainlit layer.
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "serve"))
os.environ["ARTIFACT_DIR"] = "/home/beast/projects/workspaces/juspay/artifacts"

import retrieval_engine as RE
from retrieval_engine import guardrails_content
from serve.guardrail_autosurface import (
    extract_bolded_section, candidate_modules_from_chat, surface_guardrails,
)

# Seed guardrails_content from disk (no engine load)
gr_dir = pathlib.Path(os.environ["ARTIFACT_DIR"]) / "guardrails"
for gf in sorted(gr_dir.glob("*.md")):
    guardrails_content[gf.stem] = gf.read_text()

print(f"guardrails_content loaded: {len(guardrails_content)} docs")


# ─── TEST 1: extract_bolded_section on a real guardrail ───
sample_key = sorted(guardrails_content.keys())[0]
sample_md = guardrails_content[sample_key]
wmst = extract_bolded_section(sample_md, "What must stay true")
chk = extract_bolded_section(sample_md, "Review checklist for changes")
print(f"\nTEST 1: extract sections from '{sample_key[:50]}...'")
print(f"  'What must stay true' bytes: {len(wmst)}")
print(f"  'Review checklist' bytes: {len(chk)}")
assert wmst, "What must stay true should be extracted"
assert chk, "Review checklist should be extracted"
print("  PASS")


# ─── TEST 2: surface with tool_log referencing a real module ───
tool_log = [
    {"tool": "check_criticality", "args": {"modules": [sample_key]}},
    {"tool": "search_modules", "args": {"query": "how does transaction work"}},
]
full_response = "The transaction flow starts in the OLTP::Transaction module..."
print(f"\nTEST 2: surface_guardrails with real module")
os.environ["HR_CHAT_AUTO_SURFACE_GUARDRAILS"] = "1"
out = surface_guardrails(tool_log, full_response,
                          get_guardrails_fn=RE.get_guardrails,
                          guardrails_content=guardrails_content)
print(f"  Returned {len(out)} chars")
assert "What must stay true" in out
assert "Review checklist" in out
assert "Guardrail —" in out
print(f"  First 180 chars: {out[:180]}")
print("  PASS")


# ─── TEST 3: OFF flag is a no-op ───
print(f"\nTEST 3: OFF flag respected")
os.environ["HR_CHAT_AUTO_SURFACE_GUARDRAILS"] = "0"
off_out = surface_guardrails(tool_log, full_response,
                              get_guardrails_fn=RE.get_guardrails,
                              guardrails_content=guardrails_content)
assert off_out == ""
print("  PASS")


# ─── TEST 4: No matching candidates → empty ───
print(f"\nTEST 4: no candidates in tool_log or text")
os.environ["HR_CHAT_AUTO_SURFACE_GUARDRAILS"] = "1"
empty_out = surface_guardrails(
    [{"tool": "foo", "args": {"modules": ["CompletelyFakeModule"]}}],
    "unrelated answer text",
    get_guardrails_fn=RE.get_guardrails,
    guardrails_content=guardrails_content,
)
assert empty_out == ""
print("  PASS")

print("\n=== guardrail_autosurface VERIFIED END-TO-END ===")
