"""Guardrail auto-surface for chat responses.

Generic helpers that inspect a completed chat turn and return a markdown
block summarising guardrails for any module the turn touched.

- Triggered by `HR_CHAT_AUTO_SURFACE_GUARDRAILS=1` (default on)
- Pulls 'What must stay true' and 'Review checklist for changes' sections
  out of guardrail markdown stored in `retrieval_engine.guardrails_content`
- No framework dependency: returns a string. The caller decides how to render
  it (Chainlit message, CLI print, HTTP response, etc.)
"""
from __future__ import annotations
import os

_GR_SECTION_LABELS = ("What must stay true", "Review checklist for changes")


def extract_bolded_section(md: str, label: str) -> str:
    """Extract the bullet/checklist block that follows `**<label>:**` in a guardrail doc."""
    needle = f"**{label}:**"
    idx = md.find(needle)
    if idx < 0:
        return ""
    body = md[idx + len(needle):]
    out_lines = []
    for line in body.splitlines():
        s = line.strip()
        if not out_lines and not s:
            continue
        if s.startswith(("- ", "* ", "- [")):
            out_lines.append(line.rstrip())
            continue
        if out_lines and not s:
            break
        if out_lines and not s.startswith(("- ", "* ")) and not s.startswith("**"):
            out_lines.append(line.rstrip())
            continue
        if s.startswith("**") and out_lines:
            break
    return "\n".join(out_lines).strip()


def candidate_modules_from_chat(tool_log: list, full_response: str,
                                 guardrail_keys: list) -> list:
    """Gather module-name candidates from tool args + substring matches in the answer."""
    candidates: set = set()
    for entry in tool_log or []:
        args = entry.get("args") or {}
        for key in ("modules", "module_names", "module", "name", "query"):
            v = args.get(key)
            if isinstance(v, list):
                candidates.update(str(x) for x in v if x)
            elif isinstance(v, str) and v:
                candidates.add(v)

    low = (full_response or "").lower()
    for k in guardrail_keys or []:
        short = k.split("::")[-1].split("__")[-1].replace(".md", "")
        if short and len(short) > 3 and short.lower() in low:
            candidates.add(k)
    return list(candidates)


def surface_guardrails(tool_log: list, full_response: str,
                       get_guardrails_fn, guardrails_content: dict,
                       guardrails_index: dict | None = None) -> str:
    """Build a markdown block with guardrail sections for every module touched by this turn.

    Args:
        tool_log: list of {"tool": name, "args": {...}} entries.
        full_response: the final streamed answer text.
        get_guardrails_fn: callable `list[str] -> dict` — typically
            `retrieval_engine.get_guardrails`.
        guardrails_content: `retrieval_engine.guardrails_content` (module → md string).
        guardrails_index: optional `retrieval_engine.guardrails_index` (module → meta dict).

    Returns empty string when disabled, no candidates, or nothing surfaceable.
    """
    if os.environ.get("HR_CHAT_AUTO_SURFACE_GUARDRAILS", "1") == "0":
        return ""
    if not guardrails_content and not guardrails_index:
        return ""

    gr_keys = list((guardrails_content or {}).keys()) + \
              list((guardrails_index or {}).keys())
    candidates = candidate_modules_from_chat(tool_log, full_response, gr_keys)
    if not candidates:
        return ""

    try:
        results = get_guardrails_fn(candidates) or {}
    except Exception as e:
        print(f"[guardrail_autosurface] get_guardrails failed: {e!r}")
        return ""

    surfaced = []
    seen_keys: set = set()
    for mod, entry in results.items():
        if not isinstance(entry, dict) or not entry.get("has_guardrail"):
            continue
        content = entry.get("content") or entry.get("guardrail") or ""
        if not content:
            content = (guardrails_content or {}).get(mod, "")
        if not content:
            continue

        dedup_key = (entry.get("score"), content[:80])
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        sections = []
        for label in _GR_SECTION_LABELS:
            body = extract_bolded_section(content, label)
            if body:
                sections.append(f"**{label}:**\n{body}")
        if not sections:
            continue
        score = entry.get("score", 0) or 0
        header = f"### Guardrail — `{mod}` (criticality {score:.2f})"
        surfaced.append(header + "\n\n" + "\n\n".join(sections))

    if not surfaced:
        return ""
    return ("\n\n---\n\n### Protective guardrails for modules touched by this answer\n\n"
            + "\n\n---\n\n".join(surfaced))
