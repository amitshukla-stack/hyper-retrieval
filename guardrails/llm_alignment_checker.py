"""
LLM-powered comment-code alignment checker.

Extracts (comment, code) pairs from Python/Rust/Haskell source and asks an
LLM whether the comment accurately describes the code. Language-agnostic.

Uses the same LLM config as the rest of HyperRetrieval:
  LLM_API_KEY  (env) — API key
  LLM_BASE_URL (env) — base URL, e.g. https://grid.ai.juspay.net
  LLM_MODEL    (env) — model name, default: kimi-latest

Gate: TP >= 0.80, FP <= 0.20, latency <= 2s/file.
"""
import ast
import re
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import openai as _openai
    _openai_available = True
except ImportError:
    _openai_available = False

from comment_code_checker import Finding

_LLM_API_KEY  = os.environ.get("LLM_API_KEY", "")
_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://grid.ai.juspay.net")
_LLM_MODEL    = os.environ.get("LLM_MODEL", "kimi-latest")

_ALIGNMENT_PROMPT = """\
Review this comment against the code.

Comment: {comment}
Code: {code}

MISALIGNED = code does the OPPOSITE or NOTHING of what the comment claims.
ALIGNED = code implements what the comment describes (even if imperfectly).

Reply with EXACTLY this format (verdict first, one word of reason after):
VERDICT: ALIGNED
or
VERDICT: MISALIGNED
"""


@dataclass
class CommentCodePair:
    language: str
    filename: str
    line: int
    comment: str
    code: str


def _extract_python(source: str, filename: str) -> list[CommentCodePair]:
    pairs = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return pairs
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        docstring = ast.get_docstring(node)
        if docstring:
            func_src = "\n".join(lines[node.lineno - 1: node.end_lineno])
            pairs.append(CommentCodePair("python", filename, node.lineno, docstring, func_src[:600]))
            continue
        def_line = node.lineno - 1
        if def_line > 0 and lines[def_line - 1].strip().startswith("#"):
            comment = lines[def_line - 1].strip().lstrip("# ").strip()
            func_src = "\n".join(lines[def_line: node.end_lineno])
            pairs.append(CommentCodePair("python", filename, node.lineno, comment, func_src[:600]))
    return pairs


def _extract_rust(source: str, filename: str) -> list[CommentCodePair]:
    pairs = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        comment_lines = []
        while i < len(lines) and re.match(r"\s*//[/!]?\s?(.*)", lines[i]):
            m = re.match(r"\s*//[/!]?\s?(.*)", lines[i])
            comment_lines.append(m.group(1))
            i += 1
        if not comment_lines:
            i += 1
            continue
        j = i
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and re.search(r"\bfn\s+\w+", lines[j]):
            func_src = "\n".join(lines[j: j + 20])
            comment = " ".join(comment_lines).strip()
            if len(comment) > 10:
                pairs.append(CommentCodePair("rust", filename, j + 1, comment, func_src[:600]))
        i = j + 1 if j > i else i
    return pairs


def _extract_haskell(source: str, filename: str) -> list[CommentCodePair]:
    pairs = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        comment_lines = []
        while i < len(lines) and re.match(r"\s*--\s*(.*)", lines[i]):
            m = re.match(r"\s*--\s*(.*)", lines[i])
            comment_lines.append(m.group(1))
            i += 1
        if not comment_lines:
            i += 1
            continue
        j = i
        while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith("--")):
            j += 1
        if j < len(lines) and re.match(r"^[a-z_]\w*\s+(::|\w)", lines[j]):
            func_src = "\n".join(lines[j: j + 15])
            comment = " ".join(comment_lines).strip()
            if len(comment) > 10:
                pairs.append(CommentCodePair("haskell", filename, j + 1, comment, func_src[:600]))
        i = j + 1 if j > i else i
    return pairs


def extract_pairs(source: str, filename: str, language: Optional[str] = None) -> list[CommentCodePair]:
    if language is None:
        ext = Path(filename).suffix.lower()
        language = {".py": "python", ".rs": "rust", ".hs": "haskell"}.get(ext, ext.lstrip("."))
    if language == "python":
        return _extract_python(source, filename)
    elif language == "rust":
        return _extract_rust(source, filename)
    elif language == "haskell":
        return _extract_haskell(source, filename)
    return []


def _call_llm(comment: str, code: str, timeout: float = 10.0) -> tuple[str, str]:
    """Returns (verdict, reason). verdict in {ALIGNED, MISALIGNED, PARTIAL, ERROR}."""
    if not _openai_available:
        return "ERROR", "openai package not installed"
    if not _LLM_API_KEY:
        return "ERROR", "LLM_API_KEY not set"
    try:
        client = _openai.OpenAI(base_url=_LLM_BASE_URL, api_key=_LLM_API_KEY)
        prompt = _ALIGNMENT_PROMPT.format(comment=comment[:300], code=code[:600])
        resp = client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.0,
            timeout=timeout,
        )
        raw = resp.choices[0].message.content.strip()
        # Primary: find explicit VERDICT: marker anywhere in response
        m = re.search(r"VERDICT:\s*(ALIGNED|MISALIGNED|PARTIAL)", raw, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()
            reason = raw[m.end():].strip()[:80]
        else:
            # Fallback: scan ONLY the last 300 chars with word boundaries
            # (avoids matching "MISALIGNED" in the task-description preamble)
            tail = raw[-300:]
            wm = re.search(r"\b(MISALIGNED|PARTIAL|ALIGNED)\b", tail, re.IGNORECASE)
            if wm:
                verdict = wm.group(1).upper()
                reason = tail[wm.end():].strip()[:80]
            else:
                verdict = "ERROR"
                reason = raw[:80]
        return verdict, reason
    except Exception as e:
        return "ERROR", str(e)[:80]


def check_llm_alignment(source: str, filename: str = "<stdin>",
                        language: Optional[str] = None) -> list[Finding]:
    """Run LLM alignment check on all comment-code pairs in source."""
    pairs = extract_pairs(source, filename, language)
    findings = []
    for pair in pairs:
        verdict, reason = _call_llm(pair.comment, pair.code)
        if verdict in ("MISALIGNED", "PARTIAL"):
            severity = "critical" if verdict == "MISALIGNED" else "warning"
            findings.append(Finding(
                file=filename,
                line=pair.line,
                pattern="llm-comment-alignment",
                severity=severity,
                message=f"Comment-code {verdict.lower()}: {reason}",
                comment=pair.comment[:100],
                code=pair.code[:100],
            ))
    return findings
