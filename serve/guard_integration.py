"""Thin integration layer between HR's check_my_changes and the Guard checker.

Guard ships bundled with HR at `hyperretrieval/guardrails/comment_code_checker.py`.
This module loads it by default so installed users get Guard out of the box. For
development, set `HR_GUARD_PATH` to override with a local prototype checkout
(typically `~/lab/experiments/guardrails_prototype/`).

Public API:
    run_guard_on_files(paths: list[str]) -> list[dict]
        Returns a list of finding dicts: {severity, pattern, file, line,
        message, snippet}. Empty list if Guard is not available.

    summarize_findings(findings: list[dict]) -> dict
        Returns {"count": N, "critical": N, "warning": N, "patterns": [...]}
        for quick verdict aggregation.

Env vars:
    HR_GUARD_PATH    - directory containing `comment_code_checker.py`.
                       When set, used verbatim. When unset, falls back to the
                       bundled `hyperretrieval/guardrails/` checker.
    HR_GUARD_DISABLE - set to "1" to disable Guard entirely
"""
from __future__ import annotations
import importlib.util
import os
import pathlib
from typing import Any

_GUARD_MOD: Any = None
_GUARD_LOAD_ERR: Exception | None = None

# Bundled checker that ships with HR (vendored from the prototype). Used when
# HR_GUARD_PATH is not set.
_BUNDLED_CHECKER = (
    pathlib.Path(__file__).resolve().parent.parent
    / "guardrails" / "comment_code_checker.py"
)


def _load_guard():
    global _GUARD_MOD, _GUARD_LOAD_ERR
    if _GUARD_MOD is not None or _GUARD_LOAD_ERR is not None:
        return
    if os.environ.get("HR_GUARD_DISABLE", "0") == "1":
        _GUARD_LOAD_ERR = RuntimeError("HR_GUARD_DISABLE=1")
        return
    override = os.environ.get("HR_GUARD_PATH")
    if override:
        candidate = pathlib.Path(override) / "comment_code_checker.py"
    else:
        candidate = _BUNDLED_CHECKER
    if not candidate.exists():
        _GUARD_LOAD_ERR = FileNotFoundError(str(candidate))
        return
    try:
        spec = importlib.util.spec_from_file_location("hr_guard_prototype", str(candidate))
        if spec is None or spec.loader is None:
            raise ImportError(f"spec_from_file_location returned None for {candidate}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _GUARD_MOD = mod
    except Exception as e:
        _GUARD_LOAD_ERR = e


def available() -> bool:
    _load_guard()
    return _GUARD_MOD is not None


def run_guard_on_files(paths: list) -> list:
    _load_guard()
    if _GUARD_MOD is None:
        return []
    findings_out = []
    checker = getattr(_GUARD_MOD, "check_file", None)
    if not callable(checker):
        return []
    for p in paths or []:
        path_obj = pathlib.Path(p)
        if not path_obj.is_file():
            continue
        try:
            raw_findings = checker(str(path_obj)) or []
        except Exception as e:
            print(f"[guard] {p}: {e!r}")
            continue
        for f in raw_findings:
            findings_out.append({
                "severity": getattr(f, "severity", "WARNING"),
                "pattern": getattr(f, "pattern", "unknown"),
                "file": getattr(f, "file", str(path_obj)),
                "line": getattr(f, "line", 0),
                "message": getattr(f, "message", ""),
                "snippet": getattr(f, "snippet", ""),
            })
    return findings_out


def summarize_findings(findings: list) -> dict:
    crit = sum(1 for f in findings if str(f.get("severity", "")).upper() == "CRITICAL")
    warn = sum(1 for f in findings if str(f.get("severity", "")).upper() == "WARNING")
    patterns = sorted({f.get("pattern") for f in findings if f.get("pattern")})
    return {
        "count": len(findings),
        "critical": crit,
        "warning": warn,
        "patterns": patterns,
    }
