#!/usr/bin/env python3
"""
pr_analyzer.py — PR blast-radius analysis for your codebase

Usage:
  # Pipe git diff output (most common in CI)
  git diff main...HEAD --name-only | python3 pr_analyzer.py

  # Explicit files
  python3 pr_analyzer.py --files euler-api-gateway/src/Euler/API/Gateway/Routes.hs

  # With LLM explanation (~30s, needs LLM_API_KEY)
  git diff main...HEAD --name-only | python3 pr_analyzer.py --explain --persona reliability_engineer

  # JSON output for CI pipelines
  git diff main...HEAD --name-only | python3 pr_analyzer.py --format json

  # Security gate (exits non-zero if security-sensitive modules touched)
  git diff main...HEAD --name-only | python3 pr_analyzer.py --check security
"""
import argparse, fnmatch, json, os, pathlib, sys

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

_REPO = pathlib.Path(__file__).parent.parent.parent   # apps/cli → apps → repo root
sys.path.insert(0, str(_REPO / "serve"))              # for retrieval_engine.py
sys.path.insert(0, str(_REPO))                        # for tools.py (if needed)

# Lazy import — RE may not be available if serve/ deps are absent or not initialized
_RE_AVAILABLE = False
RE = None
try:
    import retrieval_engine as RE  # type: ignore
    _RE_AVAILABLE = True
except ImportError as _re_import_err:
    pass  # RE stays None; functions check _RE_AVAILABLE before calling RE methods


def _require_re(feature: str = "this feature") -> None:
    """Raise a clear error if retrieval_engine isn't available."""
    if not _RE_AVAILABLE:
        raise RuntimeError(
            f"retrieval_engine is required for {feature} but could not be imported. "
            "Ensure serve/ is on sys.path and retrieval_engine.py dependencies are installed. "
            "Run: pip install lancedb networkx"
        )

# ── Security heuristics ───────────────────────────────────────────────────────
_SECURITY_KEYWORDS = {
    "auth", "token", "credential", "webhook", "verify", "pan", "cvv",
    "encrypt", "integrity", "secret", "password", "hmac", "signature",
    "session", "oauth", "jwt", "cert", "key",
}


def _flag_security(modules: list) -> list:
    return [m for m in modules if any(kw in m.lower() for kw in _SECURITY_KEYWORDS)]


# ── Output formatting ─────────────────────────────────────────────────────────

def _md_table(rows: list) -> str:
    if not rows:
        return "_No affected modules found._"
    headers = ["Module", "Service", "Relation", "Hop"]
    cols = [[h] + [str(r.get(h.lower(), "")) for r in rows] for h in headers]
    widths = [max(len(c) for c in col) for col in cols]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    def row_str(values):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(values, widths)) + " |"
    lines = [row_str(headers), sep]
    for r in rows:
        lines.append(row_str([str(r.get(h.lower(),"")) for h in headers]))
    return "\n".join(lines)


def _print_summary(files, seed_modules, unresolved, blast, fmt):
    all_affected = seed_modules + [n["module"] for n in blast["import_neighbors"]]
    sec_flagged  = _flag_security(all_affected)

    rows = []
    for n in blast["import_neighbors"]:
        rows.append({
            "module":   n["module"],
            "service":  n["service"],
            "relation": f"import ({n['direction']})",
            "hop":      n["hop"],
        })
    for n in blast["cochange_neighbors"]:
        rows.append({
            "module":   n["module"],
            "service":  "",
            "relation": f"co-change (w={n['weight']})",
            "hop":      n["hop"],
        })

    if fmt == "json":
        output = {
            "changed_files":     files,
            "seed_modules":      seed_modules,
            "unresolved_files":  unresolved,
            "affected_services": blast["affected_services"],
            "security_flagged":  sec_flagged,
            "import_neighbors":  blast["import_neighbors"],
            "cochange_neighbors": blast["cochange_neighbors"],
        }
        print(json.dumps(output, indent=2))
        return sec_flagged

    # Markdown output
    print(f"\n## PR Blast Radius\n")
    print(f"**Changed files:** {len(files)}")
    print(f"**Resolved modules:** {', '.join(seed_modules) or 'none'}")
    if unresolved:
        print(f"**Unresolved files:** {', '.join(unresolved)}")
    print(f"**Affected services:** {', '.join(blast['affected_services']) or 'none'}\n")

    if sec_flagged:
        print("### ⚠️  Security-sensitive modules touched")
        for m in sec_flagged:
            print(f"  - `{m}`")
        print()

    print(_md_table(rows))
    return sec_flagged


def _guardian_report(files, seed_mods, unresolved, blast, missing, fmt, rules=None,
                     reviewers=None):
    """Generate a Guardian Mode report: blast radius + missing changes + verdict."""
    if rules is None:
        rules = _DEFAULT_RULES
    if reviewers is None:
        reviewers = {"reviewers": [], "source": "unavailable"}
    coverage = missing.get("coverage_score", 1.0)
    predictions = missing.get("predictions", [])
    sec_keywords = _resolve_security_keywords(rules)
    sec_flagged = [m for m in seed_mods + [n["module"] for n in blast["import_neighbors"]]
                   if any(kw in m.lower() for kw in sec_keywords)]
    n_services = len(blast["affected_services"])
    n_import = len(blast["import_neighbors"])
    n_cochange = len(blast["cochange_neighbors"])

    # Evaluate custom rules
    custom_triggered = _evaluate_custom_rules(rules, seed_mods, files, blast)

    # Build base verdict from thresholds
    verdict = _guardian_verdict(coverage, n_services, sec_flagged, predictions, rules)

    # Custom rules can escalate verdict
    for cr in custom_triggered:
        cr_status = cr.get("verdict", "WARN")
        if cr_status == "FAIL" and verdict["status"] != "FAIL":
            verdict = {"status": "FAIL", "reason": cr.get("message", cr.get("name", "Custom rule failed"))}
        elif cr_status == "WARN" and verdict["status"] == "PASS":
            verdict = {"status": "WARN", "reason": cr.get("message", cr.get("name", "Custom rule warning"))}

    if fmt == "json":
        return json.dumps({
            "changed_files": files,
            "seed_modules": seed_mods,
            "unresolved_files": unresolved,
            "blast_radius": {
                "affected_services": blast["affected_services"],
                "import_neighbors": n_import,
                "cochange_neighbors": n_cochange,
            },
            "missing_changes": {
                "coverage_score": coverage,
                "predictions": predictions,
            },
            "security_flagged": sec_flagged,
            "custom_rules_triggered": [{"name": r.get("name",""), "verdict": r.get("verdict","WARN"),
                                         "message": r.get("message","")} for r in custom_triggered],
            "suggested_reviewers": reviewers.get("reviewers", []),
            "verdict": verdict,
        }, indent=2)

    # Markdown report
    lines = []
    lines.append("## Guardian Report\n")

    # Verdict banner
    icon = {"PASS": "OK", "WARN": "WARNING", "FAIL": "BLOCKED"}[verdict["status"]]
    lines.append(f"**{icon}:** {verdict['reason']}\n")

    # Stats
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Changed files | {len(files)} |")
    lines.append(f"| Resolved modules | {len(seed_mods)} |")
    lines.append(f"| Affected services | {n_services} ({', '.join(blast['affected_services'])}) |")
    lines.append(f"| Import neighbors | {n_import} |")
    lines.append(f"| Co-change neighbors | {n_cochange} |")
    lines.append(f"| PR completeness | {coverage*100:.0f}% |")
    lines.append("")

    # Custom rules triggered
    if custom_triggered:
        lines.append("### Policy Rules Triggered\n")
        lines.append("| Rule | Verdict | Message |")
        lines.append("|------|---------|---------|")
        for cr in custom_triggered:
            lines.append(f"| {cr.get('name','')} | {cr.get('verdict','WARN')} | {cr.get('message','')} |")
        lines.append("")

    # Missing changes
    if predictions:
        lines.append("### Likely Missing Files\n")
        lines.append("| Module | Confidence | Weight | Reason |")
        lines.append("|--------|------------|--------|--------|")
        for p in predictions[:15]:
            conf = f"{p['confidence']*100:.0f}%"
            lines.append(f"| `{p['module']}` | {conf} | {p['weight']} | {p['reason']} |")
        if len(predictions) > 15:
            lines.append(f"\n_...and {len(predictions)-15} more predictions._")
        lines.append("")

    # Security flags
    if sec_flagged:
        lines.append("### Security-Sensitive Modules Touched\n")
        for m in sec_flagged:
            lines.append(f"- `{m}`")
        lines.append("")

    # Suggested reviewers
    rev_list = reviewers.get("reviewers", [])
    if rev_list:
        lines.append("### Suggested Reviewers\n")
        lines.append("_Based on module ownership from git history:_\n")
        lines.append("| Reviewer | Commits | Modules |")
        lines.append("|----------|---------|---------|")
        for r in rev_list[:5]:
            mod_str = ", ".join(f"`{m}`" for m in r["modules"][:3])
            if len(r["modules"]) > 3:
                mod_str += f" +{len(r['modules'])-3} more"
            lines.append(f"| {r['name']} | {r['commits']} | {mod_str} |")
        lines.append("")

    # Unresolved
    if unresolved:
        lines.append("### Unresolved Files\n")
        lines.append("_These files could not be mapped to known modules:_")
        for f in unresolved:
            lines.append(f"- `{f}`")
        lines.append("")

    return "\n".join(lines)


# ── Guardian Rules Engine ─────────────────────────────────────────────────────

_DEFAULT_RULES = {
    "version": 1,
    "thresholds": {
        "coverage_fail": 0.5,
        "coverage_warn": 0.8,
        "min_predictions_for_fail": 3,
        "max_services_warn": 3,
    },
    "security": {
        "mode": "extend",
        "keywords": [],
    },
    "rules": [],
}


def _load_guardian_rules(rules_path=None):
    """Load guardian rules from YAML. Returns merged config with defaults."""
    if rules_path is None or _yaml is None:
        return _DEFAULT_RULES.copy()

    path = pathlib.Path(rules_path)
    if not path.exists():
        print(f"Guardian rules file not found: {path}", file=sys.stderr)
        return _DEFAULT_RULES.copy()

    with open(path) as f:
        user_rules = _yaml.safe_load(f) or {}

    # Merge with defaults
    merged = _DEFAULT_RULES.copy()
    if "thresholds" in user_rules:
        merged["thresholds"] = {**merged["thresholds"], **user_rules["thresholds"]}
    if "security" in user_rules:
        merged["security"] = user_rules["security"]
    if "rules" in user_rules:
        merged["rules"] = user_rules["rules"]
    return merged


def _resolve_security_keywords(rules):
    """Build final security keywords set from rules config."""
    base = set(_SECURITY_KEYWORDS)
    sec_cfg = rules.get("security", {})
    extra = set(sec_cfg.get("keywords", []))
    if sec_cfg.get("mode") == "replace":
        return extra
    return base | extra


def _match_module_pattern(module, patterns):
    """Check if a module matches any glob pattern.

    Handles both file-path patterns (with /) and module patterns (with .).
    '**/Auth*' matches both 'src/Auth/Login.hs' and 'Euler.Auth.Session'.
    """
    for pat in patterns:
        # Direct match
        if fnmatch.fnmatch(module, pat) or fnmatch.fnmatch(module.lower(), pat.lower()):
            return True
        # Convert path-style pattern to dot-style for module matching
        dot_pat = pat.replace("/", ".").replace("**.", "*").replace("..", ".")
        if fnmatch.fnmatch(module, dot_pat) or fnmatch.fnmatch(module.lower(), dot_pat.lower()):
            return True
    return False


def _evaluate_custom_rules(rules, seed_modules, changed_files, blast):
    """Evaluate custom rules from config. Returns list of triggered rules."""
    triggered = []
    for rule in rules.get("rules", []):
        match_cfg = rule.get("match", {})
        matched = False

        # Match by module patterns
        if "modules" in match_cfg:
            patterns = match_cfg["modules"]
            for mod in seed_modules:
                if _match_module_pattern(mod, patterns):
                    matched = True
                    break

        # Match by min_services
        if "min_services" in match_cfg:
            if len(blast.get("affected_services", [])) >= match_cfg["min_services"]:
                matched = True

        if not matched:
            continue

        # Check requirements (if any)
        require = rule.get("require", {})
        requirements_met = True
        if "files_present" in require:
            patterns = require["files_present"]
            found = any(
                _match_module_pattern(f, patterns) for f in changed_files
            )
            if not found:
                requirements_met = False

        if matched and not requirements_met:
            triggered.append(rule)
        elif matched and not require:
            # Rule with no requirements triggers on match alone
            triggered.append(rule)

    return triggered


def _guardian_verdict(coverage, n_services, sec_flagged, predictions, rules=None):
    """Determine pass/warn/fail based on analysis and rules."""
    if rules is None:
        rules = _DEFAULT_RULES

    t = rules.get("thresholds", _DEFAULT_RULES["thresholds"])
    cov_fail = t.get("coverage_fail", 0.5)
    cov_warn = t.get("coverage_warn", 0.8)
    min_pred = t.get("min_predictions_for_fail", 3)
    max_svc  = t.get("max_services_warn", 3)

    if coverage < cov_fail and len(predictions) >= min_pred:
        return {"status": "FAIL",
                "reason": f"PR completeness {coverage*100:.0f}% with {len(predictions)} likely-missing files"}
    if sec_flagged:
        return {"status": "WARN",
                "reason": f"{len(sec_flagged)} security-sensitive module(s) touched"}
    if coverage < cov_warn:
        return {"status": "WARN",
                "reason": f"PR completeness {coverage*100:.0f}% -- review suggested"}
    if n_services > max_svc:
        return {"status": "WARN",
                "reason": f"Blast radius spans {n_services} services"}
    return {"status": "PASS",
            "reason": f"PR completeness {coverage*100:.0f}%, blast radius contained"}


def main():
    parser = argparse.ArgumentParser(
        description="PR blast-radius analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--files",    nargs="+", help="Explicit changed file paths")
    parser.add_argument("--explain",  action="store_true", help="Add LLM explanation (~30s)")
    _persona_choices = list(RE.PERSONA_LABELS) if _RE_AVAILABLE else ["reliability_engineer", "security_engineer", "performance_engineer"]
    parser.add_argument("--persona",  default="reliability_engineer",
                        choices=_persona_choices, help="Persona for --explain")
    parser.add_argument("--format",   choices=["text","json"], default="text")
    parser.add_argument("--check",    choices=["security"],
                        help="Exit non-zero when check fails")
    parser.add_argument("--mode",     choices=["blast", "guardian"], default="blast",
                        help="blast = blast radius only; guardian = full SDLC check")
    parser.add_argument("--max-hops", type=int, default=2, dest="max_hops")
    parser.add_argument("--artifact-dir", default=None, dest="artifact_dir",
                        help="Path to artifact dir (default: auto-detect)")
    parser.add_argument("--config",   default=None, help="Path to config.yaml")
    parser.add_argument("--rules",    default=None,
                        help="Path to guardian_rules.yaml (Guardian Mode only)")
    args = parser.parse_args()

    # Collect changed files
    if args.files:
        files = args.files
    elif not sys.stdin.isatty():
        files = [l.strip() for l in sys.stdin if l.strip()]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Analyzing {len(files)} changed file(s)...", file=sys.stderr)
    _require_re("blast-radius analysis")

    artifact_dir = pathlib.Path(args.artifact_dir) if args.artifact_dir else None
    RE.initialize(
        artifact_dir=artifact_dir,
        load_embedder=False,   # keyword-only for CI speed (no GPU needed)
        config_path=args.config,
    )

    # Resolve files → modules
    resolved   = RE.resolve_files_to_modules(files)
    seed_mods  = []
    unresolved = []
    for f, mods in resolved.items():
        if mods:
            seed_mods.extend(mods)
        elif "." in f or "::" in f:
            seed_mods.append(f)   # treat as module name directly
        else:
            unresolved.append(f)

    if not seed_mods:
        print("Could not resolve any files to known modules.", file=sys.stderr)
        if unresolved:
            print(f"Unresolved: {unresolved}", file=sys.stderr)
        sys.exit(0)

    blast      = RE.get_blast_radius(seed_mods, max_hops=args.max_hops)

    if args.mode == "guardian":
        # Load guardian rules (auto-detect or explicit path)
        rules_path = args.rules
        if rules_path is None:
            # Auto-detect: check artifact dir, then repo root
            for candidate in [
                artifact_dir / "guardian_rules.yaml" if artifact_dir else None,
                _REPO / "guardian_rules.yaml",
            ]:
                if candidate and candidate.exists():
                    rules_path = str(candidate)
                    print(f"Using rules: {rules_path}", file=sys.stderr)
                    break
        rules = _load_guardian_rules(rules_path)

        missing = RE.predict_missing_changes(seed_mods)
        reviewers = RE.suggest_reviewers(seed_mods) if hasattr(RE, 'suggest_reviewers') else {}
        report = _guardian_report(files, seed_mods, unresolved, blast, missing, args.format, rules,
                                  reviewers=reviewers)
        print(report)
        # Exit non-zero on FAIL verdict
        if args.format != "json":
            sec_keywords = _resolve_security_keywords(rules)
            all_mods = seed_mods + [n["module"] for n in blast["import_neighbors"]]
            sec_flagged = [m for m in all_mods if any(kw in m.lower() for kw in sec_keywords)]
            verdict = _guardian_verdict(
                missing.get("coverage_score", 1.0),
                len(blast["affected_services"]),
                sec_flagged,
                missing.get("predictions", []),
                rules,
            )
            # Custom rules can escalate
            for cr in _evaluate_custom_rules(rules, seed_mods, files, blast):
                if cr.get("verdict") == "FAIL":
                    verdict = {"status": "FAIL", "reason": cr.get("message", "Custom rule")}
                    break
            if verdict["status"] == "FAIL":
                sys.exit(1)
        sys.exit(0)

    sec_flagged = _print_summary(files, seed_mods, unresolved, blast, args.format)

    # Optional LLM explanation
    if args.explain:
        print(f"\n## Expert Analysis ({RE.PERSONA_LABELS.get(args.persona, args.persona)})\n",
              file=sys.stderr)
        blast_ctx = (
            f"Changed modules: {', '.join(seed_mods)}\n"
            f"Affected services: {', '.join(blast['affected_services'])}\n"
            f"Import neighbors: {len(blast['import_neighbors'])}\n"
            f"Co-change neighbors: {len(blast['cochange_neighbors'])}\n"
        )
        kw_by_svc = RE.cross_service_keyword_search(" ".join(seed_mods[:3]))
        cluster_by_svc = RE.get_cluster_context_for_services(list(kw_by_svc))
        base_ctx  = RE._build_base_context({}, kw_by_svc, cluster_by_svc, args.persona)
        initial_ctx = blast_ctx + "\n\n" + base_ctx

        def _progress(fn_name, args_, result):
            fn_id = args_.get("fn_id") or args_.get("query") or ""
            print(f"  [{fn_name}] {fn_id}", file=sys.stderr)

        answer = RE.get_expert_answer(
            query=f"Analyze the blast radius of changes to: {', '.join(seed_mods)}",
            tool_name=args.persona,
            initial_context=initial_ctx,
            on_tool_call=_progress,
        )
        print(answer)

    # Security gate
    if args.check == "security" and sec_flagged:
        print(
            f"\n[SECURITY GATE FAILED] {len(sec_flagged)} security-sensitive module(s) touched: "
            + ", ".join(sec_flagged),
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
