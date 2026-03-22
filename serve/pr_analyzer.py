#!/usr/bin/env python3
"""
pr_analyzer.py — PR blast-radius analysis for the Juspay codebase

Usage:
  # Pipe git diff output (most common in CI)
  git diff main...HEAD --name-only | python3 pr_analyzer.py

  # Explicit files
  python3 pr_analyzer.py --files euler-api-gateway/src/Euler/API/Gateway/Routes.hs

  # With LLM explanation (~30s, needs KIMI_API_KEY)
  git diff main...HEAD --name-only | python3 pr_analyzer.py --explain --persona reliability_engineer

  # JSON output for CI pipelines
  git diff main...HEAD --name-only | python3 pr_analyzer.py --format json

  # Security gate (exits non-zero if security-sensitive modules touched)
  git diff main...HEAD --name-only | python3 pr_analyzer.py --check security
"""
import argparse, json, os, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import retrieval_engine as RE

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


def main():
    parser = argparse.ArgumentParser(
        description="PR blast-radius analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--files",    nargs="+", help="Explicit changed file paths")
    parser.add_argument("--explain",  action="store_true", help="Add LLM explanation (~30s)")
    parser.add_argument("--persona",  default="reliability_engineer",
                        choices=list(RE.PERSONA_LABELS), help="Persona for --explain")
    parser.add_argument("--format",   choices=["text","json"], default="text")
    parser.add_argument("--check",    choices=["security"],
                        help="Exit non-zero when check fails")
    parser.add_argument("--max-hops", type=int, default=2, dest="max_hops")
    parser.add_argument("--artifact-dir", default=None, dest="artifact_dir",
                        help="Path to artifact dir (default: auto-detect)")
    parser.add_argument("--config",   default=None, help="Path to config.yaml")
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
