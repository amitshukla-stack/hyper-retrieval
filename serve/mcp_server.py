"""
mcp_server.py — Ripple MCP server

Exposes codebase intelligence as 8 MCP tools.

Transports
----------
stdio (default — local IDEs on the same machine):
  python3 mcp_server.py
  IDE config: {"command": "wsl", "args": ["python3", "mcp_server.py"]}

http (remote access via Cloudflare tunnel):
  python3 mcp_server.py --http [--port 8002]
  IDE config: {"url": "https://<tunnel>.trycloudflare.com/sse"}

GPU strategy
------------
  Recommended: start embed_server.py first, set EMBED_SERVER_URL=http://localhost:8001
  Both Chainlit and this server share the same GPU load.
  Without EMBED_SERVER_URL: loads GPU model independently — do NOT run while
  Chainlit is active (OOM risk).

Tool usage decision tree
------------------------
  1. START with search_symbols (brief=True for orientation, brief=False for detail)
  2. To explore a whole namespace → get_module (one call instead of many get_function_body)
  3. To read one function's code → get_function_body
  4. To trace who calls a function → trace_callers
  5. To trace what a function calls → trace_callees
  6. For blast radius of a change → get_blast_radius
  7. For a deep cross-service question → get_context (LAST RESORT — expensive, ~5-18k tokens)
     - Never call get_context more than ONCE per turn
     - Use services= param to limit to relevant services (saves 70% tokens)
     - Prefer search_symbols + get_function_body for targeted questions
"""
import argparse, os, pathlib, socket, sys

# ── Add serve/ (retrieval_engine) and repo root (tools.py) to path ───────────
_SERVE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SERVE))              # for retrieval_engine.py
sys.path.insert(0, str(_SERVE.parent))       # for tools.py

import retrieval_engine as RE
import tools as T
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--http",   action="store_true", help="Run HTTP/SSE server instead of stdio")
_parser.add_argument("--port",   type=int, default=8002, help="Port for HTTP mode (default: 8002)")
_args, _ = _parser.parse_known_args()

MCP_PORT = _args.port
mcp = FastMCP(
    "ripple",
    host="127.0.0.1",
    port=MCP_PORT,
    sse_path="/sse",
    # Disable DNS rebinding protection so Cloudflare tunnel Host headers are accepted.
    # The server is still only reachable via the tunnel (bound to 127.0.0.1).
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# ════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════

def _check_port(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except Exception:
        return False


def _startup():
    using_embed_server = bool(os.environ.get("EMBED_SERVER_URL", ""))

    # Redirect stdout→stderr during initialization so retrieval_engine print()
    # calls don't corrupt the MCP JSON-RPC stdio channel.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        if not using_embed_server and _check_port(8000):
            print(
                "[mcp_server] WARNING: Chainlit detected on port 8000 and EMBED_SERVER_URL is not set.\n"
                "[mcp_server] Running with load_embedder=False to avoid GPU OOM — vector search disabled.\n"
                "[mcp_server] Start embed_server.py and set EMBED_SERVER_URL=http://localhost:8001\n"
                "[mcp_server] to enable vector search while Chainlit is running.",
            )
            RE.initialize(load_embedder=False)
        else:
            RE.initialize(load_embedder=not using_embed_server)
    finally:
        sys.stdout = _real_stdout


# ════════════════════════════════════════════════════════════════════════════
# MCP TOOLS
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def search_symbols(query: str, service: str = "", brief: bool = False) -> str:
    """
    Find functions, types, and modules by name or concept. ALWAYS start here.

    Decision guide:
    - Use brief=True first to get an overview (~50 tokens/result, name + file only)
    - Use brief=False when you need type signatures to understand interfaces
    - Use service= to scope to one microservice and cut noise by 80%
    - After finding an ID, call get_function_body() to read its implementation
    - Found a module path? Call get_module() to see all symbols in it at once

    If results look like test/harness code (file path contains "test", "spec", "harness",
    "scenario", "mock"), do NOT jump to get_context. Instead:
      → Retry with service= filter to skip test repos
      → Or use a more specific query (e.g. "payment retry workflow" not "payment retry")
      → Or call get_module() with the module prefix from the file paths you see

    Examples:
      search_symbols("payment retry", service="api-gateway") → scoped to one service
      search_symbols("auth token validation") → finds relevant functions
      search_symbols("database connection pool", brief=True) → quick overview

    Args:
        query:   Natural language or code-style search (e.g. "user authentication handler")
        service: Optional — restrict to one service (e.g. "api-gateway", "auth-service")
        brief:   True = name+file only (~50 tokens/result). False = full signature (default)
    """
    return T.tool_search_symbols(query, service, brief)


@mcp.tool()
def search_modules(query: str, service: str = "") -> str:
    """
    Search module namespaces by keyword. Returns module paths for get_module().

    Use this FIRST when you know a gateway/feature/domain name but not the
    exact function. One call surfaces the entire relevant namespace.

    Decision guide:
    - Known component name (auth, payments, notifications) → finds all its modules instantly
    - Known domain concept (billing, subscriptions, webhooks) → finds all owning modules
    - Use BEFORE get_module to discover what modules exist for a topic

    Examples:
      search_modules("auth")              → Auth.Routes, Auth.Middleware, Auth.Types
      search_modules("billing", service="payments-service")
      search_modules("webhook handler")

    Args:
        query:   Keywords to search in module paths
        service: Optional — restrict to one service
    """
    return T.tool_search_modules(query, service)


@mcp.tool()
def get_module(module_name: str, service: str = "", max_symbols: int = 30) -> str:
    """
    List all symbols in a module namespace in one call.

    Use this INSTEAD of multiple get_function_body calls when you want to see
    what a whole module exposes. Returns names, signatures, file locations, and
    which IDs have body source available.

    Decision guide:
    - Use when search_symbols returns a module path and you want the full picture
    - Use when you need to compare sibling functions in a module
    - Set max_symbols=10 for a quick overview of large modules
    - After reviewing the list, call get_function_body() on specific IDs

    Examples:
      get_module("Auth.Middleware.JWT")
      get_module("PaymentWorkflow", service="payments-service", max_symbols=15)
      get_module("User.Interface", service="api-gateway")

    Args:
        module_name: Module name or prefix (dots or :: separators both work)
        service:     Optional — restrict to one service to avoid ambiguity
        max_symbols: Max symbols to return per module (default: 30)
    """
    return T.tool_get_module(module_name, service, max_symbols)


@mcp.tool()
def get_function_body(fn_id: str, reason: str = "") -> str:
    """
    Read the source code of one function by its fully-qualified ID.

    The ID comes from search_symbols or get_module output (e.g.
    'Auth.Middleware.JWT.verifyToken').

    Also returns: log patterns emitted, direct callees listed.

    Decision guide:
    - Call AFTER search_symbols or get_module — not cold
    - If you need 3+ sibling functions, use get_module first (one call vs many)
    - If the ID doesn't resolve, try trace_callers / trace_callees to navigate

    Args:
        fn_id:  Fully-qualified function ID from search results
        reason: Why you're reading this (optional — helps trace investigation)
    """
    return T.tool_get_function_body(fn_id, reason)


@mcp.tool()
def trace_callers(fn_id: str, reason: str = "") -> str:
    """
    Find all functions that call this function (upstream / impact analysis).

    Use to answer: "Who depends on this?" "What entry points trigger this?"
    "What breaks if I change this signature?"

    Decision guide:
    - Use before modifying a function to understand blast radius
    - Combine with trace_callees to get the full call chain
    - For cross-service impact, follow up with get_blast_radius

    Args:
        fn_id:  Function ID to find callers for
        reason: Why you're tracing (optional)
    """
    return T.tool_trace_callers(fn_id, reason)


@mcp.tool()
def trace_callees(fn_id: str, reason: str = "") -> str:
    """
    Find all functions called BY this function (downstream dependencies).

    Use to answer: "What does this delegate to?" "What services does this touch?"
    "What is the full execution path below this entry point?"

    Decision guide:
    - Use to follow data flow from an entry point down to storage/external calls
    - Combine with trace_callers for bidirectional call graph exploration
    - If the callee list spans services, get_blast_radius gives a cleaner view

    Args:
        fn_id:  Function ID to get callees for
        reason: Why you're tracing (optional)
    """
    return T.tool_trace_callees(fn_id, reason)


@mcp.tool()
def get_blast_radius(files_or_modules: list[str], max_hops: int = 2) -> str:
    """
    Compute blast radius for a set of changed files or modules.

    Combines two signals:
    - Import graph: which modules transitively import the changed ones
    - Co-change history: which modules have historically changed together in git

    Decision guide:
    - Use BEFORE merging a PR or refactoring a module
    - Pass git diff --name-only output directly (file paths auto-resolved)
    - max_hops=1 for direct deps only, max_hops=2 (default) for transitive
    - For a single function's impact, trace_callers is faster

    Examples:
      get_blast_radius(["api-gateway/src/routes.py"])
      get_blast_radius(["Auth.Routes", "Payments.Checkout"])

    Args:
        files_or_modules: File paths (from git diff) or module names. Both work.
        max_hops: Import graph traversal depth (default: 2)
    """
    resolved   = RE.resolve_files_to_modules(files_or_modules)
    seed_mods  = []
    unresolved = []
    for f, mods in resolved.items():
        if mods:
            seed_mods.extend(mods)
        else:
            # If input looks like a module name already, use it directly
            if "." in f or "::" in f:
                seed_mods.append(f)
            else:
                unresolved.append(f)

    if not seed_mods:
        return (
            f"Could not resolve any inputs to known modules.\n"
            f"Unresolved: {unresolved}\n"
            f"Try passing module names directly (e.g. 'Auth.Routes')."
        )

    result = RE.get_blast_radius(seed_mods, max_hops=max_hops)

    lines = ["## Blast Radius Analysis"]
    lines.append(f"\n**Seed modules:** {', '.join(result['seed_modules'])}")
    lines.append(f"**Affected services:** {', '.join(result['affected_services']) or 'none identified'}")
    if unresolved:
        lines.append(f"**Unresolved inputs:** {', '.join(unresolved)}")

    if result["import_neighbors"]:
        lines.append(f"\n### Import Graph ({len(result['import_neighbors'])} neighbors)")
        for n in result["import_neighbors"][:25]:
            lines.append(f"  hop={n['hop']} [{n['service'] or '?'}] {n['module']} ({n['direction']})")

    if result["cochange_neighbors"]:
        lines.append(f"\n### Co-Change History ({len(result['cochange_neighbors'])} neighbors)")
        for n in result["cochange_neighbors"][:15]:
            lines.append(f"  hop={n['hop']} weight={n['weight']} {n['module']}")

    tiered = result.get("tiered_impact", [])
    if tiered:
        lines.append(f"\n### Tiered Impact Summary ({len(tiered)} total)")
        for tier_name, emoji in [("will_break", "!"), ("may_break", "?"), ("review", "~")]:
            tier_items = [t for t in tiered if t["tier"] == tier_name]
            if tier_items:
                lines.append(f"\n  [{emoji}] {tier_name.upper()} ({len(tier_items)} modules)")
                for t in tier_items[:10]:
                    sigs = t.get("signals", {})
                    parts = [f"conf={t['confidence']:.2f}"]
                    if "static_hop" in sigs:
                        parts.append(f"hop={sigs['static_hop']}")
                    if "cochange_weight" in sigs:
                        parts.append(f"cc={sigs['cochange_weight']}")
                    if "granger" in sigs:
                        g = sigs["granger"]
                        parts.append(f"granger(lag={g['lag']},p={g['p_value']:.4f})")
                    svc = f" [{t['service']}]" if t.get("service") else ""
                    lines.append(f"    {t['module']}{svc} ({', '.join(parts)})")
                if len(tier_items) > 10:
                    lines.append(f"    ... and {len(tier_items) - 10} more")

    return "\n".join(lines)


@mcp.tool()
def predict_missing_changes(changed_files: list[str], min_confidence: float = 0.1) -> str:
    """
    Predict modules likely MISSING from a changeset (PR review assistant).

    Given files changed in a PR, uses co-change history to predict what other
    modules typically change together but are NOT in the changeset. High
    confidence = "you almost certainly forgot this."

    Use cases:
    - PR review: "these files usually change together, did you forget one?"
    - Pre-commit check: "your change touches X, you might also need to update Y"
    - Refactoring: "changing this module historically requires changes to these others"

    Pass git diff --name-only output directly — file paths are auto-resolved.

    Examples:
      predict_missing_changes(["api-gateway/src/routes.py"])
      predict_missing_changes(["Auth.Routes", "Payments.Flow"])

    Args:
        changed_files: File paths (from git diff) or module names. Both work.
        min_confidence: Minimum confidence threshold (0-1). Default 0.1.
    """
    resolved = RE.resolve_files_to_modules(changed_files)
    changed_mods = []
    for f, mods in resolved.items():
        if mods:
            changed_mods.extend(mods)
        elif "." in f or "::" in f:
            changed_mods.append(f)

    if not changed_mods:
        return ("Could not resolve any inputs to known modules.\n"
                "Try passing module names directly (e.g. 'Auth.Routes').")

    result = RE.predict_missing_changes(changed_mods)

    lines = ["## Missing Change Predictions"]
    lines.append(f"\n**Changed modules ({len(result['changed'])}):** "
                 + ", ".join(result["changed"][:10])
                 + ("..." if len(result["changed"]) > 10 else ""))
    lines.append(f"**Coverage score:** {result['coverage_score']:.0%}"
                 " (higher = changeset looks complete)")

    preds = [p for p in result["predictions"] if p["confidence"] >= min_confidence]
    if preds:
        lines.append(f"\n### Predicted Missing Changes ({len(preds)} modules)")
        lines.append("")
        for p in preds:
            conf_bar = "█" * int(p["confidence"] * 10) + "░" * (10 - int(p["confidence"] * 10))
            lines.append(f"  {conf_bar} {p['confidence']:.0%}  "
                         f"[{p['service'] or '?'}] **{p['module']}**")
            lines.append(f"         {p['reason']}")
    else:
        lines.append("\n✓ No high-confidence missing changes detected. "
                     "Changeset looks complete.")

    return "\n".join(lines)


# ── Learned-rules helpers (T-030 Phase 3) ─────────────────────────────────────

def _load_active_rules() -> list[dict]:
    """Load ~/.hyperretrieval/active_rules.json; return [] if missing or invalid."""
    rules_path = Path.home() / ".hyperretrieval" / "active_rules.json"
    if not rules_path.exists():
        return []
    try:
        with open(rules_path) as f:
            data = json.load(f)
        return [r for r in data.get("rules", []) if r.get("status") == "active"]
    except Exception:
        return []


def _lookup_rules(tool: str, input_files: list[str]) -> list[dict]:
    """Return active rules for this tool whose query_files overlap with input_files."""
    input_set = {f.split("/")[-1].lower() for f in input_files}
    matched = []
    for rule in _load_active_rules():
        if rule.get("tool") != tool:
            continue
        rule_files = {f.split("/")[-1].lower() for f in rule.get("query_files", [])}
        if rule_files and rule_files & input_set:
            matched.append(rule)
    return matched


@mcp.tool()
def check_my_changes(changed_files: list[str]) -> str:
    """
    SDLC guardian: run ALL quality checks on your changes in one call.

    Combines blast_radius + predict_missing_changes + security scan into a
    single verdict. Designed for AI coding agents to self-check BEFORE
    committing or opening a PR.

    Returns: PASS / WARN / FAIL with reasons and action items.

    Use this AFTER writing code but BEFORE committing to catch:
    - Missing files that historically co-change with your modifications
    - Unexpectedly large blast radius across services
    - Security-sensitive modules that need extra review

    Examples:
      check_my_changes(["api-gateway/src/routes.py"])
      check_my_changes(["PaymentFlows", "TransactionHelper"])
    """
    # Resolve files to modules
    resolved = RE.resolve_files_to_modules(changed_files)
    changed_mods = []
    for f, mods in resolved.items():
        if mods:
            changed_mods.extend(mods)
        elif "." in f or "::" in f:
            changed_mods.append(f)

    if not changed_mods:
        return ("Could not resolve any inputs to known modules.\n"
                "Try passing module names directly (e.g. 'PaymentFlows').")

    # Run both analyses
    blast = RE.get_blast_radius(changed_mods, max_hops=2)
    missing = RE.predict_missing_changes(changed_mods)

    coverage = missing.get("coverage_score", 1.0)
    predictions = missing.get("predictions", [])
    n_services = len(blast["affected_services"])
    all_affected = changed_mods + [n["module"] for n in blast["import_neighbors"]]
    _sec_kw = {"auth", "token", "credential", "secret", "password",
               "encrypt", "hmac", "signature", "session", "oauth", "jwt"}
    sec_flagged = [m for m in all_affected
                   if any(kw in m.lower() for kw in _sec_kw)]

    # Guard: run local-file static guardrail checks if any changed_files are real paths
    try:
        from serve.guard_integration import run_guard_on_files, summarize_findings
        guard_findings = run_guard_on_files(changed_files)
        guard_summary = summarize_findings(guard_findings)
    except Exception as _e:
        print(f"[check_my_changes] guard skipped: {_e!r}")
        guard_findings, guard_summary = [], {"count": 0, "critical": 0, "warning": 0, "patterns": []}

    # Provenance: check if any changed lines are AI-generated (Git AI / Agent Blame / Tabnine)
    try:
        from serve.provenance_reader import summarize as _prov_summarize
        prov_summary = _prov_summarize(changed_files)
    except Exception as _e:
        print(f"[check_my_changes] provenance skipped: {_e!r}")
        prov_summary = {"total_ai_lines": 0, "files_with_ai": 0, "by_file": {}}

    # Guard findings on AI-touched files escalate harder — AI code without verification is riskier
    ai_guard_critical_bonus = 0
    if prov_summary["total_ai_lines"] > 0 and guard_findings:
        ai_files = set(prov_summary["by_file"].keys())
        ai_guard_critical_bonus = sum(
            1 for f in guard_findings
            if f.get("file") in ai_files or any(af.endswith(f.get("file", "")) for af in ai_files)
        )

    # Determine verdict (Guard CRITICAL escalates to FAIL; WARNING nudges toward WARN;
    # Guard finding on AI-touched file forces FAIL — unverified AI code is higher-risk)
    if ai_guard_critical_bonus > 0:
        status, reason = "FAIL", f"Guard finding on {ai_guard_critical_bonus} AI-generated file(s) — AI code + pattern violation is highest-risk"
    elif guard_summary.get("critical", 0) > 0:
        status, reason = "FAIL", f"Guard: {guard_summary['critical']} CRITICAL finding(s) — {', '.join(guard_summary['patterns'][:3])}"
    elif coverage < 0.5 and len(predictions) >= 3:
        status, reason = "FAIL", f"PR completeness {coverage:.0%} with {len(predictions)} likely-missing files"
    elif sec_flagged:
        status, reason = "WARN", f"{len(sec_flagged)} security-sensitive module(s) touched"
    elif coverage < 0.8:
        status, reason = "WARN", f"PR completeness {coverage:.0%} -- review suggested"
    elif n_services > 3:
        status, reason = "WARN", f"Blast radius spans {n_services} services"
    elif guard_summary.get("warning", 0) > 0:
        status, reason = "WARN", f"Guard: {guard_summary['warning']} warning(s) — {', '.join(guard_summary['patterns'][:3])}"
    else:
        status, reason = "PASS", f"PR completeness {coverage:.0%}, blast radius contained"

    # Build report
    lines = [f"## Guardian Check: {status}", f"\n**{reason}**\n"]
    lines.append(f"- **Changed modules:** {len(changed_mods)}")
    lines.append(f"- **Affected services:** {n_services} ({', '.join(blast['affected_services'][:5])}{'...' if n_services > 5 else ''})")
    lines.append(f"- **Import neighbors:** {len(blast['import_neighbors'])}")
    lines.append(f"- **Co-change neighbors:** {len(blast['cochange_neighbors'])}")
    lines.append(f"- **PR completeness:** {coverage:.0%}")

    tiered = blast.get("tiered_impact", [])
    if tiered:
        wb = [t for t in tiered if t["tier"] == "will_break"]
        mb = [t for t in tiered if t["tier"] == "may_break"]
        rv = [t for t in tiered if t["tier"] == "review"]
        lines.append(f"- **Impact tiers:** {len(wb)} will-break, {len(mb)} may-break, {len(rv)} review")

    if predictions[:5]:
        lines.append("\n### Likely Missing")
        for p in predictions[:5]:
            lines.append(f"  - **{p['module']}** ({p['confidence']:.0%}) -- {p['reason']}")

    if sec_flagged[:5]:
        lines.append("\n### Security Review Needed")
        for m in sec_flagged[:5]:
            lines.append(f"  - `{m}`")

    if guard_findings:
        lines.append(f"\n### Guard Findings ({guard_summary['count']} total, {guard_summary['critical']} CRITICAL)")
        for f in guard_findings[:8]:
            sev = f.get("severity", "WARNING")
            emoji = "🔴" if sev.upper() == "CRITICAL" else "🟡"
            lines.append(f"  {emoji} **{f.get('pattern', '?')}** `{f.get('file', '?')}:{f.get('line', 0)}` — {f.get('message', '')}")

    if prov_summary["total_ai_lines"] > 0:
        lines.append(f"\n### AI Provenance")
        lines.append(f"  - **{prov_summary['total_ai_lines']}** AI-generated line(s) across **{prov_summary['files_with_ai']}** file(s)")
        if ai_guard_critical_bonus > 0:
            lines.append(f"  - ⚠️ **{ai_guard_critical_bonus}** Guard finding(s) intersect AI-generated code — manual review required")

    # Suggested reviewers
    rev_data = RE.suggest_reviewers(changed_mods, top_k=3)
    if rev_data.get("reviewers"):
        lines.append("\n### Suggested Reviewers")
        for r in rev_data["reviewers"][:3]:
            mods = ", ".join(r["modules"][:3])
            lines.append(f"  - **{r['name']}** ({r['commits']} commits) -- {mods}")

    # Risk score (computed from already-fetched data to avoid redundant calls)
    risk = RE.score_change_risk(changed_mods)
    rs = risk["risk_score"]
    rl = risk["risk_level"]
    lines.insert(1, f"\n**Risk Score: {rs}/100 ({rl})**")
    comps = risk.get("components", {})
    comp_parts = []
    for name, c in comps.items():
        label = name.replace("_", " ").title()
        comp_parts.append(f"{label}: {c['score']}")
    if comp_parts:
        lines.insert(2, f"Components: {' | '.join(comp_parts)}")

    if status == "PASS":
        lines.append("\n*Your changes look complete. Safe to commit.*")
    elif status == "WARN":
        lines.append("\n*Review the warnings above before committing.*")
    else:
        lines.append("\n*Your changeset is likely incomplete. Add the missing files or confirm they are intentionally excluded.*")

    # Inject learned context from feedback loop (T-030)
    learned = _lookup_rules("check_my_changes", changed_files)
    if learned:
        lines.append("\n### Learned Context (from past reviews)")
        for r in learned[:3]:
            conf_pct = f"{r['confidence']:.0%}"
            lines.append(f"  - {r['label']}: {r['positive']}/{r['total']} reviewers found this helpful ({conf_pct} confidence)")
            for s in r.get("sample_summaries", [])[:1]:
                lines.append(f"    *\"{s}\"*")

    return "\n".join(lines)


@mcp.tool()
def suggest_reviewers(changed_files: list[str], top_k: int = 5) -> str:
    """
    Suggest PR reviewers based on module ownership from git history.

    Given changed files or module names, finds who has the most commits
    on the affected modules and their blast radius neighbors.

    Returns ranked reviewers with commit counts and module coverage.

    Examples:
      suggest_reviewers(["PaymentFlows", "TransactionHelper"])
      suggest_reviewers(["api-gateway/src/routes.py"])
    """
    # Resolve files to modules
    resolved = RE.resolve_files_to_modules(changed_files)
    changed_mods = []
    for f, mods in resolved.items():
        if mods:
            changed_mods.extend(mods)
        elif "." in f or "::" in f:
            changed_mods.append(f)

    if not changed_mods:
        return ("Could not resolve any inputs to known modules.\n"
                "Try passing module names directly (e.g. 'PaymentFlows').")

    result = RE.suggest_reviewers(changed_mods, top_k=top_k)

    if not result.get("reviewers"):
        return ("No ownership data available. Run build/08_build_ownership.py first\n"
                "to generate the ownership index from git history.")

    lines = ["## Suggested Reviewers\n"]
    lines.append("| Rank | Reviewer | Commits | Relevant Modules |")
    lines.append("|------|----------|---------|------------------|")
    for i, r in enumerate(result["reviewers"], 1):
        mods = ", ".join(f"`{m}`" for m in r["modules"][:3])
        if len(r["modules"]) > 3:
            mods += f" +{len(r['modules'])-3} more"
        lines.append(f"| {i} | {r['name']} | {r['commits']} | {mods} |")

    lines.append("\n### Per-Module Coverage\n")
    for mod, authors in result.get("coverage", {}).items():
        if authors:
            lines.append(f"- `{mod}`: {', '.join(authors[:3])}")

    return "\n".join(lines)


@mcp.tool()
def score_change_risk(changed_files: list[str], weights: dict | None = None) -> str:
    """
    Compute a composite risk score (0-100) for a set of changed files/modules.

    Combines four signals into one actionable number:
    - Blast radius (how many services affected)
    - Coverage gap (how many co-changes are missing)
    - Reviewer risk (bus factor / ownership concentration)
    - Service spread (how many services the changes span)

    Use this to gate PRs in CI/CD or prioritize review effort.

    Returns: risk score, risk level (LOW/MEDIUM/HIGH/CRITICAL),
    per-component breakdown, and recommendation.

    Examples:
      score_change_risk(["PaymentFlows", "TransactionHelper"])
      score_change_risk(["api-gateway/src/routes.py"])
    """
    # Resolve files to modules
    resolved = RE.resolve_files_to_modules(changed_files)
    changed_mods = []
    for f, mods in resolved.items():
        if mods:
            changed_mods.extend(mods)
        elif "." in f or "::" in f:
            changed_mods.append(f)

    if not changed_mods:
        return ("Could not resolve any inputs to known modules.\n"
                "Try passing module names directly (e.g. 'PaymentFlows').")

    rules = None
    if weights:
        rules = {"risk_weights": weights}

    result = RE.score_change_risk(changed_mods, rules=rules)

    lines = [f"## Change Risk Score: {result['risk_score']}/100 — {result['risk_level']}\n"]

    # Component breakdown
    lines.append("| Component | Score | Detail |")
    lines.append("|-----------|-------|--------|")
    for name, comp in result.get("components", {}).items():
        label = name.replace("_", " ").title()
        lines.append(f"| {label} | {comp['score']}/100 | {comp['detail']} |")

    lines.append(f"\n**Recommendation:** {result['recommendation']}")

    return "\n".join(lines)


@mcp.tool()
def check_criticality(modules: list[str]) -> str:
    """
    Check the criticality score of one or more modules.

    Returns a risk assessment based on 7 signals: blast radius, cross-repo
    coupling, change frequency, author concentration, recency, Granger
    causal influence, and revert history. Scores range from 0 (low risk)
    to 1 (highest risk).

    Use this BEFORE changing critical code to understand the risk.

    Examples:
      check_criticality(["Transaction", "TenantConfig"])
      check_criticality(["PaymentFlows"])
    """
    result = RE.check_criticality(modules)
    lines = ["## Criticality Assessment\n"]
    for mod, data in result.items():
        level = data.get("risk_level", "UNKNOWN")
        score = data.get("score", 0)
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")
        lines.append(f"### {icon} {mod} — {level} ({score:.3f})\n")
        if data.get("rank"):
            lines.append(f"Rank: #{data['rank']} of all modules\n")
        if data.get("reasons"):
            lines.append("**Key signals:**")
            for r in data["reasons"]:
                lines.append(f"- {r}")
        signals = data.get("signals", {})
        if signals:
            lines.append("\n| Signal | Score |")
            lines.append("|--------|-------|")
            for sig, val in sorted(signals.items(), key=lambda x: -x[1]):
                lines.append(f"| {sig.replace('_', ' ').title()} | {val:.2f} |")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def get_guardrails(modules: list[str]) -> str:
    """
    Get guardrail documents for critical modules.

    Guardrails are auto-generated protective rules that explain:
    - Why this code is critical
    - What invariant must stay true
    - What breaks if you change it
    - Who should review changes

    If a module has a guardrail, returns the full document.
    If not, returns the criticality score and a note.

    Use this when reviewing PRs that touch critical code.

    Examples:
      get_guardrails(["Transaction"])
      get_guardrails(["CardInfo", "CryptoUtils"])
    """
    result = RE.get_guardrails(modules)
    lines = ["## Guardrails\n"]
    for mod, data in result.items():
        if data.get("has_guardrail") and data.get("content"):
            lines.append(data["content"])
        else:
            lines.append(f"### {mod}\n")
            lines.append(f"No guardrail generated. {data.get('note', '')}\n")
        lines.append("---\n")
    return "\n".join(lines)


@mcp.tool()
def list_critical_modules(
    service: str = None,
    threshold: float = 0.5,
    top_k: int = 20,
) -> str:
    """
    List the most critical modules in the codebase, ranked by risk.

    Criticality is computed from blast radius, co-change coupling, author
    concentration, change frequency, Granger causality, and revert history.
    No domain knowledge needed — works on any codebase.

    Use this to understand where the landmines are before major refactors,
    during onboarding, or when planning code reviews.

    Args:
      service: Filter by service name (e.g., "euler-api-txns")
      threshold: Minimum criticality score (0-1, default 0.5)
      top_k: Number of results (default 20)

    Examples:
      list_critical_modules()
      list_critical_modules(service="euler-api-gateway", threshold=0.4)
    """
    result = RE.list_critical_modules(service=service, threshold=threshold, top_k=top_k)
    lines = [f"## Critical Modules ({result['total_above_threshold']} above threshold {threshold})\n"]
    if result.get("service_filter"):
        lines.append(f"Filtered by service: {result['service_filter']}\n")

    lines.append("| Rank | Module | Score | Risk | Guardrail |")
    lines.append("|------|--------|-------|------|-----------|")
    for m in result.get("modules", []):
        mod_short = m["module"].split("::")[-1][:40] if "::" in m["module"] else m["module"][:40]
        gr = "Yes" if m.get("has_guardrail") else "—"
        lines.append(f"| #{m.get('rank', '?')} | {mod_short} | {m['score']:.3f} | {m['risk_level']} | {gr} |")

    return "\n".join(lines)


@mcp.tool()
def fast_search(query: str, top_k: int = 10) -> str:
    """
    Zero-GPU keyword search: BM25 + IDF graph index, no embed server required.

    Use this when:
    - The embed server is not running (keyword-only or offline deployments)
    - You have an exact function/class/module name to look up
    - You need a sub-50ms result without semantic search overhead

    For natural-language or conceptual queries, use search_symbols instead —
    it adds vector search and co-change expansion for higher recall.

    Args:
        query:  Symbol name, module keyword, or exact identifier
        top_k:  Max results per service (default 10)
    """
    return T.tool_fast_search(query, top_k)


@mcp.tool()
def fast_search_reranked(query: str, top_k: int = 10) -> str:
    """
    BM25 search with cross-encoder reranking. Requires HR_RERANKER=1 at startup.
    Falls back to fast_search if the reranker is not loaded.

    Fetches BM25 top-30, scores each with ms-marco-MiniLM-L-6-v2 (CPU, ~20ms),
    and returns the reranked top-k. Significantly better than fast_search when
    the correct symbol is in the BM25 window but buried below irrelevant matches.

    Use this when:
    - fast_search returns results but top-3 feel wrong
    - Your query is conceptual ('webhook notification handler') not a bare identifier
    - You want best keyword-mode precision without GPU/embed server

    Args:
        query:  Natural-language or identifier query
        top_k:  Max total results (default 10)
    """
    return T.tool_fast_search_reranked(query, top_k)


@mcp.tool()
def get_why_context(symbol_name: str) -> str:
    """
    WHY context for a module or symbol: ownership, activity trend, Granger causal
    direction, criticality reasons, and anti-pattern flags.

    Use this BEFORE modifying critical code to understand who owns it, how often it
    changes, what it causally predicts in the codebase, and whether structural
    anti-patterns (god module, high churn, tight coupling) are present.

    Companion to get_blast_radius — that answers 'what breaks', this answers 'why
    is it this way and who should you talk to'.

    Args:
        symbol_name: Fully-qualified module or symbol name (e.g. 'PaymentGateway.Router'
                     or 'euler-api-gateway::PaymentRouter')
    """
    return T.tool_get_why_context(symbol_name)


@mcp.tool()
def get_context(
    query: str,
    persona: str = "default",
    services: list[str] | None = None,
    max_symbols: int = 0,
) -> str:
    """
    Full codebase context retrieval — expensive, use sparingly.

    Runs vector + keyword search, pulls cluster summaries, entry points, and
    doc chunks. Returns assembled context for cross-service architectural questions.
    No inference happens here — this is context gathering only.

    ⚠ LAST RESORT — 5,000–18,000 tokens per call. Hard rules:
    1. You MUST have called search_symbols at least once before calling this.
    2. You MUST have read at least one function body (get_function_body or get_module) first.
    3. If search_symbols returned ANY results (even test code), retry with service= or a
       different query before falling back here. "Results looked off" is not enough reason.
    4. Never call get_context twice in the same turn under any circumstances.
    5. With services= set, cost drops to ~2,000–4,000 tokens — always set it when possible.

    You can answer almost every "how does X work" question WITHOUT get_context by:
      search_symbols → get_module → get_function_body → trace_callees (follow the chain)

    Only call get_context when: you have no function IDs to explore AND you need
    cross-service cluster summaries that search_symbols cannot provide.

    Args:
        query:       Your question or topic (natural language)
        persona:     Leave as default — only "default" is supported
        services:    Limit to specific services — cuts tokens by 50-80%
                     e.g. ["api-gateway", "auth-service"]
        max_symbols: Cap symbols per service (0 = no cap). Use 20-30 for large queries.
    """
    # Use unified_search (RRF fusion of dense vector + BM25 + co-change expansion)
    # instead of separate keyword + vector searches
    vec_by_svc = RE.unified_search([query], k_total=300) if RE.can_embed() else {}
    kw_by_svc  = RE.cross_service_keyword_search(query, max_per_service=25)

    # Filter to requested services if specified
    if services:
        svc_set = set(services)
        kw_by_svc  = {k: v for k, v in kw_by_svc.items()  if k in svc_set}
        vec_by_svc = {k: v for k, v in vec_by_svc.items() if k in svc_set}

    # Apply per-service symbol cap if requested
    if max_symbols > 0:
        kw_by_svc  = {k: v[:max_symbols] for k, v in kw_by_svc.items()}
        vec_by_svc = {k: v[:max_symbols] for k, v in vec_by_svc.items()}

    all_svcs   = list(set(list(vec_by_svc) + list(kw_by_svc)))
    cluster_by_svc = RE.get_cluster_context_for_services(all_svcs)
    doc_hits   = []
    if RE.can_embed():
        qvec     = RE._encode_query(query)
        doc_hits = RE.doc_vector_search(qvec, top_k=20)
    return T._build_base_context(vec_by_svc, kw_by_svc, cluster_by_svc, persona, doc_hits)


# ════════════════════════════════════════════════════════════════════════════
# FEEDBACK SIGNAL CAPTURE (T-030)
# ════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def record_feedback(
    tool_name: str,
    query: str,
    signal: str,
    result_summary: str = "",
    context: str = "",
) -> str:
    """Record whether a HyperRetrieval result was helpful.

    Captures feedback signals for future rule-learning (Bugbot-style).
    Signals accumulate into learned rules that improve future results.

    Args:
        tool_name: Which tool produced the result (e.g. 'check_my_changes', 'get_blast_radius')
        query: The query or file list that was searched
        signal: 'helpful' or 'not_helpful'
        result_summary: Brief description of what the tool returned (optional)
        context: Additional context, e.g. PR URL or task description (optional)
    """
    import time
    if signal not in ("helpful", "not_helpful"):
        return f"Invalid signal '{signal}'. Use 'helpful' or 'not_helpful'."

    feedback_dir = Path.home() / ".hyperretrieval"
    feedback_dir.mkdir(exist_ok=True)
    signals_path = feedback_dir / "feedback_signals.jsonl"

    entry = {
        "ts": time.time(),
        "tool": tool_name,
        "query": query[:500],
        "signal": signal,
        "result_summary": result_summary[:300] if result_summary else "",
        "context": context[:300] if context else "",
    }
    with open(signals_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    total = sum(1 for _ in open(signals_path))
    return (
        f"Feedback recorded: {signal} for {tool_name}.\n"
        f"Total signals: {total} | File: {signals_path}"
    )


# ════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _startup()
    if _args.http:
        print(f"[mcp_server] Starting HTTP/SSE server on http://127.0.0.1:{MCP_PORT}/sse", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
