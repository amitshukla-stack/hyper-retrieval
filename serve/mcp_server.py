"""
mcp_server.py — HyperRetrieval MCP server

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
    "hyperretrieval",
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
      → Retry with service= filter (e.g. service="euler-api-txns") to skip test repos
      → Or use a more specific query (e.g. "mandate retry workflow" not "mandate retry")
      → Or call get_module() with the module prefix from the file paths you see

    Examples:
      search_symbols("mandate retry", service="euler-api-txns") → skip UCS test noise
      search_symbols("preauth auto refund") → finds trigger functions
      search_symbols("token validation", service="euler-api-gateway") → scoped scan

    Args:
        query:   Natural language or code-style search (e.g. "UPI collect flow handler")
        service: Optional — restrict to one service (e.g. "euler-api-gateway", "UCS")
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
    - Known gateway name (payu, razorpay, hdfc) → finds all its modules instantly
    - Known domain concept (mandate, refund, emi) → finds all owning modules
    - Use BEFORE get_module to discover what modules exist for a topic

    Examples:
      search_modules("PayU")              → Euler...PayU.Routes, ...Transforms, ...Types
      search_modules("mandate", service="euler-api-txns")
      search_modules("refund webhook")

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
      get_module("Euler.API.Gateway.Gateway.Razorpay.Flow")
      get_module("MandateWorkflow", service="euler-api-txns", max_symbols=15)
      get_module("Payment.Interface", service="euler-api-pre-txn")

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
    'Euler.API.Gateway.Gateway.Razorpay.Flow.Webhook.verifyAmount').

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
      get_blast_radius(["euler-api-gateway/src/Routes.hs"])
      get_blast_radius(["Euler.API.Gateway.Routes", "Euler.API.Txns.Mandate"])

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
            f"Try passing module names directly (e.g. 'Euler.API.Gateway.Routes')."
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
      predict_missing_changes(["euler-api-gateway/src/Routes.hs"])
      predict_missing_changes(["Euler.API.Gateway.Routes", "Euler.API.Txns.Flow"])

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
                "Try passing module names directly (e.g. 'Euler.API.Gateway.Routes').")

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
                     e.g. ["euler-api-gateway", "euler-api-txns"]
        max_symbols: Cap symbols per service (0 = no cap). Use 20-30 for large queries.
    """
    kw_by_svc  = RE.cross_service_keyword_search(query, max_per_service=25)
    vec_by_svc = RE.stratified_vector_search([query], k_total=300) if RE.can_embed() else {}

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
# ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _startup()
    if _args.http:
        print(f"[mcp_server] Starting HTTP/SSE server on http://127.0.0.1:{MCP_PORT}/sse", file=sys.stderr)
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
