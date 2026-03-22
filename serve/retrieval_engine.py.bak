"""
retrieval_engine.py — Generic codebase intelligence retrieval engine

Extracted from demo_server_v6.py. Works for ANY codebase that has been
processed through the pipeline (graph_with_summaries.json + vectors.lance).

Config-driven: point it at your artifact_dir and a config.yaml and it works.
No Chainlit dependency. Safe to import from MCP servers, CLI tools, or CI scripts.

GPU sharing: set EMBED_SERVER_URL=http://localhost:8001 and the embedder
runs as a separate service (embed_server.py). Both Chainlit and MCP server
then share one GPU load via HTTP — no OOM.
"""
import json, os, pathlib, re, time, threading, urllib.request, urllib.error
from collections import defaultdict

# ── LLM config (from env or config.yaml) ─────────────────────────────────────
LLM_API_KEY  = os.environ.get("LLM_API_KEY",  "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL    = os.environ.get("LLM_MODEL",    "reasoning-large-model")
MAX_TOOL_CALLS = 12

# ── Embed server URL (set to delegate GPU model to embed_server.py) ───────────
# When set, _encode_query calls this instead of loading the model in-process.
# Allows Chainlit + MCP server to share one GPU load.
EMBED_SERVER_URL = os.environ.get("EMBED_SERVER_URL", "")  # e.g. http://localhost:8001

EMBED_INSTRUCTION = (
    "Instruct: Represent this code module for finding semantically similar "
    "components across microservices. Query: "
)

# ── Default artifact dir ───────────────────────────────────────────────────────
_DEFAULT_ARTIFACT_DIR = pathlib.Path(
    os.environ.get("ARTIFACT_DIR",
                   str(pathlib.Path(__file__).parent / "demo_artifact"))
)

# ── Global state ───────────────────────────────────────────────────────────────
embedder:      object = None   # SentenceTransformer (loaded in-process, or None if using embed server)
_llm_client:   object = None   # openai.OpenAI (sync)
G:             object = None   # NetworkX node-level graph
MG:            object = None   # NetworkX module-level import graph
lance_tbl:     object = None   # LanceDB code vector table
doc_lance_tbl: object = None   # LanceDB doc vector table
reranker:      object = None   # CrossEncoder — disabled (kept for reference)

cluster_summaries:   dict  = {}
cochange_index:      dict  = {}
file_to_nodes:       dict  = {}   # module_name → [node_id, ...]
filepath_to_module:  dict  = {}   # relative_file_path → module_name
_cochange_loaded_at: float = 0.0

body_store:   dict = {}
call_graph:   dict = {}
log_patterns: dict = {}
doc_chunks:   list = []
doc_by_id:    dict = {}
gw_integrity: dict = {}

# ── Retrieval tuning defaults (override via config.yaml) ────────────────────
KNOWN_SERVICES: list = [
    "euler-api-gateway", "euler-api-txns", "UCS", "euler-db",
    "euler-api-order", "graphh", "euler-api-pre-txn",
    "euler-api-customer", "basilisk-v3", "euler-drainer",
    "token_issuer_portal_backend", "haskell-sequelize",
]
_KW_STOPWORDS: set = {
    "what", "does", "that", "this", "with", "from", "have", "been", "when",
    "where", "which", "will", "would", "could", "should", "work", "works",
    "tell", "show", "give", "list", "find", "about", "around", "across",
    "payment", "payments", "service", "services", "flow", "flows",
    "code", "function", "module", "support", "supports", "using", "used", "uses",
    "handle", "handles", "process", "processes", "call", "calls",
    "implement", "implemented", "implementation",
}
_KW_ALLOWLIST: set = {"upi", "pix", "emi", "ucs", "cvv", "pan", "otp", "kyc", "bnpl", "nfc", "qr"}

# Known payment gateway names — used to generate better query variants.
# We deliberately don't hardcode gateway→service here since routing varies per deployment.
_KNOWN_GATEWAYS: frozenset = frozenset({
    "payu", "razorpay", "stripe", "adyen", "paypal", "braintree", "checkout",
    "worldpay", "cybersource", "nuvei", "bluesnap", "rapyd", "iatapay",
    "itaubank", "bambora", "tsys", "shift4", "globepay", "helcim", "gocardless",
    "mollie", "multisafepay", "nexinets", "noon", "nmi", "paybox", "payme",
    "payone", "square", "stax", "trustpay", "airwallex", "authorizedotnet",
    "hdfc", "icici", "axis", "kotak", "yesbank",
})

# Path segments that indicate test/harness code — deprioritised in search results
_TEST_PATH_SEGMENTS: frozenset = frozenset({
    "test", "tests", "spec", "specs", "harness", "mock", "mocks",
    "scenario", "scenarios", "fixture", "fixtures", "ucs-connector-tests",
    "examples", "example",
    # UCS Hyperswitch connector-integration is Rust scaffolding not used in production
    # payment-related test files — deprioritise so core business logic surfaces first
    "connector-integration",
})

_L2_THRESHOLD = 1.18
_RERANK_BATCH = 64


# ════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING  (makes the engine generic / deployment-agnostic)
# ════════════════════════════════════════════════════════════════════════════

def load_config(config_path: pathlib.Path | str) -> dict:
    """
    Load a config.yaml that overrides defaults.  Example config.yaml:

        llm:
          api_key: sk-...
          base_url: https://api.openai.com/v1
          model: gpt-4o

        embed:
          server_url: http://localhost:8001        # use embed_server.py
          instruction: "Represent this code: "    # custom instruction prefix

        services:
          - my-api-gateway
          - my-worker-service

        kw_allowlist: [api, rpc, grpc, sdk]

        personas:
          domain_expert:
            label: "🏛️ Domain Expert"
            system_prompt: "You are an expert in this codebase..."
            framework: "Trace the data flow end to end..."
    """
    global LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, EMBED_SERVER_URL, EMBED_INSTRUCTION
    global KNOWN_SERVICES, _KW_ALLOWLIST

    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        # Fall back to json if PyYAML not installed
        with open(config_path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return {}

    llm = cfg.get("llm", {})
    if llm.get("api_key"):
        LLM_API_KEY  = llm["api_key"]
    if llm.get("base_url"):
        LLM_BASE_URL = llm["base_url"]
    if llm.get("model"):
        LLM_MODEL    = llm["model"]

    embed = cfg.get("embed", {})
    if embed.get("server_url"):
        EMBED_SERVER_URL  = embed["server_url"]
    if embed.get("instruction"):
        EMBED_INSTRUCTION = embed["instruction"]

    if cfg.get("services"):
        KNOWN_SERVICES = cfg["services"]
    if cfg.get("kw_allowlist"):
        _KW_ALLOWLIST = set(cfg["kw_allowlist"])

    # Persona overrides are loaded separately by _load_persona_config(cfg)
    if cfg.get("personas"):
        _load_persona_config(cfg["personas"])

    return cfg


def _load_persona_config(personas_cfg: dict) -> None:
    """Override PERSONA_SYSTEM_PROMPTS / PERSONA_FRAMEWORKS from config."""
    for key, val in personas_cfg.items():
        if isinstance(val, dict):
            if val.get("system_prompt"):
                PERSONA_SYSTEM_PROMPTS[key] = val["system_prompt"]
            if val.get("framework"):
                PERSONA_FRAMEWORKS[key] = val["framework"]
            if val.get("label"):
                PERSONA_LABELS[key] = val["label"]


# ════════════════════════════════════════════════════════════════════════════
# AGENT TOOL SCHEMAS
# ════════════════════════════════════════════════════════════════════════════

AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "get_function_body",
        "description": (
            "Fetch the actual source code of a function by its fully-qualified ID.\n\n"
            "Use this when:\n"
            "- You have a function ID from the context and need to read its implementation\n"
            "- Type signatures alone are not enough to understand the logic\n"
            "- You are tracing a flow and need to see what a function actually does\n\n"
            "Prefer reading bodies over guessing from signatures — the gap between spec and "
            "implementation is where bugs live.\n\n"
            "Batch multiple get_function_body calls in a single response when you need to read "
            "several functions and they are independent of each other."
        ),
        "parameters": {"type": "object", "properties": {
            "fn_id": {
                "type": "string",
                "description": (
                    "Fully-qualified function ID in dot notation matching the graph — e.g. "
                    "`Product.OLTP.Transaction.initialiseSplitPaymentTxnFlow`. "
                    "Copy the ID exactly from the search results. "
                    "NEVER use file paths (no slashes, no .hs/.py/.rs extensions)."
                )
            },
            "reason": {"type": "string", "description": "One sentence: what you expect to learn from this body."}
        }, "required": ["fn_id"]}
    }},
    {"type": "function", "function": {
        "name": "trace_callees",
        "description": (
            "List all functions called BY a given function — its direct downstream dependencies.\n\n"
            "Use this when:\n"
            "- You are tracing a payment flow forward (what does this function delegate to?)\n"
            "- You want to understand what a function orchestrates without reading every callee body\n"
            "- You need to find which service boundary is crossed downstream\n\n"
            "Call this AFTER get_function_body — you need to read the function first to decide "
            "which callees are worth following."
        ),
        "parameters": {"type": "object", "properties": {
            "fn_id":  {"type": "string", "description": "Fully-qualified function ID (dot notation, no slashes)."},
            "reason": {"type": "string", "description": "What you are looking for in the call chain."}
        }, "required": ["fn_id"]}
    }},
    {"type": "function", "function": {
        "name": "trace_callers",
        "description": (
            "List all functions that CALL a given function — its upstream callers.\n\n"
            "Use this when:\n"
            "- Assessing blast radius: who is affected if this function changes?\n"
            "- Finding entry points: who initiates this operation?\n"
            "- Debugging: where is this function being called from in an unexpected way?\n\n"
            "The result shows callers across all services — pay attention to cross-service calls "
            "as they represent contract boundaries."
        ),
        "parameters": {"type": "object", "properties": {
            "fn_id":  {"type": "string", "description": "Fully-qualified function ID (dot notation, no slashes)."},
            "reason": {"type": "string", "description": "What you are assessing (blast radius, entry point, etc.)."}
        }, "required": ["fn_id"]}
    }},
    {"type": "function", "function": {
        "name": "get_log_patterns",
        "description": (
            "Return the log strings emitted by a function — what is observable in production logs.\n\n"
            "Use this when:\n"
            "- Debugging an incident: what log lines should appear if this code path ran?\n"
            "- Verifying observability: is this payment transition traceable?\n"
            "- Finding the log pattern to grep for in production\n\n"
            "Call this alongside get_function_body — they answer complementary questions "
            "(what does it do vs. what does it emit)."
        ),
        "parameters": {"type": "object", "properties": {
            "fn_id": {"type": "string", "description": "Fully-qualified function ID (dot notation, no slashes)."}
        }, "required": ["fn_id"]}
    }},
    {"type": "function", "function": {
        "name": "search_symbols",
        "description": (
            "Search for code symbols (functions, types, modules) by concept using vector + keyword search.\n\n"
            "Use this when:\n"
            "- The context does not contain the function you need\n"
            "- You know what a function does but not its exact name\n"
            "- You need to find all implementations of a concept across services\n\n"
            "Do NOT call this more than once for the same concept — if the first result is wrong, "
            "change your query term rather than repeating. The returned IDs can be passed directly "
            "to get_function_body."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {
                "type": "string",
                "description": "Describe the concept or function you are looking for in plain English or as a Haskell/Rust identifier."
            },
            "service": {
                "type": "string",
                "description": "Restrict search to a specific service name (e.g. 'euler-api-txns'). Omit to search all services."
            }
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": (
            "Search internal documentation and public API references.\n\n"
            "Use this when:\n"
            "- The question is about a payment protocol (UPI, EMI, mandate) rather than a specific function\n"
            "- The user asks about external behaviour (API contract, gateway spec, webhook format)\n"
            "- Code search returned results but you need the design intent behind them\n\n"
            "Use AFTER code search, not instead of it — docs describe intent, code is ground truth."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "tags":  {"type": "array", "items": {"type": "string"}, "description": "Optional topic tags to narrow results."}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "get_gateway_integrity",
        "description": (
            "Fetch the signature-verification and integrity configuration for a specific payment gateway.\n\n"
            "Use this when:\n"
            "- Investigating webhook authenticity or tamper detection for a gateway\n"
            "- Security review of how a gateway's response is validated\n"
            "- The question mentions a specific gateway name and involves auth, signing, or HMAC"
        ),
        "parameters": {"type": "object", "properties": {
            "gateway_name": {"type": "string", "description": "Gateway name as it appears in the codebase (e.g. 'razorpay', 'payu', 'stripe')."}
        }, "required": ["gateway_name"]}
    }},
    {"type": "function", "function": {
        "name": "get_type_definition",
        "description": (
            "Look up a type, struct, or enum definition by name.\n\n"
            "Use this when:\n"
            "- A function body references a type whose fields matter for the answer\n"
            "- The question is about what data a type carries or what states it models\n"
            "- You see a type name in the context and need to understand its shape\n\n"
            "Prefer this over search_symbols for type lookups — it is faster and more precise."
        ),
        "parameters": {"type": "object", "properties": {
            "type_name": {"type": "string", "description": "Type name exactly as it appears in code (e.g. 'TxnDetail', 'PaymentStatus', 'OrderReference')."},
            "service":   {"type": "string", "description": "Optional: restrict to a service if the type name is ambiguous."}
        }, "required": ["type_name"]}
    }},
    {"type": "function", "function": {
        "name": "search_modules",
        "description": (
            "Search module namespaces by keyword. Returns module paths you can pass to get_module.\n\n"
            "Use this FIRST when you don't know where code lives — it orients you within the codebase "
            "before you start reading individual functions.\n\n"
            "Examples: 'UPI collect', 'gateway routes', 'mandate payment', 'card tokenization'"
        ),
        "parameters": {"type": "object", "properties": {
            "query":   {"type": "string", "description": "Keyword(s) describing the module/component you are looking for."},
            "service": {"type": "string", "description": "Optional: restrict to a specific service name."}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "get_module",
        "description": (
            "List every symbol in a module namespace. Use after search_modules to see the full surface "
            "area before deciding what to read.\n\n"
            "Pass the exact module path returned by search_modules (dot notation).\n"
            "Example: get_module('Euler.API.Gateway.Gateway.UPI')"
        ),
        "parameters": {"type": "object", "properties": {
            "module_name": {"type": "string", "description": "Module namespace path in dot notation (e.g. 'PaymentFlows', 'Euler.API.Gateway.Gateway.UPI')."},
            "service":     {"type": "string", "description": "Optional: restrict to a specific service."}
        }, "required": ["module_name"]}
    }},
    {"type": "function", "function": {
        "name": "get_blast_radius",
        "description": (
            "Import graph + co-change analysis for a set of changed files or modules.\n\n"
            "Use this for impact assessment: given a file or module that changed, what else is affected?\n"
            "Returns: direct import neighbors (who imports this?), co-change neighbors (what changed "
            "together historically?), and affected services."
        ),
        "parameters": {"type": "object", "properties": {
            "files_or_modules": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of changed file paths or module names (dot notation)."
            }
        }, "required": ["files_or_modules"]}
    }},
    {"type": "function", "function": {
        "name": "get_context",
        "description": (
            "Last resort — returns a large pre-built context block (~5k-18k tokens) covering the query.\n\n"
            "Only use if search_symbols + get_function_body have failed to find what you need. "
            "Never call twice in one session — it is expensive."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The original user question."}
        }, "required": ["query"]}
    }},
]


# ════════════════════════════════════════════════════════════════════════════
# PERSONA PROMPT & FRAMEWORK — default identity (override via config.yaml personas section)
# ════════════════════════════════════════════════════════════════════════════

_DEFAULT_SYSTEM_PROMPT = """
You are Codebase Expert — an interactive AI assistant embedded in your organisation's codebase. You help engineers understand, trace, debug, and reason about code across services using a set of retrieval tools.

## Service architecture (authoritative — never invent or deviate)

- `euler-api-txns` — core payment orchestrator; drives the full transaction lifecycle
- `euler-api-order` — order creation and routing; calls `euler-api-txns`, NOT gateways directly
- `euler-api-gateway` (Haskell) / `UCS` (Rust) — connector-aggregation layers; translate internal payment requests to gateway-specific API calls; called BY `euler-api-txns`
- `euler-api-pre-txn` — pre-transaction operations: eligibility, offers, EMI, surcharges
- `euler-db` — canonical shared type definitions (library, not a runtime service)
- Canonical call chain: `euler-api-order` → `euler-api-txns` → [`euler-api-gateway` | `UCS`] → external gateway

## Tone and style

Answer immediately. Never restate the question. Never open with an affirmation. No trailing summaries.
Keep answers as short as the evidence allows — a single focused question gets a single focused answer.
Be direct: drop hedges ("might", "possibly", "I think") when you have evidence in front of you.
No emojis. No preamble. No postamble.

## Code references

Every factual claim must be grounded in evidence from your context. Cite as `Service.Module.functionName`.
If evidence is absent, say so explicitly — never invent a module path or function name.
Prefer the body over the type signature — the gap between spec and implementation is where bugs live.

## Markdown (the UI renders it — always apply)

- Function names, types, module identifiers → `backticks`
- File paths → `backticks`
- Multi-line code or type signatures → fenced code blocks with language tag (` ```haskell `, ` ```rust `, ` ```json `)
- `##` / `###` only for multi-section answers; for direct answers, no headings
- Never use `#` (h1)

## Tool usage policy

You have tools to fetch source code, trace call chains, and search symbols. Use them efficiently:

- **Batch independent tool calls** — call multiple tools in the same response when they do not depend on each other; sequential single-call round-trips are slower and waste your budget
- **Optimal investigation chain:** `search_symbols` → `get_function_body` → `trace_callees` — find the entry point first, read its body, then trace downstream
- **For impact / blast radius:** find the function, then `trace_callers` upward
- **For observability:** `get_log_patterns` on the function you just read — tells you what is observable without reading every callsite
- **For gateway-specific behaviour:** `get_gateway_integrity` before assuming a gateway follows the default flow
- **Stop when you have enough** — do not call more tools than necessary to answer the question; synthesis beats exhaustiveness
- **Do not search for the same concept twice** — if `search_symbols` returns nothing useful, refine the query or use a different term, do not re-run the same query

## Investigation strategy by question type

**Understanding a flow:** Find the entry-point function first (`search_symbols`). Read its body. Trace callees one level at a time. Stop when the answer is clear.
**Debugging / incident:** Start from the symptom. Identify 2–3 candidate functions. Read their bodies. Check log patterns. State which hypothesis the evidence supports or denies.
**Impact of a change:** Read the function being changed. `trace_callers` to find direct callers. Note which callers are payment-blocking vs. degraded-only.
**Ambiguous query:** If the user's query could refer to two different code paths (e.g. split payment vs. split settlement), ask one focused clarifying question before calling any tools.

## Engineering principles

- Strong type safety: business invariants enforced at the type level, not at runtime
- Explicit error handling: no silent failures, no partial successes treated as success
- Idempotency at every boundary: a payment must be safe to retry at any stage
- Clear service contracts: callers should not need to understand a service's internals
- Observability by default: every payment transition must be traceable in logs
""".strip()

_MERMAID_INSTRUCTION = """

DIAGRAM: If your answer involves a sequential flow, a decision tree, or cross-service interactions, output a Mermaid diagram AFTER your written answer. Format:

```mermaid
flowchart LR
    A[euler-api-txns<br/>functionName] --> B{Decision Point}
    B -->|UPI Collect| C[euler-api-gateway<br/>handleUpiCollect]
    B -->|UPI Intent| D[UCS<br/>determine_upi_flow]
```

Rules: `flowchart LR` for flows · `flowchart TD` for hierarchies · node labels as ServiceName<br/>functionName · edge labels as condition/data type · max 15 nodes · only nodes you can name from evidence.""".strip()

_DEFAULT_FRAMEWORK = (
    "Use markdown throughout — the UI renders it. "
    "All function names, types, and module paths in backticks. "
    "All file paths in backticks. "
    "Multi-line code or type signatures in fenced code blocks with language tag.\n\n"
    "Answer immediately — no preamble, no restating the question. "
    "Adapt your response structure to what is being asked:\n\n"
    "**Understanding / tracing a flow:** "
    "Trace step by step using actual function bodies (not just type signatures). "
    "Name every service hop and branching point with `service/module/function`. "
    "Add a Mermaid diagram after the written answer if the flow is cross-service.\n\n"
    "**Implementing / modifying code:** "
    "Use `####` subheadings — `#### Locate` (exact file + function), "
    "`#### Change` (what to add/modify with fenced code block), "
    "`#### Ripple` (callers, tests, config, DB schema that must also change), "
    "`#### Risk` (one-line flags: data loss / regression / contract break). "
    "Never invent new abstractions when an existing pattern covers the case.\n\n"
    "**Blast radius / impact:** "
    "Name direct callers → transitive dependents → co-change partners (from git history). "
    "For each: silent failure or noisy? Payment-blocking or degraded? "
    "End with go/no-go and minimum safe rollout order.\n\n"
    "**Debugging / incident:** "
    "Start from the symptom. Give 3-5 ranked hypotheses — for each, name the exact log pattern "
    "or code path that confirms or denies it. State fastest safe mitigation.\n\n"
    "**Security review:** "
    "Trace where untrusted input enters and where PAN/tokens/credentials cross boundaries. "
    "Classify each finding: **Critical / High / Medium / Low**. "
    "End with the exact `file/function` to fix first.\n\n"
    "**Code review:** "
    "Give verdicts, not descriptions — Approve / Request Changes / Reject. "
    "Name the specific principle and `module/function` as evidence for each verdict.\n\n"
    "Cite module paths and function names throughout. Quote actual logic from function bodies when available."
    + "\n\n" + _MERMAID_INSTRUCTION
)

# Single-entry dicts — all keys point to the one Juspay-code identity
PERSONA_SYSTEM_PROMPTS: dict = {
    "default": _DEFAULT_SYSTEM_PROMPT,
}
PERSONA_FRAMEWORKS: dict = {
    "juspay_code": _DEFAULT_FRAMEWORK,
}
PERSONA_LABELS: dict = {
    "default": "Codebase Expert",
}

# Keep _BASE_IDENTITY as an alias for backward compatibility with any external callers
_BASE_IDENTITY = _DEFAULT_SYSTEM_PROMPT


# ════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

def initialize(
    artifact_dir: pathlib.Path | None = None,
    load_embedder: bool = True,
    config_path: pathlib.Path | str | None = None,
) -> None:
    """
    Load all data stores. Idempotent — skips if already loaded at requested level.

    artifact_dir:  path to demo_artifact/ (contains graph_with_summaries.json, vectors.lance).
                   Defaults to the directory next to this file.
    load_embedder: False skips the ~6GB Qwen3 model → keyword-only search.
                   Ignored when EMBED_SERVER_URL is set (model runs as separate service).
    config_path:   optional path to config.yaml.
    """
    global embedder, _llm_client, G, MG, lance_tbl, cluster_summaries
    global body_store, call_graph, log_patterns, doc_chunks, doc_by_id, gw_integrity
    global doc_lance_tbl, _cochange_loaded_at

    if config_path:
        load_config(config_path)

    # When using embed server, we never load the model in-process
    using_embed_server = bool(EMBED_SERVER_URL)
    if using_embed_server:
        load_embedder = False

    if G is not None and (not load_embedder or embedder is not None or using_embed_server):
        return  # Already at desired level

    if artifact_dir is None:
        artifact_dir = _DEFAULT_ARTIFACT_DIR
    artifact_dir = pathlib.Path(artifact_dir)

    GRAPH_PATH     = str(artifact_dir / "graph_with_summaries.json")
    LANCE_PATH     = str(artifact_dir / "vectors.lance")
    EMBED_MODEL    = os.environ.get("EMBED_MODEL",
                                str(artifact_dir.parent / "models" / "qwen3-embed-8b"))
    BODY_STORE_P   = artifact_dir.parent / "output" / "body_store.json"
    CALL_GRAPH_P   = artifact_dir.parent / "output" / "call_graph.json"
    LOG_PATTERNS_P = artifact_dir.parent / "output" / "log_patterns.json"
    DOC_CHUNKS_P   = artifact_dir.parent / "output" / "doc_chunks.json"
    GW_INTEGRITY_P = artifact_dir.parent / "output" / "gateway_integrity_config.json"
    cochange_path  = artifact_dir / "cochange_index.json"

    import networkx as nx, lancedb

    if G is None:
        print("Loading graph...")
        graph_data = json.load(open(GRAPH_PATH))
        G = nx.node_link_graph(graph_data["networkx"])
        cluster_summaries.update(graph_data.get("cluster_summaries", {}))

        cluster_attrs = {
            n["id"]: {
                "cluster_name":    n.get("cluster_name", ""),
                "cluster_purpose": n.get("cluster_purpose", ""),
                "type":            n.get("type", ""),
                "ghost_deps":      n.get("ghost_deps", []),
                "file":            n.get("file", ""),
            }
            for n in graph_data["nodes"]
            if n.get("cluster_name") or n.get("type") or n.get("file")
        }
        nx.set_node_attributes(G, cluster_attrs)
        print(f"  {G.number_of_nodes():,} nodes  {G.number_of_edges():,} edges  {len(cluster_summaries)} summaries")

        print("Building module-level traversal graph...")
        _mg = nx.DiGraph()
        raw_edges  = graph_data.get("edges", [])
        mod_to_svc = {n.get("module", ""): n.get("service", "") for n in graph_data["nodes"]}
        for e in raw_edges:
            src, dst, kind = e.get("from",""), e.get("to",""), e.get("kind","")
            if not src or not dst or kind != "import":
                continue
            if src in mod_to_svc and dst in mod_to_svc:
                _mg.add_node(src, service=mod_to_svc.get(src,""))
                _mg.add_node(dst, service=mod_to_svc.get(dst,""))
                if _mg.has_edge(src, dst):
                    _mg[src][dst]["weight"] += 1
                else:
                    _mg.add_edge(src, dst, kind="import", weight=1)
        # Assign to module-level MG
        globals()["MG"] = _mg

        cs_edges = sum(1 for u,v in _mg.edges()
                       if _mg.nodes[u].get("service") != _mg.nodes[v].get("service"))
        print(f"  {_mg.number_of_nodes():,} modules  {_mg.number_of_edges():,} import edges  {cs_edges:,} cross-service")

        print("Loading vector index...")
        db = lancedb.connect(LANCE_PATH)
        lance_tbl = db.open_table("chunks")
        print(f"  {len(lance_tbl):,} vectors @ 4096d")

        try:
            doc_db = lancedb.connect(str(artifact_dir.parent / "output"))
            _tnames = doc_db.list_tables()
            _tnames = _tnames.tables if hasattr(_tnames, "tables") else list(_tnames)
            if "docs" in _tnames:
                doc_lance_tbl = doc_db.open_table("docs")
                print(f"  Doc vectors: {len(doc_lance_tbl):,} chunks @ 4096d")
            else:
                print("  docs.lance: not built yet — run 07_chunk_docs.py")
        except Exception as e:
            print(f"  docs.lance: {e}")

        print("Building module + filepath indexes...")
        for nid, d in G.nodes(data=True):
            mod = d.get("module", "")
            if mod:
                file_to_nodes.setdefault(mod, []).append(nid)
            f = d.get("file", "")
            m = d.get("module", "") or nid
            if f and m:
                filepath_to_module[f] = m

        # v6 stores (all optional — degrade gracefully)
        for path, store, name in [
            (BODY_STORE_P,   body_store,   "body store"),
            (CALL_GRAPH_P,   call_graph,   "call graph"),
            (LOG_PATTERNS_P, log_patterns, "log patterns"),
        ]:
            if path.exists():
                print(f"Loading {name}...")
                store.update(json.loads(path.read_text()))
                print(f"  {len(store):,} entries")
            else:
                print(f"  {name}: not found — run 01_extract_v2.py")

        if DOC_CHUNKS_P.exists():
            print("Loading doc chunks...")
            doc_chunks.extend(json.loads(DOC_CHUNKS_P.read_text()))
            doc_by_id.update({c["id"]: c for c in doc_chunks})
            print(f"  {len(doc_chunks):,} chunks")

        if GW_INTEGRITY_P.exists():
            print("Loading gateway integrity config...")
            gw_integrity.update(json.loads(GW_INTEGRITY_P.read_text()))
            print(f"  {len(gw_integrity):,} gateway configs")

        if cochange_path.exists():
            print("Loading co-change index...")
            ci = json.load(open(str(cochange_path)))
            cochange_index.update(ci.get("edges", {}))
            _cochange_loaded_at = cochange_path.stat().st_mtime
            meta = ci.get("meta", {})
            print(f"  {meta.get('total_modules',0):,} modules  {meta.get('total_pairs',0):,} pairs")
        else:
            print("  co-change index: not built yet")

        # Hot-reload watcher for co-change (used when builder is still running)
        def _cochange_watcher():
            global _cochange_loaded_at
            while True:
                time.sleep(30)
                try:
                    mtime = cochange_path.stat().st_mtime
                    if mtime > _cochange_loaded_at and cochange_path.stat().st_size > 1000:
                        ci = json.load(open(str(cochange_path)))
                        cochange_index.clear()
                        cochange_index.update(ci.get("edges", {}))
                        _cochange_loaded_at = mtime
                        meta = ci.get("meta", {})
                        print(f"[hot-reload] co-change: {meta.get('total_modules',0):,} modules")
                except Exception:
                    pass
        threading.Thread(target=_cochange_watcher, daemon=True).start()

    # Embedder: load in-process only if not using embed server
    if load_embedder and embedder is None and not using_embed_server:
        import torch
        from sentence_transformers import SentenceTransformer
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading embedding model on {device}...")
        embedder = SentenceTransformer(
            EMBED_MODEL, device=device, trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
        )
        print(f"  Loaded")
    elif using_embed_server:
        print(f"  Embed server: {EMBED_SERVER_URL} (no local GPU load)")
    else:
        print("  Embedder: skipped (keyword-only mode)")

    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    status = (
        f"embedder={'server' if using_embed_server else ('yes' if embedder else 'keyword-only')}, "
        f"bodies={'yes' if body_store else 'no'}, "
        f"cochange={'yes' if cochange_index else 'no'}"
    )
    print(f"✓ Engine ready ({status})")


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING  (local model OR embed server — transparent to callers)
# ════════════════════════════════════════════════════════════════════════════

def _encode_queries_batch(queries: list) -> list:
    """Encode multiple queries in one HTTP call to embed server, or locally."""
    if not queries:
        return []
    if EMBED_SERVER_URL:
        try:
            payload = json.dumps({"texts": queries, "instruction": EMBED_INSTRUCTION}).encode()
            req = urllib.request.Request(
                EMBED_SERVER_URL.rstrip("/") + "/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            return resp["embeddings"]
        except Exception as e:
            print(f"[embed_server] batch error: {e} — falling back to keyword search")
            return [[] for _ in queries]
    if embedder is not None:
        texts = [EMBED_INSTRUCTION + q for q in queries]
        vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vecs]
    return [[] for _ in queries]


def _encode_query(query: str) -> list:
    """Encode a single query."""
    vecs = _encode_queries_batch([query])
    return vecs[0] if vecs else []


def can_embed() -> bool:
    """True when vector search is available (either embed server or local model)."""
    return bool(EMBED_SERVER_URL) or embedder is not None


# ════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ════════════════════════════════════════════════════════════════════════════

def _is_test_path(fpath: str) -> bool:
    """Return True if the file lives in a test/harness directory."""
    parts = set(pathlib.PurePosixPath(fpath.replace("\\", "/")).parts)
    return bool(parts & _TEST_PATH_SEGMENTS)


def stratified_vector_search(queries: list, k_total: int = 250,
                              service_weights: dict = None) -> dict:
    """
    Multi-query vector search with dynamic per-service budget allocation.
    Returns {} when embedding is unavailable (keyword-only mode).
    """
    if not can_embed() or lance_tbl is None:
        return {}

    # Encode all queries in ONE HTTP call, then search LanceDB per vector
    vecs = _encode_queries_batch(queries)
    all_hits: dict = {}
    for qvec in vecs:
        if not qvec:
            continue
        for h in lance_tbl.search(qvec).limit(k_total).to_list():
            nid = h.get("id") or h.get("name", "")
            if nid and (nid not in all_hits or
                        h.get("_distance", 1) < all_hits[nid].get("_distance", 1)):
                all_hits[nid] = h

    # Down-weight test/harness paths so production code surfaces first
    for h in all_hits.values():
        if _is_test_path(h.get("file", "")):
            h["_distance"] = h.get("_distance", 1.0) * 1.5

    if service_weights:
        total_w = sum(service_weights.values()) or 1.0
        budgets = {svc: max(3, int(k_total * w / total_w)) for svc, w in service_weights.items()}
        default_budget = max(8, k_total // max(len(KNOWN_SERVICES), 1))
    else:
        budgets = {}
        default_budget = 12

    by_service: dict = defaultdict(list)
    for h in sorted(all_hits.values(), key=lambda x: x.get("_distance", 1.0)):
        svc = h.get("service", "unknown")
        cap = budgets.get(svc, default_budget)
        if len(by_service[svc]) < cap:
            by_service[svc].append(h)

    return dict(by_service)


def cross_service_keyword_search(query: str, max_per_service: int = 15) -> dict:
    if G is None:
        return {}
    words = [
        w.lower() for w in re.split(r"\W+", query)
        if (len(w) >= 4 or w.lower() in _KW_ALLOWLIST) and w.lower() not in _KW_STOPWORDS
    ]
    if not words:
        return {}
    prod_results: dict = defaultdict(list)
    test_results: dict = defaultdict(list)
    seen: set = set()
    for nid, d in G.nodes(data=True):
        if d.get("kind") == "phantom":
            continue
        haystack = (d.get("name", "") + " " + d.get("module", "")).lower()
        match_count = sum(1 for w in words if w in haystack)
        if match_count > 0 and nid not in seen:
            svc = d.get("service", "unknown")
            seen.add(nid)
            node = {**d, "id": nid, "_kw_score": match_count}
            bucket = test_results if _is_test_path(d.get("file", "")) else prod_results
            bucket[svc].append(node)

    # Sort by match count descending — nodes matching more query words rank first
    for bucket in (prod_results, test_results):
        for svc in bucket:
            bucket[svc].sort(key=lambda n: -n.get("_kw_score", 0))

    # Merge: fill each service slot with production results first, test code as overflow
    results: dict = defaultdict(list)
    all_svcs = set(prod_results) | set(test_results)
    for svc in all_svcs:
        combined = prod_results[svc][:max_per_service]
        remaining = max_per_service - len(combined)
        if remaining > 0:
            combined += test_results[svc][:remaining]
        results[svc] = combined
    return dict(results)


def doc_vector_search(query_vec: list, top_k: int = 20) -> list:
    if doc_lance_tbl is None or not query_vec:
        return []
    try:
        return doc_lance_tbl.search(query_vec).limit(top_k).to_list()
    except Exception as e:
        print(f"[doc_vector_search] error: {e}")
        return []


def module_graph_expand(seed_modules: list, depth: int = 2) -> dict:
    if MG is None or not seed_modules:
        return {}
    visited = {}
    queue = [(m, 0) for m in seed_modules if m in MG]
    for m in seed_modules:
        if m in MG:
            visited[m] = {"service": MG.nodes[m].get("service",""), "hop": 0, "direction": "seed"}
    while queue:
        current, hop = queue.pop(0)
        if hop >= depth:
            continue
        for nb in list(MG.successors(current)) + list(MG.predecessors(current)):
            if nb not in visited:
                direction = "imports" if nb in MG.successors(current) else "imported_by"
                visited[nb] = {"service": MG.nodes[nb].get("service",""), "hop": hop+1, "direction": direction}
                queue.append((nb, hop+1))
    return {m: d for m, d in visited.items() if d["hop"] > 0}


def cochange_path_traverse(seed_modules: list, max_hops: int = 4,
                            top_k: int = 15, min_weight: int = 5) -> list:
    if not cochange_index:
        return []
    visited = {}
    queue = [(m, 999, 0) for m in seed_modules if m in cochange_index]
    for mod in seed_modules:
        visited[mod] = (999, 0)
    while queue:
        current, _, hop = queue.pop(0)
        if hop >= max_hops:
            continue
        for p in cochange_index.get(current, [])[:top_k]:
            pm, pw = p["module"], p["weight"]
            if pw < min_weight:
                break
            if pm not in visited:
                visited[pm] = (pw, hop+1)
                queue.append((pm, pw, hop+1))
    result = [{"module": m, "weight": w, "hop": h} for m,(w,h) in visited.items() if h > 0]
    result.sort(key=lambda x: (x["hop"], -x["weight"]))
    return result


def get_entry_points(query_words: list) -> list:
    if G is None:
        return []
    entry_keywords = {"route","server","api","handler","endpoint","app","main","controller","middleware","router"}
    results = []
    for nid, d in G.nodes(data=True):
        if d.get("kind") == "phantom":
            continue
        mod  = d.get("module","").lower()
        name = d.get("name","").lower()
        if (any(k in mod or k in name for k in entry_keywords)
                and any(w in name or w in mod for w in query_words)):
            results.append({**d, "id": nid, "_entry": True})
            if len(results) >= 8:
                break
    return results


def get_cluster_context_for_services(services: list) -> dict:
    if G is None:
        return {}
    svc_clusters: dict = defaultdict(set)
    for nid, d in G.nodes(data=True):
        svc = d.get("service","")
        if svc in services:
            cid = str(d.get("cluster",-1))
            if cid != "-1" and cid in cluster_summaries:
                svc_clusters[svc].add(cid)
    return {
        svc: [{"cluster_id": cid, **cluster_summaries[cid]} for cid in cids]
        for svc, cids in svc_clusters.items()
    }


# ════════════════════════════════════════════════════════════════════════════
# AGENT TOOL IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════════════════

def _fuzzy_fn_lookup(fn_id: str) -> tuple:
    if fn_id in body_store:
        return fn_id, body_store[fn_id]
    parts = re.split(r'[.:]', fn_id)
    # Try progressively shorter suffix matches (most specific first).
    # Minimum 2 components — single-name matches are too ambiguous.
    for n_comps in range(len(parts), 1, -1):
        suffix = ".".join(parts[-n_comps:])
        candidates = [k for k in body_store
                      if k == suffix
                      or k.endswith("." + suffix)
                      or k.endswith("::" + suffix)]
        if candidates:
            best = max(candidates, key=lambda k: len(os.path.commonprefix([fn_id, k])))
            return best, body_store[best]
    return None, ""


def _same_name_impls(fn_id: str, exclude: str | None = None) -> list[str]:
    """Return all body_store keys that share the short function name with fn_id."""
    fn_short = fn_id.split(".")[-1].split("::")[-1]
    return [k for k in body_store
            if k != exclude
            and (k.split(".")[-1] == fn_short or k.split("::")[-1] == fn_short)]


def tool_get_function_body(fn_id: str, reason: str = "") -> str:
    matched, body = _fuzzy_fn_lookup(fn_id)
    if body:
        lines = [f"FUNCTION BODY: {matched}"]
        if reason:
            lines.append(f"Reason: {reason}")
        lines += ["```", body[:3000], "```"]
        logs = log_patterns.get(matched, [])
        if logs:
            lines.append(f"\nLog patterns: {json.dumps(logs[:5])}")
        callees = call_graph.get(matched, {}).get("callees", [])
        if callees:
            lines.append(f"\nDirect callees: {', '.join(callees[:12])}")

        # Fix B: auto-expand if body is a short delegation (≤8 non-blank, non-comment lines)
        real_lines = [l for l in body.strip().split("\n")
                      if l.strip() and not l.strip().startswith("--") and not l.strip().startswith("//")]
        if len(real_lines) <= 8 and callees:
            expanded = []
            for callee in callees[:3]:
                _, cb = _fuzzy_fn_lookup(callee)
                if cb:
                    expanded.append(f"\n  CALLEE {callee}:\n  ```\n  {cb[:1500]}\n  ```")
            if expanded:
                lines.append("\n--- AUTO-EXPANDED (delegation detected) ---")
                lines.extend(expanded)

        # Fix D: show other implementations with the same function name
        others = _same_name_impls(fn_id, exclude=matched)
        if others:
            lines.append(f"\nOther implementations of same name: {', '.join(others[:6])}")
            lines.append("→ Call get_function_body() on these to see provider/module-specific logic")

        return "\n".join(lines)

    if G is None:
        return f"NOT FOUND: '{fn_id}' (graph not loaded)."
    node_data = G.nodes.get(fn_id, {})
    if not node_data:
        fn_short = fn_id.split(".")[-1].split("::")[-1]
        matches = [(nid, d) for nid, d in G.nodes(data=True) if d.get("name") == fn_short]
        if matches:
            fn_id, node_data = matches[0][0], matches[0][1]
    if node_data:
        lines = [f"SYMBOL (no body extracted): {fn_id}",
                 f"Kind:  {node_data.get('kind','?')}",
                 f"Type:  {node_data.get('type','?')}",
                 f"File:  {node_data.get('file','?')}"]
        if node_data.get("constructors"):
            lines.append(f"Constructors: {', '.join(node_data['constructors'][:10])}")
        if node_data.get("fields"):
            lines.append(f"Fields: {', '.join(node_data['fields'][:10])}")
        if node_data.get("purpose"):
            lines.append(f"Purpose: {node_data['purpose'][:200]}")
        if node_data.get("cluster_name"):
            lines.append(f"Cluster: {node_data['cluster_name']}")
        # Fix C: include known callees so the LLM can continue drilling
        callees = call_graph.get(fn_id, {}).get("callees", [])
        if callees:
            lines.append(f"\nKnown callees (drill into these): {', '.join(callees[:8])}")
        # Fix D: show body_store implementations with same name
        others = _same_name_impls(fn_id)
        if others:
            lines.append(f"\nImplementations with same name in body_store: {', '.join(others[:6])}")
            lines.append("→ Call get_function_body() on these IDs to see provider-specific logic")
        return "\n".join(lines)

    # Complete NOT FOUND — still surface same-name body_store hits
    others = _same_name_impls(fn_id)
    if others:
        return (f"NOT FOUND: '{fn_id}' in graph.\n"
                f"But found same-name implementations: {', '.join(others[:6])}\n"
                f"→ Call get_function_body() on these IDs")
    return f"NOT FOUND: '{fn_id}'."


def tool_trace_callees(fn_id: str, reason: str = "") -> str:
    matched, _ = _fuzzy_fn_lookup(fn_id)
    target = matched or fn_id
    callees = call_graph.get(target, {}).get("callees", [])
    if not callees:
        return f"No callees found for '{target}'."

    # Derive the module prefix of the target so we can qualify short callee names
    # e.g. "Euler.API.Gateway.Gateway.PayU.Routes.payuBaseUrl" → "Euler.API.Gateway.Gateway.PayU.Routes"
    target_module = ".".join(target.split(".")[:-1]) if "." in target else ""

    # Build a name→id index for fast lookup — prefer same-module matches
    name_to_id: dict = {}
    if G:
        for nid, d in G.nodes(data=True):
            n = d.get("name", "")
            if not n:
                continue
            mod = d.get("module", "")
            # Same-module match wins; otherwise keep first seen
            if n not in name_to_id or (target_module and mod and target_module.lower() in mod.lower()):
                name_to_id[n] = (nid, d)

    lines = [f"CALLEES of {target}:"]
    if reason:
        lines.append(f"Reason: {reason}")
    for callee in callees[:20]:
        # Try: same-module qualified ID first, then graph lookup by name
        qualified = f"{target_module}.{callee}" if target_module else callee
        node_data = G.nodes.get(qualified) if G else None
        if node_data:
            lines.append(f"  → {qualified} :: {node_data.get('type','?')[:70]}  @ {node_data.get('file','?')}")
        elif callee in name_to_id:
            nid, d = name_to_id[callee]
            lines.append(f"  → {nid} :: {d.get('type','?')[:70]}  @ {d.get('file','?')}")
        else:
            lines.append(f"  → {callee}  (unresolved — local var or stdlib)")
    if len(callees) > 20:
        lines.append(f"  ... and {len(callees)-20} more")
    lines.append("\n→ Call get_function_body() using the full ID shown above")
    return "\n".join(lines)


def tool_trace_callers(fn_id: str, reason: str = "") -> str:
    matched, _ = _fuzzy_fn_lookup(fn_id)
    target = matched or fn_id
    callers = call_graph.get(target, {}).get("callers", [])
    if not callers:
        return f"No callers found for '{target}'."
    lines = [f"CALLERS of {target}:"]
    if reason:
        lines.append(f"Reason: {reason}")
    for caller in callers[:15]:
        # Callers are stored as full IDs — look up directly in G, not by short name
        node_data = G.nodes.get(caller) if G else None
        if node_data:
            lines.append(f"  ← {caller} :: {node_data.get('type','?')[:70]}  @ {node_data.get('file','?')}")
        else:
            lines.append(f"  ← {caller}")
    if len(callers) > 15:
        lines.append(f"  ... and {len(callers)-15} more")
    lines.append("\n→ Call trace_callers() on any caller above to walk further up the chain")
    return "\n".join(lines)


def tool_get_log_patterns(fn_id: str) -> str:
    matched, _ = _fuzzy_fn_lookup(fn_id)
    target = matched or fn_id
    logs = log_patterns.get(target, [])
    if not logs:
        return f"No log patterns found for '{target}'."
    return "LOG PATTERNS in " + target + ":\n" + "\n".join(f"  • {p}" for p in logs[:10])


def tool_search_symbols(query: str, service: str = "", brief: bool = False) -> str:
    vec_by_svc = stratified_vector_search([query], k_total=150)
    kw_by_svc  = cross_service_keyword_search(query, max_per_service=20)

    # Collect unique modules referenced by results — suggest get_module calls
    referenced_modules: dict = {}  # module -> (svc, count)

    lines = [f"SYMBOL SEARCH: '{query}'"]
    for svc in sorted(set(list(vec_by_svc) + list(kw_by_svc))):
        if service and svc != service:
            continue
        seen_ids: set = set()
        svc_lines = []

        # Vector results first (semantic similarity), then keyword-only hits
        for h in vec_by_svc.get(svc, []):
            nid = h.get("id") or h.get("name", "?")
            if nid not in seen_ids:
                sig = "" if brief else f" :: {h.get('type','?')[:60]}"
                svc_lines.append(f"  ID: {nid}{sig}")
                seen_ids.add(nid)
                mod = h.get("module", "")
                if mod:
                    referenced_modules[mod] = referenced_modules.get(mod, (svc, 0))
                    referenced_modules[mod] = (svc, referenced_modules[mod][1] + 1)

        for n in kw_by_svc.get(svc, []):
            nid = n.get("id") or n.get("name", "?")
            if nid not in seen_ids:
                sig = "" if brief else f" :: {n.get('type','?')[:60]}"
                svc_lines.append(f"  ID: {nid}{sig}")
                seen_ids.add(nid)
                mod = n.get("module", "")
                if mod:
                    referenced_modules[mod] = referenced_modules.get(mod, (svc, 0))
                    referenced_modules[mod] = (svc, referenced_modules[mod][1] + 1)

        lines.extend(svc_lines[:12])

    # Suggest hot modules — those with 2+ hits are worth a get_module call
    hot_mods = sorted(
        [(mod, svc, cnt) for mod, (svc, cnt) in referenced_modules.items() if cnt >= 2],
        key=lambda x: -x[2]
    )[:6]
    if hot_mods:
        lines.append("\n→ Modules with multiple hits (call get_module for full picture):")
        for mod, svc, cnt in hot_mods:
            lines.append(f"  get_module(\"{mod}\")  [{svc}, {cnt} symbols]")

    lines.append("\n→ Use the ID: value above with get_function_body(). IDs are dot-separated module paths — never file paths or slashes.")
    return "\n".join(lines) if len(lines) > 1 else f"No symbols found for '{query}'."


def tool_search_modules(query: str, service: str = "") -> str:
    """
    Search module namespaces by keyword. Returns matching module paths that can
    be passed directly to get_module(). Much faster than search_symbols when you
    know the component name (gateway, feature, domain) but not the exact function.
    """
    if G is None:
        return "Graph not loaded."

    words = [w.lower() for w in re.split(r"\W+", query) if len(w) >= 3]
    if not words:
        return "Query too short."

    # Collect all unique modules and score by how many query words they contain
    mod_scores: dict = {}  # module -> (svc, score, symbol_count)
    for nid, d in G.nodes(data=True):
        if d.get("kind") == "phantom":
            continue
        mod = (d.get("module", "") or "").lower().replace("::", ".")
        if not mod:
            continue
        svc = d.get("service", "")
        if service and svc != service:
            continue
        score = sum(1 for w in words if w in mod)
        if score > 0:
            existing = mod_scores.get(mod, (svc, 0, 0))
            mod_scores[mod] = (svc, max(existing[1], score), existing[2] + 1)

    if not mod_scores:
        return f"No modules found matching '{query}'."

    # Sort by score desc, then by symbol count desc
    ranked = sorted(mod_scores.items(), key=lambda x: (-x[1][1], -x[1][2]))

    # Deduplicate: prefer leaf modules over parent prefixes when parent score = leaf score
    seen_prefixes: set = set()
    deduped = []
    for mod, (svc, score, cnt) in ranked:
        if not any(mod.startswith(p + ".") for p in seen_prefixes):
            deduped.append((mod, svc, score, cnt))
        seen_prefixes.add(mod)
        if len(deduped) >= 20:
            break

    lines = [f"MODULE SEARCH: '{query}' — {len(deduped)} modules found"]
    for mod, svc, score, cnt in deduped:
        lines.append(f"  get_module(\"{mod}\")  [{svc}, {cnt} symbols, score={score}]")
    lines.append("\n→ Call get_module() with any path above to see all symbols in that namespace.")
    return "\n".join(lines)


def tool_get_module(module_name: str, service: str = "", max_symbols: int = 30) -> str:
    """
    Return all symbols in a module or module prefix.
    Lets the LLM explore an entire namespace in one call instead of
    making multiple get_function_body calls for sibling functions.
    """
    if G is None:
        return "Graph not loaded."

    q = module_name.lower().replace("::", ".").replace("/", ".")
    matches: list = []
    for nid, d in G.nodes(data=True):
        if d.get("kind") == "phantom":
            continue
        mod = (d.get("module", "") or "").lower().replace("::", ".")
        if not mod:
            continue
        svc = d.get("service", "")
        if service and svc != service:
            continue
        if mod == q or mod.startswith(q + ".") or mod.endswith("." + q) or q in mod:
            matches.append((nid, d))

    # Also check body_store keys for module prefix matches
    body_hits: list = []
    for k in body_store:
        k_mod = ".".join(k.split(".")[:-1]).lower()
        if q in k_mod:
            if not any(nid == k for nid, _ in matches):
                body_hits.append(k)

    if not matches and not body_hits:
        # Try fuzzy: any part of the module name
        parts = q.split(".")
        for nid, d in G.nodes(data=True):
            if d.get("kind") == "phantom":
                continue
            mod = (d.get("module", "") or "").lower()
            if any(p in mod for p in parts if len(p) > 4):
                matches.append((nid, d))
                if len(matches) >= max_symbols:
                    break

    if not matches and not body_hits:
        return f"Module '{module_name}' not found. Try a shorter prefix or check the service name."

    # Group by exact module
    by_mod: dict = defaultdict(list)
    for nid, d in matches[:max_symbols]:
        mod_key = d.get("module", d.get("service", "?"))
        by_mod[mod_key].append((nid, d))

    lines = [f"MODULE: '{module_name}' — {len(matches)} symbols found"]
    for mod_key in sorted(by_mod):
        lines.append(f"\n  [{mod_key}]")
        for nid, d in by_mod[mod_key][:max_symbols]:
            sig  = d.get("type", "")
            name = d.get("name", nid)
            fpath = d.get("file", "")
            entry = f"    {name}"
            if sig:
                entry += f" :: {sig[:80]}"
            if fpath:
                entry += f"  @ {fpath}"
            if nid in body_store:
                entry += "  [body available]"
            lines.append(entry)

    if body_hits:
        lines.append(f"\n  [body_store only — not in graph]")
        for k in body_hits[:10]:
            lines.append(f"    {k}  [body available]")

    lines.append(f"\n→ Call get_function_body() on any ID above to read its implementation.")
    return "\n".join(lines)


def tool_search_docs(query: str, tags: list = None) -> str:
    if not doc_chunks:
        return "Documentation not loaded. Run 07_chunk_docs.py."
    results = []
    if doc_lance_tbl is not None and can_embed():
        try:
            qvec = _encode_query(query)
            hits = doc_lance_tbl.search(qvec).limit(10).to_list()
            if tags:
                hits = [h for h in hits if any(t in h.get("tags","") for t in tags)] or hits
            results = hits[:5]
        except Exception as e:
            print(f"[search_docs] {e}")
    if not results:
        words  = re.split(r'\W+', query.lower())
        scored = []
        for chunk in doc_chunks:
            text  = (chunk["section_title"] + " " + chunk["text"]).lower()
            score = sum(1 for w in words if len(w) >= 3 and w in text)
            if tags:
                ctags = chunk.get("tags","") if isinstance(chunk.get("tags"), str) else ",".join(chunk.get("tags",[]))
                score += sum(2 for t in tags if t in ctags)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda x: -x[0])
        results = [c for _,c in scored[:5]]
    if not results:
        return f"No docs found for '{query}'."
    lines = [f"DOCS for '{query}':"]
    for r in results:
        lines.append(f"\n### [{r.get('source_file','?')}] {r.get('section_title','?')}")
        if r.get("url"):
            lines.append(f"URL: {r['url']}")
        lines.append(r.get("text","")[:800] + "...")
    return "\n".join(lines)


def tool_get_gateway_integrity(gateway_name: str) -> str:
    if not gw_integrity:
        return ("Gateway integrity config not built yet.\n"
                "Run the gateway scan to build gateway_integrity_config.json.")
    gw_lower = gateway_name.lower()
    matched = next((k for k in gw_integrity if k.lower() == gw_lower or gw_lower in k.lower()), None)
    if not matched:
        return (f"Gateway '{gateway_name}' not found.\nAvailable: {', '.join(list(gw_integrity)[:20])}")
    lines = [f"GATEWAY INTEGRITY: {matched}"]
    for k, v in gw_integrity[matched].items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def tool_get_type_definition(type_name: str, service: str = "") -> str:
    if G is None:
        return "Graph not loaded."
    matches = [(nid, d) for nid, d in G.nodes(data=True)
               if d.get("name") == type_name and d.get("kind") in ("type","class")
               and (not service or d.get("service") == service)]
    if not matches:
        matches = [(nid, d) for nid, d in G.nodes(data=True)
                   if type_name.lower() in d.get("name","").lower()
                   and d.get("kind") in ("type","class")][:5]
    if not matches:
        return f"Type '{type_name}' not found."
    lines = [f"TYPE DEFINITIONS for '{type_name}':"]
    for nid, d in matches[:5]:
        lines.append(f"\n  {d.get('service','')} / {d.get('module','')} / {d.get('name','?')}")
        lines.append(f"  File: {d.get('file','?')}")
        if d.get("type"):
            lines.append(f"  :: {d['type'][:200]}")
        body = body_store.get(nid,"") or _fuzzy_fn_lookup(nid)[1]
        if body:
            lines.append(f"  Body:\n  {body[:600]}")
    return "\n".join(lines)


def tool_get_context(query: str) -> str:
    """Last-resort: build full cross-service context for a query (expensive, ~5k-18k tokens)."""
    if G is None:
        return "Graph not loaded."
    vec_by_svc  = stratified_vector_search(query)
    kw_by_svc   = cross_service_keyword_search(query)
    cluster_by_svc = get_cluster_context_for_services(
        list(set(list(vec_by_svc) + list(kw_by_svc)))
    )
    doc_hits = doc_vector_search(query, k=3) if doc_lance_tbl is not None else []
    return _build_base_context(vec_by_svc, kw_by_svc, cluster_by_svc, "default", doc_hits)


TOOL_DISPATCH: dict = {
    "get_function_body":     lambda a: tool_get_function_body(a.get("fn_id",""), a.get("reason","")),
    "trace_callees":         lambda a: tool_trace_callees(a.get("fn_id",""), a.get("reason","")),
    "trace_callers":         lambda a: tool_trace_callers(a.get("fn_id",""), a.get("reason","")),
    "get_log_patterns":      lambda a: tool_get_log_patterns(a.get("fn_id","")),
    "search_symbols":        lambda a: tool_search_symbols(a.get("query",""), a.get("service",""), a.get("brief", False)),
    "search_modules":        lambda a: tool_search_modules(a.get("query",""), a.get("service","")),
    "get_module":            lambda a: tool_get_module(a.get("module_name",""), a.get("service",""), a.get("max_symbols", 30)),
    "get_blast_radius":      lambda a: str(get_blast_radius(a.get("files_or_modules", []))),
    "get_context":           lambda a: tool_get_context(a.get("query","")),
    "search_docs":           lambda a: tool_search_docs(a.get("query",""), a.get("tags",[])),
    "get_gateway_integrity": lambda a: tool_get_gateway_integrity(a.get("gateway_name","")),
    "get_type_definition":   lambda a: tool_get_type_definition(a.get("type_name",""), a.get("service","")),
}


# ════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDING
# ════════════════════════════════════════════════════════════════════════════

def _fmt_symbol(d: dict, include_body: bool = True) -> str:
    lang  = d.get("lang","?")
    kind  = d.get("kind","?")
    name  = d.get("name","?")
    cname = d.get("cluster_name","") or "(unclustered)"
    sig   = d.get("type","")
    fpath = d.get("file","")
    nid   = d.get("id", name)
    line  = f"    [{lang}/{kind}] {name}"
    if sig:
        line += f" :: {sig[:80]}"
    if fpath:
        line += f"  @ {fpath}"
    line += f"  — {cname}"
    if include_body and nid in body_store:
        preview = "\n".join(body_store[nid].split("\n")[:3])
        line += f"\n    Body preview: {preview[:150]}"
    return line


def build_service_context_block(vec_by_svc: dict, kw_by_svc: dict,
                                 include_bodies: bool = True) -> str:
    sections = []
    for svc in sorted(set(list(vec_by_svc) + list(kw_by_svc))):
        lines = []
        seen_names: set = set()
        ghost_deps: set = set()
        for h in vec_by_svc.get(svc, []):
            lines.append(_fmt_symbol(h, include_body=include_bodies))
            seen_names.add(h.get("name"))
            ghost_deps.update(h.get("ghost_deps") or [])
        if vec_by_svc.get(svc):
            lines.insert(0, "  [semantic matches]")
        kw_hits = [n for n in kw_by_svc.get(svc, []) if n.get("name") not in seen_names]
        if kw_hits:
            lines.append("  [keyword matches]")
            for n in kw_hits:
                lines.append(_fmt_symbol(n, include_body=include_bodies))
                ghost_deps.update(n.get("ghost_deps") or [])
        if ghost_deps:
            lines.append(f"  [external deps: {', '.join(sorted(ghost_deps)[:8])}]")
        if lines:
            sections.append(f"### {svc}:\n" + "\n".join(lines))
    return "\n\n".join(sections) if sections else "  (no results)"


def _build_base_context(vec_by_svc: dict, kw_by_svc: dict,
                         cluster_by_svc: dict, tool_name: str,
                         doc_hits: list = None) -> str:
    framework     = PERSONA_FRAMEWORKS.get(tool_name, "")
    context_block = build_service_context_block(vec_by_svc, kw_by_svc)

    # Collect cluster names referenced by matched symbols
    matched_cluster_names: set = set()
    for hits in list(vec_by_svc.values()) + list(kw_by_svc.values()):
        for h in hits:
            cn = h.get("cluster_name", "")
            if cn and cn != "(unclustered)":
                matched_cluster_names.add(cn)

    cluster_lines = []
    for svc in sorted(cluster_by_svc):
        summaries = cluster_by_svc[svc]
        if summaries:
            relevant = [c for c in summaries if c.get("name", "") in matched_cluster_names]
            if not relevant:
                relevant = summaries[:3]  # fallback: top 3 clusters per service (cluster_name empty in LanceDB — known issue)
            c_lines = "\n".join(
                f"    [{c.get('name','?')}]: {c.get('purpose','')[:120]}"
                f"\n      deps={', '.join(c.get('ghost_deps',[])[:5])}  "
                f"risks={', '.join(c.get('risk_flags',[])[:2])}"
                for c in relevant
            )
            cluster_lines.append(f"  {svc}:\n{c_lines}")
    cluster_block = "\n".join(cluster_lines) or "  (none)"

    seed_mods = list({
        h.get("module","") for hits in vec_by_svc.values() for h in hits[:3] if h.get("module")
    })
    mg_expanded = module_graph_expand(seed_mods, depth=2)
    mg_block = ""
    if mg_expanded:
        by_svc_hop: dict = defaultdict(list)
        for mod, info in mg_expanded.items():
            by_svc_hop[info["service"]].append((info["hop"], info["direction"], mod))
        mg_lines = []
        for svc in sorted(by_svc_hop):
            for hop, direction, mod in sorted(by_svc_hop[svc])[:5]:
                sample = [
                    G.nodes[nid].get("name","?")
                    for nid in file_to_nodes.get(mod,[])[:2]
                    if G and nid in G and G.nodes[nid].get("kind") != "phantom"
                ]
                mg_lines.append(
                    f"  [{svc}] hop={hop} ({direction}): {mod}"
                    + (f"  [{', '.join(sample)}]" if sample else "")
                )
        mg_block = "IMPORT GRAPH EXPANSION:\n" + "\n".join(mg_lines[:25])

    cc_path = cochange_path_traverse(seed_mods)
    cc_block = ""
    if cc_path:
        cc_block = "EVOLUTIONARY COUPLING (co-changed in git):\n" + "\n".join(
            f"  hop={c['hop']} weight={c['weight']:3d}: {c['module']}" for c in cc_path[:15]
        )

    kw_words = [
        w.lower() for w in re.split(r"\W+", " ".join(v for v in [framework[:50]] if v))
        if len(w) >= 4 and w.lower() not in _KW_STOPWORDS
    ]
    entries = get_entry_points(kw_words) if kw_words else []
    entry_block = ""
    if entries:
        entry_block = "ENTRY POINTS:\n" + "\n".join(_fmt_symbol(e, include_body=False) for e in entries[:5])

    doc_block = ""
    if doc_hits:
        doc_block = "DOCUMENTATION:\n" + "\n\n".join(
            f"  [{d.get('source_file','?')}] {d.get('section_title','?')}"
            + (f"  ({d.get('url','')})" if d.get("url") else "")
            + f"\n  {d.get('text','')[:600]}"
            for d in doc_hits
        )

    parts = [f"SUBSYSTEM SUMMARIES:\n{cluster_block}", f"CODE EVIDENCE BY SERVICE:\n{context_block}"]
    for b in [doc_block, mg_block, cc_block, entry_block]:
        if b:
            parts.append(b)
    return "\n\n".join(parts)


def _build_answer_messages(messages: list, query: str, tool_name: str,
                            investigation_summary: str) -> list:
    framework    = PERSONA_FRAMEWORKS.get(tool_name, "")
    final_prompt = f"Now produce the complete answer for: '{query}'\n\nApply this framework:\n{framework}"
    if investigation_summary:
        final_prompt += investigation_summary
    return messages + [{"role": "user", "content": final_prompt}]


# ════════════════════════════════════════════════════════════════════════════
# FILE → MODULE RESOLUTION  (for PR blast-radius analysis)
# ════════════════════════════════════════════════════════════════════════════

def resolve_files_to_modules(file_paths: list) -> dict:
    """
    Map git-diff file paths to module names.

    Tries in order:
    1. Exact suffix match against filepath_to_module keys
    2. Partial suffix match (shrinking window of path components)
    3. Stem match against MG node names (basename without extension)

    Returns {file_path: [module_name, ...]}. Unmatched → [].
    """
    result = {}
    for fp in file_paths:
        fp_norm = fp.replace("\\", "/").lstrip("/")
        found = []

        # 1 + 2: suffix match with shrinking window
        fp_parts = fp_norm.split("/")
        for n in range(len(fp_parts), 0, -1):
            suffix = "/".join(fp_parts[-n:])
            for known_path, mod in filepath_to_module.items():
                known_norm = known_path.replace("\\", "/").lstrip("/")
                if known_norm == suffix or known_norm.endswith("/" + suffix) or suffix.endswith("/" + known_norm):
                    if mod not in found:
                        found.append(mod)
            if found:
                break

        # 3: stem match against MG
        if not found and MG is not None:
            stem = pathlib.Path(fp_norm).stem
            for mod in MG.nodes():
                if mod.split(".")[-1] == stem or mod.split("::")[-1] == stem:
                    if mod not in found:
                        found.append(mod)
                    if len(found) >= 3:
                        break

        result[fp] = found
    return result


# ════════════════════════════════════════════════════════════════════════════
# BLAST RADIUS  (composite: import graph + co-change)
# ════════════════════════════════════════════════════════════════════════════

def get_blast_radius(module_names: list, max_hops: int = 2) -> dict:
    """
    Returns:
    {
      "seed_modules":       [str, ...],
      "import_neighbors":   [{"module", "service", "hop", "direction"}, ...],
      "cochange_neighbors": [{"module", "weight", "hop"}, ...],
      "affected_services":  [str, ...],
    }
    """
    seed = [m for m in module_names if m]
    affected_services: set = set()
    import_neighbors = []
    cochange_neighbors = []

    if MG is not None:
        for m in seed:
            if m in MG.nodes:
                svc = MG.nodes[m].get("service","")
                if svc:
                    affected_services.add(svc)
        for mod, info in module_graph_expand(seed, depth=max_hops).items():
            import_neighbors.append({
                "module": mod, "service": info.get("service",""),
                "hop": info["hop"], "direction": info["direction"],
            })
            if info.get("service"):
                affected_services.add(info["service"])

    if cochange_index:
        cochange_neighbors = cochange_path_traverse(seed, max_hops=max_hops)
        for item in cochange_neighbors:
            if MG is not None and item["module"] in MG.nodes:
                svc = MG.nodes[item["module"]].get("service","")
                if svc:
                    affected_services.add(svc)

    import_neighbors.sort(key=lambda x: (x["hop"], x["module"]))

    return {
        "seed_modules":       seed,
        "import_neighbors":   import_neighbors,
        "cochange_neighbors": cochange_neighbors,
        "affected_services":  sorted(affected_services),
    }


# ════════════════════════════════════════════════════════════════════════════
# SYNCHRONOUS AGENT LOOP  (no Chainlit — for MCP server, CLI, CI)
# ════════════════════════════════════════════════════════════════════════════

def run_agent_loop_sync(
    query: str,
    tool_name: str,
    initial_context: str,
    history: list,
    max_tool_calls: int = 50,
    on_tool_call=None,   # callback(fn_name, args, result) — for progress display
) -> tuple:
    """
    Synchronous agent loop using TOOL_DISPATCH. No Chainlit dependency.
    Returns (messages_for_final_answer, investigation_summary).
    on_tool_call is called synchronously after each tool invocation.
    """
    if _llm_client is None:
        raise RuntimeError("Call retrieval_engine.initialize() before run_agent_loop_sync()")

    system_prompt = PERSONA_SYSTEM_PROMPTS.get(tool_name, _BASE_IDENTITY)
    accumulated   = []
    n_calls       = 0

    messages = list(history) + [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_routing", "type": "function",
                            "function": {"name": tool_name,
                                         "arguments": json.dumps({"intent": query, "focus": ""})}}]
        },
        {"role": "tool", "tool_call_id": "call_routing", "content": initial_context},
    ]

    while n_calls < max_tool_calls:
        resp = _llm_client.chat.completions.create(
            model=LLM_MODEL, messages=messages, tools=AGENT_TOOLS,
            tool_choice="auto", temperature=0.1, max_tokens=2000,
        )
        asst_msg = resp.choices[0].message
        if not asst_msg.tool_calls:
            break

        tc_results = []
        for tc in asst_msg.tool_calls:
            n_calls += 1
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            dispatcher = TOOL_DISPATCH.get(fn_name)
            result     = dispatcher(args) if dispatcher else f"Unknown tool: {fn_name}"
            if on_tool_call is not None:
                try:
                    on_tool_call(fn_name, args, result)
                except Exception:
                    pass
            tc_results.append({"tool_call_id": tc.id, "role": "tool", "content": result})
            accumulated.append(f"[{fn_name}]: {result[:200]}...")

        messages.append({
            "role": "assistant", "content": asst_msg.content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in asst_msg.tool_calls]
        })
        messages.extend(tc_results)

        if n_calls >= max_tool_calls:
            messages.append({"role": "user",
                             "content": "You've reached the investigation limit. Synthesize your findings now."})
            break

    summary = ""
    if accumulated:
        summary = (f"\n\n[Investigation: {n_calls} tool calls — "
                   + "; ".join(accumulated[:3]) + "]")
    return messages, summary


def get_expert_answer(
    query: str,
    tool_name: str,
    initial_context: str,
    history: list = None,
    max_tool_calls: int = 50,
    on_tool_call=None,
) -> str:
    """
    Full agentic Q&A: agent loop + final LLM answer call. Returns answer string.
    Convenience wrapper used by MCP server and CLI tools.
    """
    msgs, summary = run_agent_loop_sync(
        query, tool_name, initial_context, history or [],
        max_tool_calls=max_tool_calls, on_tool_call=on_tool_call,
    )
    answer_msgs = _build_answer_messages(msgs, query, tool_name, summary)
    resp = _llm_client.chat.completions.create(
        model=LLM_MODEL, messages=answer_msgs, temperature=0.2, max_tokens=8000,
    )
    return resp.choices[0].message.content or ""
