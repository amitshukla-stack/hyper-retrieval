"""
tools.py — Agent tools layer (org-customizable).

This is where you add, remove, or modify tools for your deployment.
- Add a new tool: write a function, add its schema to AGENT_TOOLS, add it to TOOL_DISPATCH.
- Override persona: set _DEFAULT_SYSTEM_PROMPT and _DEFAULT_FRAMEWORK below.
- The retrieval primitives (vector search, graph traversal) live in retrieval_engine.py — no need to touch them.
"""
import sys, json, os, re, hashlib, threading
import pathlib

# ── HyperCode coding tools (apps/cli/tools/) ─────────────────────────────────
# Allows the Chainlit chat + MCP server to use the same coding tools as the CLI.
_CLI_TOOLS = pathlib.Path(__file__).parent / "apps" / "cli" / "tools"
# Load CLI tools by explicit file path to avoid circular `import tools` collision
# (root tools.py is already `tools` in sys.modules when the chat app loads us).
try:
    import importlib.util as _ilu

    def _load_cli_module(name: str, filepath: pathlib.Path):
        spec = _ilu.spec_from_file_location(name, str(filepath))
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _bash_mod = _load_cli_module("_hr_bash_tool",  _CLI_TOOLS / "bash_tool.py")
    _file_mod = _load_cli_module("_hr_file_tools", _CLI_TOOLS / "file_tools.py")
    _run_bash   = _bash_mod.run_bash
    _read_file  = _file_mod.read_file
    _write_file = _file_mod.write_file
    _edit_file  = _file_mod.edit_file
    _glob_files = _file_mod.glob_files
    _grep_files = _file_mod.grep_files
    _CODING_TOOLS_AVAILABLE = True
except Exception as _e:
    _CODING_TOOLS_AVAILABLE = False
    print(f"[tools] CLI tools not available: {_e}")

# ── Context chunk cache ───────────────────────────────────────────────────────
_CTX_CACHE: dict[str, list[str]] = {}
_CTX_LOCK   = threading.Lock()
_MAX_CTX_CACHE = 30   # max concurrent cached contexts (oldest evicted)
sys.path.insert(0, str(pathlib.Path(__file__).parent / "serve"))
import retrieval_engine as RE


# ════════════════════════════════════════════════════════════════════════════
# PERSONA CONTENT  (org-customizable — override via config.yaml personas section
#                   or by editing the strings directly)
# ════════════════════════════════════════════════════════════════════════════

_DEFAULT_SYSTEM_PROMPT = """
You are Codebase Expert — an interactive AI assistant embedded in your organisation's codebase. You help engineers understand, trace, debug, and reason about code across services using a set of retrieval tools.

## Service architecture

Service architecture is derived from the indexed codebase. Use `search_modules` and `get_module` to discover
the actual service topology — never invent or assume service names or call chains.

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
    A[ServiceA<br/>functionName] --> B{Decision Point}
    B -->|Path 1| C[ServiceB<br/>handleRequest]
    B -->|Path 2| D[ServiceC<br/>processFlow]
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

# Single-entry dicts — all keys point to the default identity
PERSONA_SYSTEM_PROMPTS: dict = {
    "default": _DEFAULT_SYSTEM_PROMPT,
}
PERSONA_FRAMEWORKS: dict = {
    "default": _DEFAULT_FRAMEWORK,
}
PERSONA_LABELS: dict = {
    "default": "Codebase Expert",
}

# Keep _BASE_IDENTITY as an alias for backward compatibility with any external callers
_BASE_IDENTITY = _DEFAULT_SYSTEM_PROMPT


def apply_persona_config(personas_cfg: dict) -> None:
    """
    Update PERSONA_SYSTEM_PROMPTS / PERSONA_FRAMEWORKS / PERSONA_LABELS from
    a config dict (the 'personas' section of config.yaml).

    Call this after RE.load_config() returns, passing cfg.get('personas', {}).
    """
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
            "several functions and they are independent of each other.\n\n"
            "If the response says 'SYMBOL (no body extracted)', the ID points to a type alias, "
            "constant, or data constructor — not a function. Do NOT retry get_function_body on it. "
            "Use get_type_definition instead for types, or read the surrounding module with get_module."
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
                "description": "Restrict search to a specific service name (e.g. 'api-gateway'). Omit to search all services."
            }
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": (
            "Search indexed internal developer documentation.\n\n"
            "Searches indexed developer documentation (markdown files processed during build).\n\n"
            "Use this when the question is about internal library usage, architecture notes, "
            "or code style guides. For questions about specific code behaviour, "
            "use search_symbols and get_function_body instead."
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
            "gateway_name": {"type": "string", "description": "Gateway or connector name as it appears in the codebase (e.g. 'stripe', 'adyen')."}
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
        "name": "fast_search",
        "description": (
            "Zero-GPU keyword search: BM25 + IDF graph index, RRF merged. Returns in <50ms "
            "with no embed server required.\n\n"
            "Use this when:\n"
            "- The embed server is not running (keyword-only deployments)\n"
            "- You have an exact identifier, class name, or module keyword\n"
            "- You want a quick symbol lookup without semantic overhead\n"
            "- You want to confirm a symbol exists before calling search_symbols\n\n"
            "For conceptual or natural-language queries, use search_symbols — it adds\n"
            "semantic vector search and co-change expansion for better recall."
        ),
        "parameters": {"type": "object", "properties": {
            "query":  {"type": "string", "description": "Symbol name, module keyword, or identifier to find"},
            "top_k":  {"type": "integer", "description": "Max results per service (default 10)"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "fast_search_reranked",
        "description": (
            "BM25 search with cross-encoder reranking. Requires HR_RERANKER=1 at startup; "
            "falls back to fast_search otherwise.\n\n"
            "Fetches BM25 top-30, then scores each candidate with a cross-encoder "
            "(ms-marco-MiniLM-L-6-v2) using symbol ID + function body as context. "
            "Adds ~20ms over fast_search. Significantly better for queries where the "
            "right symbol is in the BM25 window but ranked below irrelevant high-score matches.\n\n"
            "Use this when:\n"
            "- fast_search returns results but the top-3 feel wrong\n"
            "- Your query is conceptual ('webhook notification handler') not an identifier\n"
            "- You want the best keyword-mode result without GPU/embed server"
        ),
        "parameters": {"type": "object", "properties": {
            "query":  {"type": "string", "description": "Natural-language or identifier query"},
            "top_k":  {"type": "integer", "description": "Max total results (default 10)"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "search_requirements",
        "description": (
            "Search for functional requirement clusters that match a behavioral or flow query.\n\n"
            "Use this when the user asks HOW something works, not WHERE a specific symbol is:\n"
            "- 'How does retry logic work?'\n"
            "- 'What happens when a payment fails?'\n"
            "- 'Which modules handle authentication?'\n\n"
            "Returns requirement clusters — groups of modules that together implement a behavior. "
            "Each cluster has a one-sentence requirement description, confidence score, "
            "and the module names involved.\n\n"
            "Requires the requirements index to be built (build/11_build_requirements.py --embed). "
            "Falls back gracefully if index is absent."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Behavioral or flow question in natural language"},
            "k":     {"type": "integer", "description": "Max clusters to return (default 5)"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "get_why_context",
        "description": (
            "Returns WHY context for a module or symbol: ownership history, activity trend, "
            "Granger causal direction, criticality reasons, and anti-pattern flags.\n\n"
            "Use this BEFORE modifying critical code to understand:\n"
            "- Who owns this code and how often it changes\n"
            "- What causal relationships exist (this module predicts changes elsewhere)\n"
            "- Why the criticality score is what it is\n"
            "- Whether anti-patterns (god module, high churn, tight coupling) are present\n\n"
            "Companion to get_blast_radius (what breaks) — this answers what motivates "
            "the code and who to consult. Best called after search_symbols identifies "
            "the exact module name."
        ),
        "parameters": {"type": "object", "properties": {
            "symbol_name": {"type": "string", "description": "Fully-qualified module or symbol name (e.g. 'euler-api-gateway::PaymentRouter' or 'PaymentGateway.Router')"}
        }, "required": ["symbol_name"]}
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
            "Example: get_module('Auth.Middleware.JWT')"
        ),
        "parameters": {"type": "object", "properties": {
            "module_name": {"type": "string", "description": "Module namespace path in dot notation (e.g. 'Auth.Middleware.JWT')."},
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
            "Last resort — builds a full cross-service context block and returns part 1 of 3.\n\n"
            "Only call if search_symbols + get_function_body have genuinely failed after 3+ attempts. "
            "Part 1 alone is often sufficient — read it before deciding whether to call get_context_continue. "
            "Do not call this if a targeted search has not been tried yet."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The original user question."}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "get_context_continue",
        "description": (
            "Retrieve part 2 or 3 of a get_context result. "
            "Only call if the previous part was insufficient and you need more context. "
            "Use the token returned by get_context."
        ),
        "parameters": {"type": "object", "properties": {
            "token": {"type": "string", "description": "The token returned by get_context."},
            "part":  {"type": "integer", "description": "The part number to retrieve (2 or 3)."},
        }, "required": ["token", "part"]}
    }},

    # ── Guardian / PR analysis tools ─────────────────────────────────────────
    {"type": "function", "function": {
        "name": "predict_missing_changes",
        "description": (
            "Given a list of changed files or modules, predict what else should change.\n\n"
            "Uses co-change history + Granger causality to find files that historically change "
            "together with the given files but are missing from the changeset.\n\n"
            "Use this for PR review: 'I changed X and Y — am I missing anything?'"
        ),
        "parameters": {"type": "object", "properties": {
            "changed_files": {
                "type": "array", "items": {"type": "string"},
                "description": "List of changed file paths or module names."
            },
            "min_confidence": {
                "type": "number",
                "description": "Minimum confidence threshold 0-1 (default 0.1)."
            }
        }, "required": ["changed_files"]}
    }},
    {"type": "function", "function": {
        "name": "suggest_reviewers",
        "description": (
            "Suggest PR reviewers based on module ownership from git history.\n\n"
            "Returns ranked reviewers who have the most context on the affected modules, "
            "including modules in the blast radius — not just the directly changed files."
        ),
        "parameters": {"type": "object", "properties": {
            "changed_files": {
                "type": "array", "items": {"type": "string"},
                "description": "List of changed file paths or module names."
            }
        }, "required": ["changed_files"]}
    }},
    {"type": "function", "function": {
        "name": "score_change_risk",
        "description": (
            "Compute a composite risk score (0-100) for a set of changed files/modules.\n\n"
            "Combines blast radius, coverage gap, reviewer concentration, and service spread "
            "into one actionable number with a LOW/MEDIUM/HIGH/CRITICAL verdict.\n\n"
            "Use to gate PRs in CI/CD or prioritize review effort."
        ),
        "parameters": {"type": "object", "properties": {
            "changed_files": {
                "type": "array", "items": {"type": "string"},
                "description": "List of changed file paths or module names."
            }
        }, "required": ["changed_files"]}
    }},
    # ── HyperCode coding tools ────────────────────────────────────────────────
    # Enabled when apps/cli/tools/ is importable (_CODING_TOOLS_AVAILABLE = True).
    # These allow the Chainlit chat + MCP server to read/write/edit files and run
    # shell commands — full codetoolcli parity in the chat interface.

    {"type": "function", "function": {
        "name": "run_bash",
        "description": (
            "Execute a shell command and return its output.\n\n"
            "Use for: git operations, running scripts/tests/builds, checking system state.\n"
            "Rules:\n"
            "- Prefer non-interactive commands (avoid prompts)\n"
            "- For file reading prefer read_file (preserves line numbers)\n"
            "- Explain destructive operations before running them"
        ),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default 120)."},
        }, "required": ["command"]},
    }},

    {"type": "function", "function": {
        "name": "read_file",
        "description": (
            "Read a file with line numbers.\n\n"
            "Always read before editing — edit_file requires knowing the exact content. "
            "Use offset + limit to read a section of a large file."
        ),
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "File path (absolute or relative to cwd)."},
            "offset":    {"type": "integer", "description": "1-based start line (default: 1)."},
            "limit":     {"type": "integer", "description": "Max lines to return (default: 2000)."},
        }, "required": ["file_path"]},
    }},

    {"type": "function", "function": {
        "name": "write_file",
        "description": (
            "Write content to a file (creates or fully overwrites).\n\n"
            "For targeted changes to existing files, prefer edit_file."
        ),
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Destination file path."},
            "content":   {"type": "string", "description": "Full file content to write."},
        }, "required": ["file_path", "content"]},
    }},

    {"type": "function", "function": {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file.\n\n"
            "old_string must appear verbatim and be unique — add surrounding lines for context. "
            "Use replace_all=true to rename across the whole file."
        ),
        "parameters": {"type": "object", "properties": {
            "file_path":   {"type": "string", "description": "File to edit."},
            "old_string":  {"type": "string", "description": "Exact text to replace (must be unique in file)."},
            "new_string":  {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)."},
        }, "required": ["file_path", "old_string", "new_string"]},
    }},

    {"type": "function", "function": {
        "name": "glob_files",
        "description": (
            "Find files by glob pattern (supports **). "
            "Returns paths sorted by modification time.\n"
            "Examples: 'src/**/*.py', 'tests/test_*.py'"
        ),
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "path":    {"type": "string", "description": "Base directory (default: repo root)."},
        }, "required": ["pattern"]},
    }},

    {"type": "function", "function": {
        "name": "grep_files",
        "description": (
            "Search file contents for a regex pattern. Returns file:line matches.\n"
            "Uses ripgrep when available. "
            "Examples: grep_files('def.*payment'), grep_files('TODO', file_glob='*.py')"
        ),
        "parameters": {"type": "object", "properties": {
            "pattern":          {"type": "string", "description": "Regex pattern."},
            "path":             {"type": "string", "description": "Directory to search (default: repo root)."},
            "file_glob":        {"type": "string", "description": "Restrict to files matching this glob."},
            "case_insensitive": {"type": "boolean", "description": "Case-insensitive match (default false)."},
            "context_lines":    {"type": "integer", "description": "Context lines around matches (default 0)."},
        }, "required": ["pattern"]},
    }},
    # ── Guardrails / Criticality tools (SDLC guardian surface) ────────────────
    {"type": "function", "function": {
        "name": "check_my_changes",
        "description": (
            "SDLC guardian: run ALL quality checks on a set of changed files in one call. "
            "Combines blast_radius + predict_missing_changes + security scan + Guard static "
            "checks (comment-code alignment, lock/transaction/auth patterns). Returns a single "
            "PASS / WARN / FAIL verdict with reasons and action items. Designed for AI coding "
            "agents to self-check BEFORE committing.\n\n"
            "Use AFTER writing code but BEFORE committing — catches missing co-changes, "
            "unexpectedly large blast radius, comment/code mismatches, and security-sensitive "
            "modules that need review.\n\n"
            "Accepts file paths or module names.\n"
            "Examples:\n"
            "  check_my_changes(['api-gateway/src/routes.py'])\n"
            "  check_my_changes(['PaymentFlows', 'TransactionHelper'])"
        ),
        "parameters": {"type": "object", "properties": {
            "changed_files": {
                "type": "array", "items": {"type": "string"},
                "description": "Paths or module names of the files you changed."
            },
        }, "required": ["changed_files"]},
    }},
    {"type": "function", "function": {
        "name": "check_criticality",
        "description": (
            "Get the criticality score (0-1) of one or more modules. Score is computed from 7 "
            "signals: blast radius, cross-repo coupling, change frequency, author concentration, "
            "recency, Granger causal influence, and revert history. Use BEFORE changing critical "
            "code to quantify the risk.\n\n"
            "Returns per-module: risk level (CRITICAL/HIGH/MEDIUM/LOW), score, rank, key signals, "
            "and human-readable reasons.\n\n"
            "Examples: check_criticality(['Transaction']), check_criticality(['TenantConfig', 'PaymentFlows'])"
        ),
        "parameters": {"type": "object", "properties": {
            "modules": {
                "type": "array", "items": {"type": "string"},
                "description": "Module names to score."
            },
        }, "required": ["modules"]},
    }},
    {"type": "function", "function": {
        "name": "get_guardrails",
        "description": (
            "Fetch auto-generated guardrail documents for critical modules. Each guardrail "
            "explains: why this code is critical, what invariant must stay true, what breaks "
            "if you change it, and who should review changes. If a module has no guardrail, "
            "returns its criticality score with a note.\n\n"
            "Use when reviewing or writing PRs that touch critical code — the guardrail tells "
            "you what to preserve and what to test.\n\n"
            "Examples: get_guardrails(['Transaction']), get_guardrails(['CardInfo', 'CryptoUtils'])"
        ),
        "parameters": {"type": "object", "properties": {
            "modules": {
                "type": "array", "items": {"type": "string"},
                "description": "Module names to fetch guardrails for."
            },
        }, "required": ["modules"]},
    }},
    {"type": "function", "function": {
        "name": "list_critical_modules",
        "description": (
            "List the most critical modules in the codebase, ranked by risk score. Works on "
            "any codebase — no domain knowledge required. Use before major refactors, during "
            "onboarding, or when planning code reviews.\n\n"
            "Optional filter by service name. Default threshold 0.5 (moderately critical and up), "
            "default top_k 20."
        ),
        "parameters": {"type": "object", "properties": {
            "service":   {"type": "string",  "description": "Filter by service name (e.g., 'euler-api-txns'). Optional."},
            "threshold": {"type": "number",  "description": "Minimum criticality score (0-1). Default 0.5."},
            "top_k":     {"type": "integer", "description": "Number of results to return. Default 20."},
        }, "required": []},
    }},
]


# ════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# All RE.* references resolve after RE.initialize() populates module globals.
# ════════════════════════════════════════════════════════════════════════════

def _fuzzy_fn_lookup(fn_id: str) -> tuple:
    if fn_id in RE.body_store:
        return fn_id, RE.body_store[fn_id]
    parts = re.split(r'[.:]', fn_id)
    # Try progressively shorter suffix matches (most specific first).
    # Minimum 2 components — single-name matches are too ambiguous.
    for n_comps in range(len(parts), 1, -1):
        suffix = ".".join(parts[-n_comps:])
        candidates = [k for k in RE.body_store
                      if k == suffix
                      or k.endswith("." + suffix)
                      or k.endswith("::" + suffix)]
        if candidates:
            best = max(candidates, key=lambda k: len(os.path.commonprefix([fn_id, k])))
            return best, RE.body_store[best]
    return None, ""


def _same_name_impls(fn_id: str, exclude: str | None = None) -> list[str]:
    """Return all body_store keys that share the short function name with fn_id."""
    fn_short = fn_id.split(".")[-1].split("::")[-1]
    return [k for k in RE.body_store
            if k != exclude
            and (k.split(".")[-1] == fn_short or k.split("::")[-1] == fn_short)]


def tool_get_function_body(fn_id: str, reason: str = "") -> str:
    matched, body = _fuzzy_fn_lookup(fn_id)
    if body:
        lines = [f"FUNCTION BODY: {matched}"]
        if reason:
            lines.append(f"Reason: {reason}")
        lines += ["```", body[:3000], "```"]
        logs = RE.log_patterns.get(matched, [])
        if logs:
            lines.append(f"\nLog patterns: {json.dumps(logs[:5])}")
        callees = RE.call_graph.get(matched, {}).get("callees", [])
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

    if RE.G is None:
        return f"NOT FOUND: '{fn_id}' (graph not loaded)."
    node_data = RE.G.nodes.get(fn_id, {})
    if not node_data:
        fn_short = fn_id.split(".")[-1].split("::")[-1]
        matches = [(nid, d) for nid, d in RE.G.nodes(data=True) if d.get("name") == fn_short]
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
        callees = RE.call_graph.get(fn_id, {}).get("callees", [])
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
    callees = RE.call_graph.get(target, {}).get("callees", [])
    if not callees:
        return f"No callees found for '{target}'."

    # Derive the module prefix of the target so we can qualify short callee names
    # e.g. "Auth.Middleware.JWT.verifyToken" → "Auth.Middleware.JWT"
    target_module = ".".join(target.split(".")[:-1]) if "." in target else ""

    # Build a name→id index for fast lookup — prefer same-module matches
    name_to_id: dict = {}
    if RE.G:
        for nid, d in RE.G.nodes(data=True):
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
        node_data = RE.G.nodes.get(qualified) if RE.G else None
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
    callers = RE.call_graph.get(target, {}).get("callers", [])
    if not callers:
        return f"No callers found for '{target}'."
    lines = [f"CALLERS of {target}:"]
    if reason:
        lines.append(f"Reason: {reason}")
    for caller in callers[:15]:
        # Callers are stored as full IDs — look up directly in G, not by short name
        node_data = RE.G.nodes.get(caller) if RE.G else None
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
    logs = RE.log_patterns.get(target, [])
    if not logs:
        return f"No log patterns found for '{target}'."
    return "LOG PATTERNS in " + target + ":\n" + "\n".join(f"  • {p}" for p in logs[:10])


def tool_search_symbols(query: str, service: str = "", brief: bool = False) -> str:
    # RRF fusion of dense vector + BM25 — single ranked list per service
    merged = RE.unified_search([query])

    # Collect unique modules referenced by results — suggest get_module calls
    referenced_modules: dict = {}  # module -> (svc, count)

    lines = [f"SYMBOL SEARCH: '{query}'"]
    for svc in sorted(merged):
        if service and svc != service:
            continue
        svc_lines = []
        for h in merged[svc]:
            nid = h.get("id") or h.get("name", "?")
            sig = "" if brief else f" :: {h.get('type','?')[:60]}"
            svc_lines.append(f"  ID: {nid}{sig}")
            mod = h.get("module", "")
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


def tool_fast_search(query: str, top_k: int = 10) -> str:
    """Zero-GPU keyword search: BM25 + IDF graph index, RRF merged."""
    merged = RE.fast_search(query, top_k=top_k)
    if not merged:
        return f"No keyword results for '{query}'. Try search_symbols for semantic matching."
    lines = [f"FAST SEARCH (BM25+keyword): '{query}'"]
    for svc in sorted(merged):
        for h in merged[svc]:
            nid = h.get("id") or h.get("name", "?")
            score = h.get("_rrf_score", 0)
            lines.append(f"  [{svc}] {nid}  (rrf={score:.4f})")
    lines.append("\n→ Keyword match only. For semantic/conceptual queries use search_symbols.")
    return "\n".join(lines)


def tool_fast_search_reranked(query: str, top_k: int = 10) -> str:
    """BM25 top-30 → cross-encoder rerank → top-k. Falls back to fast_search if reranker absent."""
    merged = RE.fast_search_reranked(query, top_k=top_k)
    if not merged:
        return f"No results for '{query}'. Try search_symbols for semantic matching."
    reranked = RE.reranker is not None
    tag = "BM25+RERANKED" if reranked else "BM25+keyword (reranker not loaded)"
    lines = [f"FAST SEARCH ({tag}): '{query}'"]
    for svc in sorted(merged):
        for h in merged[svc]:
            nid = h.get("id") or h.get("name", "?")
            score = h.get("_rerank_score", h.get("_bm25_score", 0))
            lines.append(f"  [{svc}] {nid}  (score={score:.4f})")
    lines.append("\n→ Use get_function_body() with the full ID shown above.")
    return "\n".join(lines)


def tool_search_requirements(query: str, k: int = 5) -> str:
    """Search requirement-graph clusters for behavioral/flow queries."""
    try:
        import sys, os
        _serve = os.path.join(os.path.dirname(__file__), "serve")
        if _serve not in sys.path:
            sys.path.insert(0, _serve)
        import requirements_index as RI
        if not RI.is_ready():
            RI.initialize()
        if not RI.is_ready():
            return ("search_requirements index not built yet. "
                    "Run: python3 build/11_build_requirements.py --embed")
        clusters = RI.search_requirements(query, k=k)
        return RI.format_for_mcp(clusters)
    except Exception as e:
        return f"search_requirements error: {e}"


def tool_get_why_context(symbol_name: str) -> str:
    """WHY context: ownership, activity trend, Granger causality, anti-patterns."""
    data = RE.get_why_context(symbol_name)
    if not data["found"]:
        return (f"No context found for '{symbol_name}'. "
                "Try search_symbols first to confirm the exact module name.")
    lines = [f"WHY CONTEXT: {symbol_name}"]

    if data["summary"]:
        lines.append(f"\nSummary: {data['summary']}")

    if data["owners"]:
        lines.append("\nOwners:")
        for o in data["owners"]:
            lines.append(f"  {o['name']} <{o['email']}> — {o['commits']} commits")

    if data["activity"]:
        act = data["activity"]
        lines.append(
            f"\nActivity: score={act['score']}  trend={act['trend']}"
            f"  (recent-50={act['recent_50']}, recent-200={act['recent_200']})"
        )

    if data["criticality"] and data["criticality"].get("score", 0) > 0:
        crit = data["criticality"]
        lines.append(f"\nCriticality: {crit['score']:.3f} (rank #{crit.get('rank', '?')})")
        for r in crit.get("reasons", []):
            lines.append(f"  - {r}")

    if data["causal_outputs"]:
        lines.append(f"\nCausal outputs — changes here predict changes in:")
        for c in data["causal_outputs"]:
            lines.append(f"  → {c['target']}  lag={c['lag_days']}d  p={c['p_value']}  [{c['strength']}]")

    if data["causal_inputs"]:
        lines.append(f"\nCausal inputs — changes here are predicted by:")
        for c in data["causal_inputs"]:
            lines.append(f"  ← {c['source']}  lag={c['lag_days']}d  p={c['p_value']}  [{c['strength']}]")

    if data["anti_patterns"]:
        lines.append("\nAnti-patterns detected:")
        for ap in data["anti_patterns"]:
            lines.append(f"  ⚠  {ap}")

    return "\n".join(lines)


def tool_search_modules(query: str, service: str = "") -> str:
    """
    Search module namespaces by keyword. Returns matching module paths that can
    be passed directly to get_module(). Much faster than search_symbols when you
    know the component name (gateway, feature, domain) but not the exact function.
    """
    if RE.G is None:
        return "Graph not loaded."

    words = [w.lower() for w in re.split(r"\W+", query) if len(w) >= 3]
    if not words:
        return "Query too short."

    # Collect all unique modules and score by how many query words they contain
    mod_scores: dict = {}  # module -> (svc, score, symbol_count)
    for nid, d in RE.G.nodes(data=True):
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
    from collections import defaultdict
    if RE.G is None:
        return "Graph not loaded."

    q = module_name.lower().replace("::", ".").replace("/", ".")
    matches: list = []
    for nid, d in RE.G.nodes(data=True):
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
    for k in RE.body_store:
        k_mod = ".".join(k.split(".")[:-1]).lower()
        if q in k_mod:
            if not any(nid == k for nid, _ in matches):
                body_hits.append(k)

    if not matches and not body_hits:
        # Try fuzzy: any part of the module name
        parts = q.split(".")
        for nid, d in RE.G.nodes(data=True):
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
            if nid in RE.body_store:
                entry += "  [body available]"
            lines.append(entry)

    if body_hits:
        lines.append(f"\n  [body_store only — not in graph]")
        for k in body_hits[:10]:
            lines.append(f"    {k}  [body available]")

    lines.append(f"\n→ Call get_function_body() on any ID above to read its implementation.")
    return "\n".join(lines)


def tool_search_docs(query: str, tags: list = None) -> str:
    if not RE.doc_chunks:
        return "Documentation not loaded. Run 07_chunk_docs.py."
    results = []
    if RE.doc_lance_tbl is not None and RE.can_embed():
        try:
            qvec = RE._encode_query(query)
            hits = RE.doc_lance_tbl.search(qvec).limit(10).to_list()
            if tags:
                hits = [h for h in hits if any(t in h.get("tags","") for t in tags)] or hits
            results = hits[:5]
        except Exception as e:
            print(f"[search_docs] {e}")
    if not results:
        words  = re.split(r'\W+', query.lower())
        scored = []
        for chunk in RE.doc_chunks:
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
    if not RE.gw_integrity:
        return ("Gateway integrity config not built yet.\n"
                "Run the gateway scan to build gateway_integrity_config.json.")
    gw_lower = gateway_name.lower()
    matched = next((k for k in RE.gw_integrity if k.lower() == gw_lower or gw_lower in k.lower()), None)
    if not matched:
        return (f"Gateway '{gateway_name}' not found.\nAvailable: {', '.join(list(RE.gw_integrity)[:20])}")
    lines = [f"GATEWAY INTEGRITY: {matched}"]
    for k, v in RE.gw_integrity[matched].items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def tool_get_type_definition(type_name: str, service: str = "") -> str:
    if RE.G is None:
        return "Graph not loaded."
    matches = [(nid, d) for nid, d in RE.G.nodes(data=True)
               if d.get("name") == type_name and d.get("kind") in ("type","class")
               and (not service or d.get("service") == service)]
    if not matches:
        matches = [(nid, d) for nid, d in RE.G.nodes(data=True)
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
        body = RE.body_store.get(nid,"") or _fuzzy_fn_lookup(nid)[1]
        if body:
            lines.append(f"  Body:\n  {body[:600]}")
    return "\n".join(lines)


def tool_get_context(query: str) -> str:
    """Last-resort: build full cross-service context, split into 3 chunks. Returns chunk 1/3."""
    if RE.G is None:
        return "Graph not loaded."
    vec_by_svc     = RE.stratified_vector_search(query)
    kw_by_svc      = RE.cross_service_keyword_search(query)
    cluster_by_svc = RE.get_cluster_context_for_services(
        list(set(list(vec_by_svc) + list(kw_by_svc)))
    )
    doc_hits = RE.doc_vector_search(RE._encode_query(query), top_k=3) if RE.doc_lance_tbl is not None else []
    full = _build_base_context(vec_by_svc, kw_by_svc, cluster_by_svc, "default", doc_hits)

    # Split into 3 equal chunks by character count
    n = len(full)
    size = max(n // 3, 1)
    chunks = [full[:size], full[size:2*size], full[2*size:]]
    chunks = [c for c in chunks if c.strip()]  # drop empty tail chunks

    token = hashlib.md5(query.encode()).hexdigest()[:10]
    with _CTX_LOCK:
        if len(_CTX_CACHE) >= _MAX_CTX_CACHE:
            del _CTX_CACHE[next(iter(_CTX_CACHE))]  # evict oldest
        _CTX_CACHE[token] = chunks

    total = len(chunks)
    suffix = (f"\n\n[Part 1/{total} — if this is insufficient, call "
              f"get_context_continue with token='{token}' and part=2]") if total > 1 else ""
    return f"[Context part 1/{total}]\n\n{chunks[0]}{suffix}"


def tool_get_context_continue(token: str, part: int) -> str:
    """Retrieve the next chunk of a get_context result. Use only if part 1 was insufficient."""
    chunks = _CTX_CACHE.get(token)
    if not chunks:
        return f"Token '{token}' not found or expired — call get_context again."
    if part < 2 or part > len(chunks):
        return f"Invalid part {part}. Valid range: 2–{len(chunks)}."
    chunk = chunks[part - 1]
    total = len(chunks)
    is_last = (part == total)
    suffix = ("\n\n[End of context]" if is_last else
              f"\n\n[Call get_context_continue(token='{token}', part={part+1}) for more]")
    return f"[Context part {part}/{total}]\n\n{chunk}{suffix}"


# ════════════════════════════════════════════════════════════════════════════
# TOOL DISPATCH
# ════════════════════════════════════════════════════════════════════════════

def _tool_guardian(mode: str, changed_files: list, min_confidence: float = 0.1) -> str:
    """Unified guardian tool handler — resolves files to modules, then dispatches."""
    import json as _json
    resolved = RE.resolve_files_to_modules(changed_files)
    seed_mods = []
    for fp, mods in resolved.items():
        seed_mods.extend(mods if mods else [fp])
    if not seed_mods:
        return "Could not resolve any modules from the given files."

    if mode == "predict":
        result = RE.predict_missing_changes(seed_mods)
        preds = result.get("predictions", []) if isinstance(result, dict) else result
        if not preds:
            return "No missing changes predicted — your changeset looks complete."
        lines = ["**Predicted Missing Changes:**\n"]
        for item in preds[:15]:
            conf = f"{item.get('confidence', 0):.0%}"
            lines.append(f"- `{item['module']}` (confidence: {conf}, weight: {item.get('weight', 0)})")
            if item.get("causal_note"):
                lines.append(f"  _{item['causal_note']}_")
        return "\n".join(lines)

    elif mode == "check":
        blast = RE.get_blast_radius(seed_mods)
        missing_result = RE.predict_missing_changes(seed_mods)
        missing = missing_result.get("predictions", []) if isinstance(missing_result, dict) else missing_result
        risk = RE.score_change_risk(seed_mods)
        reviewers = RE.suggest_reviewers(seed_mods, top_k=3)

        lines = [f"## Guardian Report\n"]
        status = "PASS" if not missing else ("WARN" if len(missing) < 3 else "FAIL")
        lines.append(f"**Verdict: {status}**\n")
        lines.append(f"**Risk Score: {risk.get('risk_score', 0)}/100 ({risk.get('risk_level', 'LOW')})**\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Changed modules | {len(seed_mods)} |")
        lines.append(f"| Import neighbors | {len(blast.get('import_neighbors', []))} |")
        lines.append(f"| Co-change neighbors | {len(blast.get('cochange_neighbors', []))} |")
        lines.append(f"| Affected services | {len(blast.get('affected_services', []))} |")
        if missing:
            lines.append(f"\n### Likely Missing Files")
            for m in missing[:5]:
                lines.append(f"- `{m['module']}` (confidence: {m.get('confidence',0):.0%})")
        if reviewers.get("reviewers"):
            lines.append(f"\n### Suggested Reviewers")
            for r in reviewers["reviewers"][:3]:
                lines.append(f"- **{r['name']}** ({r['commits']} commits)")
        return "\n".join(lines)

    elif mode == "reviewers":
        result = RE.suggest_reviewers(seed_mods, top_k=5)
        if not result.get("reviewers"):
            return "No reviewer data available for these modules."
        lines = ["**Suggested Reviewers:**\n"]
        for r in result["reviewers"]:
            mods_str = ", ".join(f"`{m}`" for m in r.get("modules", [])[:3])
            lines.append(f"- **{r['name']}** ({r['email']}): {r['commits']} commits — {mods_str}")
        return "\n".join(lines)

    elif mode == "risk":
        result = RE.score_change_risk(seed_mods)
        lines = [f"**Risk Score: {result.get('risk_score', 0)}/100 ({result.get('risk_level', 'LOW')})**\n"]
        components = result.get("components", {})
        for name, comp in components.items():
            lines.append(f"- **{name}**: {comp.get('score', 0)}/100 — {comp.get('detail', '')}")
        return "\n".join(lines)

    return "Unknown guardian mode."


def _delegate_to_mcp(tool_name: str, **kwargs) -> str:
    """Call an MCP server tool function by name and return its formatted output.

    Keeps the guardrails/criticality tools exposed from one place — the MCP
    server module — so tools.py cannot silently drift from mcp_server.py.
    Required for the App-Core Sync Invariant (OPERATING_PRINCIPLES.md).
    """
    try:
        from serve import mcp_server as _mcp  # noqa: F401
    except Exception as e:
        return f"[{tool_name}] MCP server module not importable: {e!r}"
    fn = getattr(_mcp, tool_name, None)
    # FastMCP may wrap the decorated fn; if so, unwrap
    if fn is not None and not callable(fn):
        fn = getattr(fn, "fn", None) or getattr(fn, "func", None) or getattr(fn, "__wrapped__", None)
    if not callable(fn):
        return f"[{tool_name}] not exposed by serve.mcp_server"
    try:
        return str(fn(**kwargs))
    except Exception as e:
        return f"[{tool_name}] call failed: {e!r}"


TOOL_DISPATCH: dict = {
    "get_function_body":     lambda a: tool_get_function_body(a.get("fn_id",""), a.get("reason","")),
    "trace_callees":         lambda a: tool_trace_callees(a.get("fn_id",""), a.get("reason","")),
    "trace_callers":         lambda a: tool_trace_callers(a.get("fn_id",""), a.get("reason","")),
    "get_log_patterns":      lambda a: tool_get_log_patterns(a.get("fn_id","")),
    "fast_search":           lambda a: tool_fast_search(a.get("query",""), int(a.get("top_k", 10))),
    "fast_search_reranked":  lambda a: tool_fast_search_reranked(a.get("query",""), int(a.get("top_k", 10))),
    "search_requirements":   lambda a: tool_search_requirements(a.get("query",""), int(a.get("k", 5))),
    "get_why_context":       lambda a: tool_get_why_context(a.get("symbol_name","")),
    "search_symbols":        lambda a: tool_search_symbols(a.get("query",""), a.get("service",""), a.get("brief", False)),
    "search_modules":        lambda a: tool_search_modules(a.get("query",""), a.get("service","")),
    "get_module":            lambda a: tool_get_module(a.get("module_name",""), a.get("service",""), a.get("max_symbols", 30)),
    "get_blast_radius":      lambda a: str(RE.get_blast_radius(a.get("files_or_modules", []))),
    "get_context":           lambda a: tool_get_context(a.get("query","")),
    "get_context_continue":  lambda a: tool_get_context_continue(a.get("token",""), int(a.get("part", 2))),
    "search_docs":           lambda a: tool_search_docs(a.get("query",""), a.get("tags",[])),
    "get_gateway_integrity": lambda a: tool_get_gateway_integrity(a.get("gateway_name","")),
    "get_type_definition":   lambda a: tool_get_type_definition(a.get("type_name",""), a.get("service","")),
    # ── Guardian / PR analysis tools ──────────────────────────────────────────
    "predict_missing_changes": lambda a: _tool_guardian("predict", a.get("changed_files", []), a.get("min_confidence", 0.1)),
    "suggest_reviewers":       lambda a: _tool_guardian("reviewers", a.get("changed_files", [])),
    "score_change_risk":       lambda a: _tool_guardian("risk", a.get("changed_files", [])),
    # ── Guardrails / Criticality tools (delegated to MCP server implementations
    #    so tools.py and mcp_server.py stay in sync; see OPERATING_PRINCIPLES.md
    #    App-Core Sync Invariant) ────────────────────────────────────────────────
    "check_my_changes":      lambda a: _delegate_to_mcp("check_my_changes",      changed_files=a.get("changed_files", [])),
    "check_criticality":     lambda a: _delegate_to_mcp("check_criticality",     modules=a.get("modules", [])),
    "get_guardrails":        lambda a: _delegate_to_mcp("get_guardrails",        modules=a.get("modules", [])),
    "list_critical_modules": lambda a: _delegate_to_mcp("list_critical_modules",
                                                        service=a.get("service"),
                                                        threshold=a.get("threshold", 0.5),
                                                        top_k=a.get("top_k", 20)),
    # ── HyperCode coding tools (available when _CODING_TOOLS_AVAILABLE) ───────
    "run_bash":   lambda a: (
        _run_bash(a.get("command",""), a.get("timeout"), None)
        if _CODING_TOOLS_AVAILABLE else "run_bash not available (apps/cli/tools not found)"),
    "read_file":  lambda a: (
        _read_file(a.get("file_path",""), a.get("offset",1), a.get("limit"))
        if _CODING_TOOLS_AVAILABLE else "read_file not available"),
    "write_file": lambda a: (
        _write_file(a.get("file_path",""), a.get("content",""))
        if _CODING_TOOLS_AVAILABLE else "write_file not available"),
    "edit_file":  lambda a: (
        _edit_file(a.get("file_path",""), a.get("old_string",""),
                   a.get("new_string",""), a.get("replace_all", False))
        if _CODING_TOOLS_AVAILABLE else "edit_file not available"),
    "glob_files": lambda a: (
        _glob_files(a.get("pattern",""), a.get("path"))
        if _CODING_TOOLS_AVAILABLE else "glob_files not available"),
    "grep_files": lambda a: (
        _grep_files(a.get("pattern",""), a.get("path"), a.get("file_glob"),
                    a.get("case_insensitive", False), a.get("context_lines", 0))
        if _CODING_TOOLS_AVAILABLE else "grep_files not available"),
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
    if include_body and nid in RE.body_store:
        preview = "\n".join(RE.body_store[nid].split("\n")[:3])
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
    from collections import defaultdict
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
    mg_expanded = RE.module_graph_expand(seed_mods, depth=2)
    mg_block = ""
    if mg_expanded:
        by_svc_hop: dict = defaultdict(list)
        for mod, info in mg_expanded.items():
            by_svc_hop[info["service"]].append((info["hop"], info["direction"], mod))
        mg_lines = []
        for svc in sorted(by_svc_hop):
            for hop, direction, mod in sorted(by_svc_hop[svc])[:5]:
                sample = [
                    RE.G.nodes[nid].get("name","?")
                    for nid in RE.file_to_nodes.get(mod,[])[:2]
                    if RE.G and nid in RE.G and RE.G.nodes[nid].get("kind") != "phantom"
                ]
                mg_lines.append(
                    f"  [{svc}] hop={hop} ({direction}): {mod}"
                    + (f"  [{', '.join(sample)}]" if sample else "")
                )
        mg_block = "IMPORT GRAPH EXPANSION:\n" + "\n".join(mg_lines[:25])

    cc_path = RE.cochange_path_traverse(seed_mods)
    cc_block = ""
    if cc_path:
        cc_block = "EVOLUTIONARY COUPLING (co-changed in git):\n" + "\n".join(
            f"  hop={c['hop']} weight={c['weight']:3d}: {c['module']}" for c in cc_path[:15]
        )

    kw_words = [
        w.lower() for w in re.split(r"\W+", " ".join(v for v in [framework[:50]] if v))
        if len(w) >= 4 and w.lower() not in RE._KW_STOPWORDS
    ]
    entries = RE.get_entry_points(kw_words) if kw_words else []
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
    if RE._llm_client is None:
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
        resp = RE._llm_client.chat.completions.create(
            model=RE.LLM_MODEL, messages=messages, tools=AGENT_TOOLS,
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
    resp = RE._llm_client.chat.completions.create(
        model=RE.LLM_MODEL, messages=answer_msgs, temperature=0.2, max_tokens=8000,
    )
    return resp.choices[0].message.content or ""
