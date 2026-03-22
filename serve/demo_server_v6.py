"""
HyperRetrieval — Agentic codebase intelligence (Chainlit UI)

Clean ReAct architecture:
  - Single system prompt (no routing, no persona selection)
  - LLM starts with the user query and decides what to look up
  - Tool calls rendered as Chainlit steps
  - Final answer streamed when LLM stops calling tools

No pre-retrieval, no fast_route, no context pre-loading.
"""
import asyncio, json, os, pathlib, sys, time
import chainlit as cl

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import retrieval_engine as RE

# ── Config ────────────────────────────────────────────────────────────────────
_HERE          = pathlib.Path(__file__).resolve().parent
ARTIFACT_DIR   = pathlib.Path(os.environ.get(
    "ARTIFACT_DIR",
    str(_HERE / "demo_artifact" if (_HERE / "demo_artifact").exists() else _HERE)
))
LLM_API_KEY    = os.environ.get("LLM_API_KEY",  "")
LLM_BASE_URL   = os.environ.get("LLM_BASE_URL", "")
LLM_MODEL      = os.environ.get("LLM_MODEL",    "reasoning-large-model")
MAX_HISTORY    = 6
MAX_TOOL_CALLS = 12
RETRIEVAL_LOG  = pathlib.Path("/tmp/retrieval_log.jsonl")

llm_client       = None
async_llm_client = None
_load_all_done   = False


# ════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT  — tells the LLM who it is, what it can do, and how to think
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are Codebase Expert, an expert AI assistant with deep knowledge of your organisation's \
payment platform codebase. You help engineers understand, debug, and extend \
the system by reading actual source code — not guessing.

## Codebase

94,244 indexed symbols across 12 services (Haskell primary, some Rust/JS):

| Service | Symbols | Role |
|---|---|---|
| euler-api-gateway | 39,806 | HTTP entry point, routing, auth, gateway connectors |
| euler-api-txns | 30,673 | Transaction lifecycle, payment flows, mandate, EMI, tokenization |
| UCS | 7,787 | Universal Connector Service — third-party gateway integration |
| euler-db | 5,610 | Database layer, OLTP models |
| euler-api-order | 3,652 | Order management |
| graphh | 2,377 | Graph / analytics |
| euler-api-pre-txn | 2,364 | Pre-transaction validation |
| euler-api-customer | 1,231 | Customer profile |
| basilisk-v3, euler-drainer, token_issuer_portal_backend, haskell-sequelize | <400 each | Specialised |

Payment flow direction: euler-api-gateway → euler-api-txns → UCS → euler-db

## Tools

Use tools iteratively. Each result tells you what to look up next.

**search_modules(query)**
Find module namespaces by keyword. Use this FIRST when you don't know where \
code lives. Returns module names you can pass to get_module.
→ "UPI collect", "gateway routes", "mandate payment"

**get_module(module_name)**
List every symbol in a module namespace. Use after search_modules to see the \
full surface area before deciding what to read.
→ "Euler.API.Gateway.Gateway.UPI", "PaymentFlows"

**search_symbols(query)**
Semantic + keyword search across all 94k symbols. Use when you know what a \
function does but not its exact name. Returns fully-qualified IDs.
→ "card tokenization flow", "emandate debit execution"

**get_function_body(fn_id)**
Read actual source code of a function by its fully-qualified ID. This is your \
primary way to understand implementation. Batch multiple calls when independent.
→ "PaymentFlows.getAllPaymentFlowsForTxn"

**trace_callers(fn_id)**
List all functions that call this one (upstream). Use to find entry points or \
assess who is affected by a change.

**trace_callees(fn_id)**
List all functions this one calls (downstream). Use to trace a flow forward \
through the system.

**get_blast_radius(files_or_modules)**
Import graph + co-change analysis. Use for impact assessment when something changes.

**get_context(query)**
Last resort — returns a large pre-built context block (~5k-18k tokens). \
Only use if search_symbols + get_function_body have failed to find what you need. \
Never call twice in one session.

## How to reason

1. **Orient** — use search_modules to find where the relevant code lives
2. **Read** — get_module to see the namespace, then get_function_body for key functions
3. **Trace** — follow callers or callees to understand the full flow
4. **Synthesize** — answer from what you have actually read

Rules:
- Never guess implementation details — always read the code first
- When comparing code across services, search each service separately
- Batch independent tool calls in a single response turn
- If a search returns unexpected results, refine the query rather than repeating it
- Fully-qualified IDs use dot notation: Module.SubModule.functionName (no slashes, no extensions)
"""


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_all():
    global llm_client, async_llm_client, _load_all_done
    if _load_all_done:
        return
    from openai import OpenAI, AsyncOpenAI
    RE.initialize(ARTIFACT_DIR)
    llm_client       = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    async_llm_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    _load_all_done   = True


# ════════════════════════════════════════════════════════════════════════════
# TOOL STEP RENDERING
# ════════════════════════════════════════════════════════════════════════════

_TOOL_LABELS = {
    "get_function_body":  ("📖", "Reading"),
    "trace_callers":      ("⬆",  "Callers of"),
    "trace_callees":      ("⬇",  "Calls from"),
    "search_symbols":     ("🔍", "Searching"),
    "search_modules":     ("📦", "Modules"),
    "get_module":         ("📂", "Module"),
    "get_blast_radius":   ("💥", "Blast radius"),
    "get_log_patterns":   ("📋", "Log patterns"),
    "get_context":        ("📚", "Context"),
}

def _step_name(fn_name: str, args: dict) -> str:
    icon, verb = _TOOL_LABELS.get(fn_name, ("⚙", fn_name))
    subject = (args.get("fn_id") or args.get("query") or
               args.get("module_name") or args.get("files_or_modules") or "")
    parts = [p for p in str(subject).replace("/", ".").split(".")
             if p and p not in ("hs", "py", "rs", "js", "ts")]
    short = parts[-1] if parts else str(subject)
    return f"{icon} {verb} · {short}" if short else f"{icon} {verb}"


# ════════════════════════════════════════════════════════════════════════════
# CHAINLIT HANDLERS
# ════════════════════════════════════════════════════════════════════════════

@cl.set_chat_profiles
async def set_chat_profiles():
    return [
        cl.ChatProfile(
            name="HyperRetrieval",
            markdown_description="Codebase intelligence — ask anything about your codebase.",
            starters=[
                cl.Starter(label="Card payment flow end-to-end",
                           message="Walk me through the complete card payment flow from order creation to gateway response."),
                cl.Starter(label="How does UPI Collect work?",
                           message="Explain the UPI Collect flow — which services are involved and what happens at each step?"),
                cl.Starter(label="Gateways across services",
                           message="What is the difference between gateways written in api-txns, api-gateway and UCS?"),
                cl.Starter(label="What does UCS do?",
                           message="What is UCS and how does it relate to euler-api-gateway? When is each used?"),
                cl.Starter(label="Payment stuck in PENDING",
                           message="What are the failure scenarios that can leave a payment permanently stuck in PENDING state?"),
                cl.Starter(label="Where is card data handled?",
                           message="Where is cardholder data (PAN, CVV) handled in the codebase? List every location where it is stored, logged, or transmitted."),
            ],
        ),
    ]


@cl.on_chat_start
async def on_start():
    await asyncio.to_thread(load_all)
    cl.user_session.set("history", [])
    await cl.Message(
        content="**HyperRetrieval** ready. Ask anything about your codebase, or type `/status` for system stats."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()

    # /status command
    if query.lower().startswith("/status"):
        lines = [
            "**HyperRetrieval Status**",
            f"- Graph: {RE.G.number_of_nodes():,} nodes, {RE.G.number_of_edges():,} edges",
            f"- Vectors: {len(RE.lance_tbl):,} @ 4096d",
            f"- Module graph: {RE.MG.number_of_nodes():,} modules, {RE.MG.number_of_edges():,} edges",
            f"- Co-change: {len(RE.cochange_index):,} modules indexed",
            f"- Body store: {len(RE.body_store):,} bodies",
            f"- Call graph: {len(RE.call_graph):,} entries",
            f"- Log patterns: {len(RE.log_patterns):,} functions",
            f"- Doc chunks: {len(RE.doc_chunks):,}",
            f"- LLM: {LLM_MODEL}",
        ]
        await cl.Message(content="\n".join(lines)).send()
        return

    history = cl.user_session.get("history", [])

    try:
        # ── Build initial messages: system first, then history, then user query ─
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for past_q, past_a in history[-MAX_HISTORY:]:
            messages.append({"role": "user",      "content": past_q})
            messages.append({"role": "assistant", "content": past_a})
        messages.append({"role": "user", "content": query})

        # ── ReAct loop: LLM reasons and calls tools until it has enough info ──
        tool_calls_count = 0
        tool_log = []

        while tool_calls_count < MAX_TOOL_CALLS:
            resp = await asyncio.to_thread(
                llm_client.chat.completions.create,
                model=LLM_MODEL,
                messages=messages,
                tools=RE.AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.15,
                max_tokens=4000,
            )

            assistant_msg = resp.choices[0].message

            # No tool calls → LLM has finished reasoning, stream final answer
            if not assistant_msg.tool_calls:
                break

            # Append assistant's reasoning + tool decisions to message history
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in assistant_msg.tool_calls
                ],
            })

            # Execute each tool call, render as a Chainlit step
            tool_results = []
            for tc in assistant_msg.tool_calls:
                tool_calls_count += 1
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}

                async with cl.Step(name=_step_name(fn_name, args), type="tool",
                                   show_input=False) as step:
                    dispatcher = RE.TOOL_DISPATCH.get(fn_name)
                    result = await asyncio.to_thread(dispatcher, args) if dispatcher \
                             else f"Unknown tool: {fn_name}"
                    lines = [l.strip() for l in result.splitlines()
                             if l.strip() and not l.startswith(("#", "=", "-"))]
                    step.output = lines[0][:160] if lines else result[:160]

                tool_log.append({"tool": fn_name, "args": args})
                tool_results.append({
                    "role": "tool", "tool_call_id": tc.id, "content": result,
                })

            messages.extend(tool_results)

            if tool_calls_count >= MAX_TOOL_CALLS:
                messages.append({
                    "role": "user",
                    "content": "You have reached the investigation limit. Synthesize your findings into a complete answer now.",
                })
                break

        # ── Stream final answer ────────────────────────────────────────────────
        stream = await async_llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=32000,
            stream=True,
        )

        response_msg = cl.Message(content="")
        await response_msg.send()
        full_response = ""

        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                await response_msg.stream_token(token)

        await response_msg.update()

        # ── Update history ─────────────────────────────────────────────────────
        history.append((query, full_response[:3000]))
        cl.user_session.set("history", history[-MAX_HISTORY:])

        # ── Log for offline analysis ───────────────────────────────────────────
        try:
            with open(RETRIEVAL_LOG, "a") as lf:
                lf.write(json.dumps({
                    "ts": time.time(), "query": query, "tool_calls": tool_log,
                }) + "\n")
        except Exception:
            pass

    except Exception as e:
        import traceback
        await cl.Message(
            content=f"**Error:** {e}\n```\n{traceback.format_exc()[:1000]}\n```"
        ).send()
