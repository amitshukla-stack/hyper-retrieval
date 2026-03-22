# HyperRetrieval

A self-hosted codebase intelligence platform. Point it at your source code, run a 7-stage build pipeline, and get a structured knowledge graph of your entire codebase — queryable by humans and AI agents alike.

The **Chat UI** and **MCP server** are two reference implementations on top of that knowledge graph. The real value is the data layer: once your codebase is indexed, you can build any number of applications on top of it.

---

## What it does

Indexes your codebase into five complementary data structures:

| Index | What it stores | Used for |
|-------|---------------|----------|
| **Symbol graph** | Functions, types, modules + import edges | Navigation, blast-radius analysis |
| **Vector index** | Semantic embedding of every symbol | Natural-language search |
| **Body store** | Full source text per function | Code reading, LLM context |
| **Call graph** | Caller/callee relationships | Flow tracing, impact analysis |
| **Co-change index** | Files that historically change together | Risk-aware PR review |

Once indexed, the same data powers multiple entry points — all shipped in this repo:

- **Chat UI** (Chainlit) — conversational interface for engineers to explore the codebase
- **MCP server** — exposes tools directly inside AI coding assistants (Claude Code, Cursor, Windsurf)
- **PR analyser** — CLI blast-radius report for CI/CD pipelines

---

## What data it works best on

- **Large, multi-service codebases** — 10k–200k symbols across 5–20 services is the sweet spot
- **Codebases with meaningful git history** — co-change analysis requires commit history to be useful
- **Mixed-language repos** — works well when services have clear boundaries and consistent module naming
- **Codebases with internal documentation** — markdown docs are embedded alongside code and searchable together

**Less effective for:**
- Single-file or tiny projects (< 500 symbols) — the indexing overhead is not worthwhile
- Repos with little or no git history
- Codebases where the dominant language is not yet supported (see Language support below)

---

## Language support

| Language       | Symbols | Bodies | Call graph | Log patterns |
|----------------|---------|--------|------------|--------------|
| Haskell        | ✓       | ✓      | ✓ (approx) | ✓            |
| Rust           | ✓       | ✓      | ✓          | ✓            |
| JavaScript/TypeScript | ✓ | ✓   | ✓ (AST)    | —            |
| Python         | ✓       | ✓      | ✓ (AST)    | —            |
| Groovy         | ✓       | ✓      | —          | ✓            |

Adding a new language means implementing `parse_<lang>_file()` in `build/01_extract.py` — see [Adding a language parser](#adding-a-language-parser).

---

## Folder structure

```
hyperretrieval/
├── build/                     ← 8-stage pipeline (run once to build indexes)
│   ├── 01_extract.py          ← Parse source → symbols, bodies, call graph, log patterns
│   │                             Tree-sitter for Haskell/Rust/JS/TS/Groovy; ast for Python
│   │                             Parallel across services (multiprocessing.Pool)
│   ├── 02_build_graph.py      ← Build NetworkX graph, Leiden clustering at module level
│   ├── 03_embed.py            ← GPU-batch embed all symbols → LanceDB (vectors.lance)
│   ├── 04_summarize.py        ← LLM-summarize each cluster → human-readable descriptions
│   │                             Adaptive sampling: scales with cluster size, business-logic
│   │                             modules (Product, OLTP, Flow, Handler) sampled first
│   ├── 05_package.py          ← Copy final artifacts into workspace/artifacts/
│   ├── 06_build_cochange.py   ← Parse git history → co-change index (streaming, O(1) memory)
│   ├── 07_chunk_docs.py       ← Chunk + embed markdown docs → docs.lance via embed server
│   │                             Reads docs/ and docs/generated/ from workspace
│   ├── 08_generate_arch_docs.py ← [BETA] Auto-generate architecture docs via LLM
│   │                             Generate → Verify → Correct loop, saves to docs/generated/
│   │                             See file header for known issues and improvement TODOs
│   └── run_pipeline.sh        ← Run all stages end-to-end
│                                 Flags: --from-stage N, --only-stage N
│                                 Auto-cleans stale outputs before each run
│                                 Stage order: 1→2→3→4→5→6→8→7 (8 before 7 so generated
│                                 docs are embedded by stage 7)
│
├── serve/                     ← Core engine — all retrieval logic lives here
│   ├── retrieval_engine.py    ← Core library: loads all indexes, graph traversal,
│   │                             BM25 + vector RRF fusion search, co-change queries
│   ├── embed_server.py        ← Shared embedding server (port 8001) — start FIRST
│   ├── mcp_server.py          ← MCP SSE server (port 8002) — 8 tools for AI agents
│   ├── pr_analyzer.py         ← CLI blast-radius report for CI/CD pipelines
│   ├── public/                ← Chainlit CSS + theme
│   └── .chainlit/             ← Chainlit config (name, layout, CSS path)
│
├── tools.py                   ← Org-customizable tool implementations (gitignored)
│                                 AGENT_TOOLS, TOOL_DISPATCH, persona prompts
│                                 Uses RE.unified_search() for BM25+vector RRF fusion
│
├── tools/
│   └── generate_mindmap.py    ← Visualise the graph as an HTML mindmap (WebGL)
│
├── tests/
│   ├── test_01_artifacts.py   ← Verify build outputs exist and are non-empty
│   ├── test_02_retrieval_logic.py ← Unit tests for retrieval functions
│   ├── test_03_canary.py      ← Smoke test: does a basic query return results?
│   ├── test_04_retrieval_accuracy.py ← Benchmark: known-answer queries
│   ├── test_05_integration.py ← End-to-end server + query test
│   ├── test_06_auto_eval.py   ← LLM-as-judge retrieval quality eval
│   └── run_all.sh
│
└── config.example.yaml        ← Template workspace config (copy to your workspace)
```

```
workspaces/YOUR_ORG/           ← Org-specific data (not in git)
├── config.yaml
├── source/                    ← Your source repos (one subdirectory per service)
├── artifacts/                 ← Indexes loaded at runtime
│   ├── graph_with_summaries.json
│   ├── vectors.lance/
│   └── cochange_index.json
├── output/                    ← Intermediate build outputs
│   ├── body_store.json
│   ├── call_graph.json
│   ├── log_patterns.json
│   └── docs.lance/
├── docs/                      ← Markdown documentation (embedded in stage 7)
└── git_history.json

models/
└── <embedding-model>/         ← Model weights (local provider only)
```

---

## Setup from scratch

### Prerequisites

```bash
python3 --version   # 3.11+

pip install chainlit openai lancedb sentence-transformers networkx \
            pyarrow leidenalg igraph rank-bm25 mcp ijson pyyaml \
            tree-sitter tree-sitter-haskell tree-sitter-rust tree-sitter-groovy

# GPU needed only for local embedding (stage 3 + embed_server with EMBED_PROVIDER=local)
# Cloud providers (openai, voyage, cohere, etc.) require no GPU — see Embedding providers
nvidia-smi
```

### Step 1 — Prepare your workspace

```bash
mkdir -p ~/projects/workspaces/YOUR_ORG/{source,artifacts,output}

# One subdirectory per service
cp -r /path/to/service-a ~/projects/workspaces/YOUR_ORG/source/
cp -r /path/to/service-b ~/projects/workspaces/YOUR_ORG/source/

cp ~/projects/hyperretrieval/config.example.yaml \
   ~/projects/workspaces/YOUR_ORG/config.yaml
# Edit: set LLM endpoint, API keys, ports
```

### Step 2 — Export git history

```bash
# Run for each service repo, append to a single file
git -C ~/projects/workspaces/YOUR_ORG/source/service-a \
    log --all --name-only --format="COMMIT|%H|%s|%ae|%ai" \
    >> ~/projects/workspaces/YOUR_ORG/git_history.json
```

### Step 3 — Choose an embedding provider

See [Embedding providers](#embedding-providers) below. If using a cloud provider, skip the model download. If using local:

```bash
mkdir -p ~/projects/models
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('Qwen/Qwen3-Embedding-8B')
m.save('/path/to/models/qwen3-embed-8b')
"
```

### Step 4 — Run the build pipeline

```bash
cd ~/projects/hyperretrieval

export REPO_ROOT=~/projects/workspaces/YOUR_ORG/source
export OUTPUT_DIR=~/projects/workspaces/YOUR_ORG/output
export ARTIFACT_DIR=~/projects/workspaces/YOUR_ORG/artifacts
export EMBED_MODEL=/path/to/models/your-embed-model  # or set EMBED_PROVIDER + API key
export LLM_API_KEY=your_llm_api_key
export LLM_BASE_URL=https://your-llm-endpoint
export LLM_MODEL=your-model-name

bash build/run_pipeline.sh   # 30 min – 2 h depending on codebase size

# Resume from a specific stage (stages before N are skipped, outputs preserved)
bash build/run_pipeline.sh --from-stage 4

# Run a single stage only
bash build/run_pipeline.sh --only-stage 7

# Or stage by stage:
python3 build/01_extract.py        # 5–15 min for 100k symbols (parallel across services)
python3 build/02_build_graph.py    # ~2 min (Leiden clustering)
python3 build/03_embed.py          # 20–60 min (GPU-heavy, or fast with cloud provider)
python3 build/04_summarize.py      # ~30 min (LLM API, crash-safe/resumable)
python3 build/05_package.py        # ~1 min
python3 build/06_build_cochange.py # 10–30 min (streaming, safe on large history)
python3 build/07_chunk_docs.py     # ~5 min
python3 build/08_generate_arch_docs.py  # 10–30 min (LLM, resumable)
```

### Step 5 — Start the servers

**Always start the embedding server first.** Only one process should load the embedding model at a time.

```bash
# 1. Embedding server
export EMBED_MODEL=/path/to/model   # or EMBED_PROVIDER + API key
cd ~/projects/hyperretrieval/serve
python3 embed_server.py
# Wait for: [embed_server] Ready on 127.0.0.1:8001

# 2. Chat UI  (demo_server_v6.py is org-specific — gitignored, copy from template)
export EMBED_SERVER_URL=http://localhost:8001
export ARTIFACT_DIR=~/projects/workspaces/YOUR_ORG/artifacts
cd ~/projects/hyperretrieval/serve
chainlit run demo_server_v6.py --port 8000
# Open http://localhost:8000

# 3. MCP server (optional — for AI coding assistant integration)
export EMBED_SERVER_URL=http://localhost:8001
export ARTIFACT_DIR=~/projects/workspaces/YOUR_ORG/artifacts
cd ~/projects/hyperretrieval
python3 serve/mcp_server.py
# SSE endpoint: http://localhost:8002/sse
```

---

## Embedding providers

`embed_server.py` provides a unified HTTP interface regardless of where embeddings come from. Switch providers with a single env var — nothing else changes.

```bash
# Local GPU
EMBED_MODEL=/path/to/model python3 serve/embed_server.py

# Cloud providers (no GPU needed)
EMBED_PROVIDER=openai  OPENAI_API_KEY=...  python3 serve/embed_server.py
EMBED_PROVIDER=cohere  COHERE_API_KEY=...  python3 serve/embed_server.py
EMBED_PROVIDER=voyage  VOYAGE_API_KEY=...  python3 serve/embed_server.py
EMBED_PROVIDER=jina    JINA_API_KEY=...    python3 serve/embed_server.py

# Fully local, no GPU (requires Ollama running)
EMBED_PROVIDER=ollama  EMBED_PROVIDER_MODEL=nomic-embed-text  python3 serve/embed_server.py
```

| Provider | Default model | Dim | GPU needed |
|----------|--------------|-----|------------|
| local | qwen3-embed-8b | 4096 | Yes (~6GB) |
| openai | see provider docs | varies | No |
| cohere | see provider docs | varies | No |
| voyage | voyage-code-3 | 1024 | No |
| jina | jina-embeddings-v3 | 1024 | No |
| ollama | nomic-embed-text | 768 | No (local CPU) |

**Tested with:** local/qwen3-embed-8b, openai/text-embedding-3-large, voyage/voyage-code-3.

**Guidance:** retrieval quality scales with embedding dimension. Use the highest-dimension model available to you, and prefer models evaluated on code or technical text retrieval. Check your provider's documentation for their recommended model for this workload.

> **Important:** embedding dimension is fixed when you run stage 3. Switching providers later requires rebuilding `vectors.lance` by re-running `build/03_embed.py`.

---

## Connecting AI agents via MCP

Add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "codebase": {
      "type": "sse",
      "url": "http://127.0.0.1:8002/sse"
    }
  }
}
```

Works with Claude Code, Cursor, and Windsurf.

### The 8 MCP tools

| Tool | Use when |
|------|----------|
| `search_modules` | **Start here.** Find which namespace contains relevant code. |
| `get_module` | List all symbols in a namespace. |
| `search_symbols` | Semantic search — you know what a function does, not its name. |
| `get_function_body` | Read source of a function by its fully-qualified ID. |
| `trace_callers` | Who calls this function? (upstream impact) |
| `trace_callees` | What does this function call? (downstream dependencies) |
| `get_blast_radius` | Import graph + co-change impact for changed files or modules. |
| `get_context` | **Last resort.** Pre-built context block (large). Use only if targeted searches failed. |

**Optimal chain:** `search_modules → get_module → get_function_body → trace_callees`

**Retrieval:** `search_symbols` uses BM25 + dense vector **RRF fusion** (`unified_search`). Results are ranked by Reciprocal Rank Fusion across both signals — keyword hits for exact identifiers, vector search for semantic similarity. Co-change data is added as a structural coupling signal on top.

---

## PR blast-radius analysis

```bash
git diff main...HEAD --name-only | python3 serve/pr_analyzer.py
python3 serve/pr_analyzer.py --files src/Routes.hs src/Gateway.hs
git diff main...HEAD --name-only | python3 serve/pr_analyzer.py --format json
git diff main...HEAD --name-only | python3 serve/pr_analyzer.py --check security
```

---

## Applications you can build

The Chat UI and MCP server are two reference implementations. The retrieval engine is a general-purpose data layer — any application that benefits from understanding a codebase can be built on top of it.

### What large engineering organisations have used similar platforms for

**Developer tooling**
- Automated onboarding guides — generate a "tour" of any service for new engineers
- Codebase-aware code review bots — flag changes that touch historically fragile modules
- On-call runbooks — auto-generated from log patterns and call graphs, linked to actual code
- IDE plugins — inline answers to "what does this function do?" without leaving the editor

**Engineering operations**
- Incident response assistants — given an error log, trace back to the responsible module and its owner
- Dependency audits — identify which services depend on a library before upgrading it
- Security scanning agents — find every location where sensitive data (credentials, PAN, tokens) is handled
- Technical debt dashboards — surface modules with high co-change churn, low test coverage, or complex call graphs

**AI agent infrastructure**
- Coding agents that understand your internal APIs without fine-tuning
- Test generation agents — read a function body and generate unit tests grounded in actual behaviour
- Migration assistants — trace all call sites before deprecating an API
- Documentation generators — auto-draft docstrings and API guides from source + commit history

**CI/CD integration**
- Pre-merge blast-radius checks — block risky changes automatically
- Change-linked release notes — describe what changed and what it affects
- Cross-service impact summaries for release managers

The pattern in all cases is the same: **retrieve relevant context from the index, pass it to an LLM, act on the answer.**

---

## Building a ReAct agent

`retrieval_engine.py` is the data layer. You can write any agent that uses it — the Chat UI and MCP server are just two examples. Here is the minimal pattern:

### How ReAct works

ReAct (Reason + Act) is a loop where an LLM:
1. Reads the question and any prior tool results
2. Decides which tool to call (or stops if it has enough information)
3. The tool runs and the result is appended to the conversation
4. Repeat until the LLM produces a final answer

```
User question
     ↓
[LLM] → tool_call: search_modules("payment gateway")
     ↓
[Tool runs] → returns module list
     ↓
[LLM] → tool_call: get_function_body("Gateway.processPayment")
     ↓
[Tool runs] → returns source code
     ↓
[LLM] → no more tool calls → final answer streamed to user
```

### Minimal agent loop

```python
import retrieval_engine as RE
import tools  # AGENT_TOOLS and TOOL_DISPATCH live here
from openai import OpenAI

RE.initialize("/path/to/workspaces/YOUR_ORG/artifacts")
client = OpenAI(api_key="...", base_url="...")

def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a codebase expert. Use tools to answer questions."},
        {"role": "user",   "content": question},
    ]

    for _ in range(12):   # max tool calls
        resp = client.chat.completions.create(
            model="your-model",
            messages=messages,
            tools=tools.AGENT_TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content   # LLM finished reasoning

        # Append assistant turn + execute each tool
        messages.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [...]})   # serialise tool_calls
        for tc in msg.tool_calls:
            import json
            args   = json.loads(tc.function.arguments)
            result = tools.TOOL_DISPATCH[tc.function.name](args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "Investigation limit reached."
```

The real implementations in `demo_server_v6.py` (Chainlit) and `mcp_server.py` follow this exact pattern — the only difference is how tool steps are rendered and how the final answer is delivered.

### Choosing what to expose

`AGENT_TOOLS` in `tools.py` is the full list of tools the LLM can call. `TOOL_DISPATCH` maps tool names to functions. Your agent can use all of them or a subset depending on the use case:

| Use case | Recommended tools |
|----------|------------------|
| Answering architecture questions | search_modules, get_module, get_function_body, trace_callees |
| Impact analysis / PR review | get_blast_radius, trace_callers, search_symbols |
| Incident investigation | search_symbols, get_function_body, get_log_patterns |
| Security audit | search_symbols, get_function_body, trace_callers |
| Documentation generation | get_module, get_function_body, get_type_definition |

---

## Adding tools

Every tool in the system has three parts. Add all three to expose a new capability.

### 1. The function in `tools.py`

```python
def tool_find_tests(fn_id: str) -> str:
    """Find test files that reference a given function ID."""
    if not RE.G or fn_id not in RE.G.nodes:
        return f"Function '{fn_id}' not found."

    # Your retrieval logic here — search the graph, body_store, call_graph, etc.
    results = [
        nid for nid, d in RE.G.nodes(data=True)
        if "test" in d.get("file", "").lower()
        and fn_id.split(".")[-1] in RE.body_store.get(nid, "")
    ]
    return "\n".join(results[:20]) or "No test references found."
```

### 2. The schema in `AGENT_TOOLS`

`AGENT_TOOLS` is a list of OpenAI function-calling schemas. Add an entry so the LLM knows the tool exists and how to call it:

```python
{
    "type": "function",
    "function": {
        "name": "find_tests",
        "description": "Find test functions that reference a given function ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "fn_id": {
                    "type": "string",
                    "description": "Fully-qualified function ID, e.g. Module.SubModule.functionName"
                }
            },
            "required": ["fn_id"]
        }
    }
}
```

### 3. The dispatch entry in `TOOL_DISPATCH`

```python
TOOL_DISPATCH: dict = {
    # ... existing tools ...
    "find_tests": lambda a: tool_find_tests(a.get("fn_id", "")),
}
```

### 4. Expose via MCP (optional)

To make the tool available in AI coding assistants, add it to `serve/mcp_server.py`:

```python
@mcp.tool()
def find_tests(fn_id: str) -> str:
    """Find test functions that reference a given function ID."""
    RE.ensure_initialized()
    return RE.tool_find_tests(fn_id)
```

That is the complete change. The tool is now available in the Chat UI, MCP server, and any custom agent that uses `TOOL_DISPATCH`.

### Adding a language parser

To add support for a new language, implement the parser function in `build/01_extract.py` following the same contract as the existing parsers:

```python
def parse_javascript_file(path: pathlib.Path, service: str) \
        -> tuple[list, list, dict, dict, dict]:
    """Returns: (symbols, edges, body_store, call_store, log_store)"""
    symbols, edges = [], []
    body_store, call_store, log_store = {}, {}, {}

    # Each symbol must have: id, name, module, kind, type, file, lang, service
    # Each edge must have: from, to, kind, lang
    # body_store: fn_id → source text
    # call_store: fn_id → {"callees": [name, ...], "callers": []}
    # log_store:  fn_id → [log string, ...]

    return symbols, edges, body_store, call_store, log_store
```

Then add the file glob and parser call inside the service loop in `main()`.

---

## Config reference

`config.yaml` (copied from `config.example.yaml`) controls the build and serve layers.

```yaml
workspace: your-org

artifacts_dir: /path/to/workspaces/your-org/artifacts
output_dir:   /path/to/workspaces/your-org/output
source_dir:   /path/to/workspaces/your-org/source
git_history:  /path/to/workspaces/your-org/git_history.json
model_dir:    /path/to/models/your-embed-model

llm_base_url: https://your-llm-endpoint
llm_model:    your-model-name
llm_api_key:  your-api-key   # also readable from LLM_API_KEY env var

chainlit_port: 8000
mcp_port:      8002
embed_port:    8001

# ── Service profiles ──────────────────────────────────────────────────────────
# role:           orchestrator | connector | entry_point | persistence | worker
# traffic_weight: 0.0–1.0 — relative production importance; biases retrieval
# region:         which deployment region this service handles
service_profiles:
  payment-core:
    role:           orchestrator
    traffic_weight: 1.0
    region:         global
    description:    "Core transaction processing. Every payment passes through here."
  gateway-connector:
    role:           connector
    traffic_weight: 0.85
    region:         apac
    description:    "Connector layer for payment gateways."

# ── Keyword allowlist ─────────────────────────────────────────────────────────
# Short domain terms that bypass the stopword/length filter in keyword search
kw_allowlist:
  - upi
  - emi
  - mandate
  - 3ds

# ── Architecture doc generation (stage 8) ────────────────────────────────────
doc_generation:
  enabled:                true
  output_dir:             /path/to/workspaces/your-org/docs/generated
  max_verify_iterations:  3
  min_accuracy_threshold: 0.80
  entry_point_patterns:   [WorkFlow, Flow, Handler, Router, Service, Manager]
  skip_if_exists:         true
  model:                  your-model-name
```

---

## Best practices

### Build pipeline

- Run stage 3 (embed) with no other embedding workloads active — it is the most memory-intensive stage
- Stage 4 (summarize) checkpoints after every cluster — safe to interrupt and resume
- Always run stage 5 after stage 4 — it copies the final graph to `artifacts/` where servers load it from
- Stage 6 (co-change) streams git history at commit level — O(1) memory, safe on arbitrarily large history files
- Validate your git history export before running stage 6: check the last few bytes for truncation

### Running servers

- Start the embedding server before all others — if other servers start first they attempt in-process model loading
- Set `EMBED_SERVER_URL` in every server process that uses embeddings
- Run long-lived server processes with proper session isolation (`start_new_session=True` in Python subprocess, or a process manager like systemd/supervisor in production)
- In production, put servers behind a reverse proxy (nginx, Caddy) — do not expose ports directly

### Querying and retrieval

- Start exploration with `search_modules` — it returns namespaces, which are then browsable with `get_module`. This is faster and more precise than open-ended `search_symbols`.
- Short domain-specific terms (acronyms, payment codes, protocol names) may be filtered by length. Add them to `_KW_ALLOWLIST` in `retrieval_engine.py`.
- Batch independent tool calls in a single LLM turn — sequential single-call round-trips are slower and use more tokens
- `get_context` is a large fallback — only invoke it if targeted searches have genuinely failed
- If search results are consistently wrong for a particular term, check whether it is being split or normalised unexpectedly before the keyword search

### Workspace hygiene

- Keep platform code (`hyperretrieval/`) and org data (`workspaces/YOUR_ORG/`) in separate git repos — the data is not part of the product
- Rebuild the vector index (`03_embed.py`) whenever you change the embedding provider or model
- Re-run the full pipeline periodically as the codebase evolves — the co-change index especially benefits from fresh commit history
- Store API keys in environment variables or a secrets manager, never in `config.yaml`

---

## Design decisions and learnings

Every non-obvious decision in this system has a reason. This section explains what was built, what was tried first, what failed, and why the current approach is the right one. It is written to help anyone picking up this codebase understand not just *what* the system does but *why* it does it that way.

---

### Why tree-sitter over regex for parsing (Stage 1)

**The problem with regex:** Haskell and Rust use syntactic structures — indentation blocks, nested braces, `where` clauses — that are impossible to reliably delimit with regular expressions. The original regex-based body extraction either captured too little (cut off mid-function) or too much (bled into the next definition). An extracted body that starts correctly but ends at the wrong line confuses the embedding model and produces misleading similarity scores.

**What tree-sitter gives you:** Tree-sitter builds a full Abstract Syntax Tree (AST) using byte offsets. Every function node has a precise `start_byte` and `end_byte` — the body is exactly `src[start_byte:end_byte]`, no heuristic needed.

**The analogy:** Regex parsing is like estimating where a room ends by counting steps from the door. Tree-sitter is GPS — it gives you the exact boundary coordinates.

**The fallback:** Tree-sitter parsers are installed per-language (`tree-sitter-haskell`, `tree-sitter-rust`, `tree-sitter-groovy`). If a parser is unavailable, each extractor falls back to the regex approach automatically — so the system degrades gracefully rather than failing hard.

---

### Why Leiden over Louvain for clustering (Stage 2)

**The defect in Louvain:** Louvain has a mathematically proven flaw — it can produce *disconnected* communities, where nodes in the same cluster have no path between them in the graph. This means two modules with no shared imports or co-change history could end up in the same cluster, producing nonsensical summaries.

**Leiden fixes this provably.** The Leiden algorithm guarantees that every cluster is internally connected. It runs in O(n log n) — faster than Louvain on large graphs — and uses the same API, making it a drop-in replacement.

**Why it matters for us:** Cluster summaries are the high-level map that retrieval uses to route queries. A disconnected cluster produces a summary that mixes unrelated modules. The LLM summarizer then writes a confused description that sends queries to the wrong part of the codebase.

---

### Why BM25 + Reciprocal Rank Fusion alongside vector search (Serve layer)

**The exact-name problem:** Dense vector embeddings excel at semantic similarity — "what handles retry logic?" will find the right function even if it's named `executeWithBackoff`. But if you search for `TxnSplitDetail` (an exact type name), the embedding model may rank semantically-similar-but-wrong types above the exact match, because embeddings encode meaning, not spelling.

**BM25** (a classical information retrieval algorithm) uses IDF weighting — rare tokens score very high. `TxnSplitDetail` appears in few documents, so BM25 ranks the exact match first.

**Reciprocal Rank Fusion (RRF):** Instead of trying to combine two scored lists by tuning weights (fragile, hyperparameter-sensitive), RRF converts each list to rank positions and scores each document as `Σ 1/(60 + rank_i)`. The constant 60 is empirically validated across many retrieval benchmarks. No hyperparameters, no tuning — just merge and rank.

**The analogy:** Vector search is like a librarian who understands what you mean. BM25 is a librarian who reads the exact words on the spine. RRF is asking both and trusting whichever one agrees most.

---

### Why service importance weighting

**The problem without it:** In a multi-service codebase, the largest service (by symbol count) tends to dominate retrieval results — not because it's the right answer, but because it has the most vectors. A global expansion connector with 8,000 nodes and zero production traffic today would outscore a core payment processor with 2,000 nodes, simply because the search space is larger.

**The fix:** Each service gets a `traffic_weight` in `config.yaml` (0.0 to 1.0). This scalar is multiplied into retrieval scores before final ranking. A service marked `traffic_weight: 0.3` contributes proportionally less to results for queries where multiple services are candidates.

**The learning:** This came from observing that a global expansion connector with ~8K symbols and good graph centrality kept appearing in results for region-specific queries — despite having zero production traffic in that region. It was the right shape but the wrong service. Setting `traffic_weight: 0.3` for that service eliminated the noise.

---

### Why adaptive sampling in cluster summarisation (Stage 4)

**The problem with fixed sampling:** Stage 4 sends a sample of a cluster's nodes to an LLM for summarisation. With a fixed limit of 80 nodes and a cluster containing 820 modules, the sample was chosen by shuffling all modules randomly and taking one node from each until 80 were collected.

**What happened in practice:** The lottery had 820 entrants. There were 30+ individual gateway integration modules (each ~200–350 nodes) and 4 core transaction orchestration modules. Statistically, the sample contained ~4× more gateway integration than core business logic. The resulting LLM summary said the cluster was "a multi-provider payment orchestrator integrating 15+ gateways" — technically accurate but missing the mandate execution, retry logic, txn state machine, and split payment flows entirely.

**The fix has three parts:**
1. **Adaptive sample size:** For a cluster with 820 modules, sample `min(300, 820 // 3) = 273` nodes, not 80.
2. **Priority ordering:** Modules whose names contain `Product`, `OLTP`, `Flow`, `Handler`, `Transaction`, `Mandate`, `Payment`, `Routing` are sampled first, at 2× density.
3. **Boilerplate suppression:** Modules named `*.Lenses`, `Config.Constants`, `*.Shims`, `*.Generated` are sampled last, at ¼ density. These are auto-generated or mechanical files — no business logic lives there.

**The analogy:** If you need to understand what a hospital does, you interview doctors and nurses first. The lottery approach would give an equal chance to the boilerplate administrative forms as to the surgical team.

---

### Why parallel extraction (Stage 1)

Services in a multi-service codebase are completely independent data sources — no service's symbols depend on another service's symbols being parsed first. The original implementation processed them sequentially (one service, then the next), leaving N-1 cores idle.

Using `multiprocessing.Pool` with one worker per service means all services parse concurrently. On a 24-core machine with 13 services, all 13 run simultaneously. Wall time drops from ~hours to ~minutes.

**Important constraint:** The merge (combining all symbol lists, deduplication, caller index, writing outputs) still happens in the main process after all workers complete. This is intentional — merging requires a global view that no individual worker has.

---

### Why the body store is a separate file

When the serve layer starts, it loads the symbol graph (nodes + edges) into memory — this enables graph traversal, vector search, and keyword search. The graph JSON for 114K symbols is ~250MB. That's acceptable to keep resident.

The body store (full source text of every function) adds another ~50MB+ and is never needed unless a user explicitly requests `get_function_body`. Loading it at startup wastes memory and slows startup for a capability that is used in a minority of queries.

Solution: body store is loaded separately and accessed by key. Tools that don't need source text pay zero cost. Tools that do (`get_function_body`, `get_context`) read from it on demand.

---

### Why co-change beats call graph for cross-service coupling

For calls *within* a service (in-process function calls), the static call graph is authoritative. But services in a real system communicate over HTTP — and no static analysis can trace an HTTP boundary without knowing the runtime URL and routing table.

Co-change analysis sidesteps this entirely. If module A in service X and module B in service Y always change in the same commit, they are coupled — regardless of *how* they communicate. The coupling is observable in git history without understanding the runtime topology.

This is particularly valuable for blast-radius analysis: "if I change this module, what else do I need to test?" The answer from co-change is empirical (what *has* broken together historically), not theoretical (what *could* break given the type system).

---

### Enterprise calibration vs. single-repo findings

Several published retrieval benchmarks recommend capping search results at K=5–8, citing precision degradation beyond that point. Those benchmarks were conducted on single repositories with < 50K symbols.

In a multi-service enterprise codebase with 12+ services and 100K+ symbols:
- Relevant code may genuinely span 3–4 services
- A cap of 5 often excludes the right answer entirely
- The signal-to-noise problem is about *service bias*, not *total count*

The right fix for enterprise is **stratification** (cap per service, not per query) and **density thresholds** (drop results that score below 60% of the top result) — not reducing the total cap. This was an explicit decision to reject the benchmark recommendation as inapplicable to the target environment.

---

## Architecture deep-dive

```
embed_server.py (:8001)
  One process loads the embedding model (local GPU or cloud API).
  Serves POST /embed → {embeddings: [[float, ...]]}
  GET /health → {provider, model, dim, device, loaded}
  All other processes connect via EMBED_SERVER_URL — zero model duplication.

retrieval_engine.py  (imported by every entry point)
  initialize(artifact_dir) loads all indexes into memory:
    graph_with_summaries.json → NetworkX symbol graph G + module graph MG
    vectors.lance             → LanceDB vector table
    body_store.json           → fn_id → source body
    call_graph.json           → fn_id → {callees, callers}
    cochange_index.json       → module → [(module, weight), ...]
    log_patterns.json         → fn_id → [observable log strings]
    docs.lance                → LanceDB documentation chunks

  Exposes:
    AGENT_TOOLS     — OpenAI function-calling schemas for the LLM
    TOOL_DISPATCH   — name → callable, used by every agent implementation

Entry points (all import retrieval_engine, none duplicate logic):

  apps/chat/demo_server_v6.py (:8000)    Chainlit chat UI
    ReAct loop renders tool calls as expandable steps.
    Streams the final answer token by token.

  serve/mcp_server.py (:8002/sse)        MCP server
    Wraps TOOL_DISPATCH as MCP tools over SSE transport.
    Compatible with any MCP client.

  apps/cli/pr_analyzer.py                CLI
    resolve files → blast radius → optional LLM explanation.
    Exits non-zero for CI gates.

  your_app.py                  Anything you build
    Import retrieval_engine + tools, call RE.initialize(), use tools.TOOL_DISPATCH.
```

**Why two graphs (G and MG)?**
Import edges connect module *names*, not function IDs. G is a symbol-level graph — it holds function nodes and call/instance edges. MG is a module-level graph built at startup from raw import edges. Blast-radius traversal and module search use MG; function body lookup and call tracing use G and the body/call stores.

**Why stratified vector search?**
Without stratification, nearest-neighbour search returns results biased toward the largest service (most symbols). Stratification caps results per service before final ranking, ensuring all services get representation regardless of size.

**Why BM25 + RRF alongside vector search?**
Dense vector search excels at semantic similarity but can miss exact symbol name matches (e.g. `TxnSplitDetail`, `createPaymentLink`). BM25 scores rare identifiers highly using IDF weighting. Reciprocal Rank Fusion (RRF, k=60) merges both ranked lists without hand-tuned weights — each document scores `Σ 1/(60 + rank_i)` across retrieval methods. The combined result handles both "what handles retry logic?" (semantic) and "find TxnSplitDetail" (exact) correctly.

**Service importance weighting**
Each service in `config.yaml` has a `traffic_weight` (0.0–1.0) reflecting its production significance. This biases retrieval toward high-traffic services (e.g. the core transaction orchestrator) over low-traffic ones (e.g. a global expansion connector with zero production traffic today), preventing large-but-irrelevant services from monopolising results.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `name '_encode_queries_batch' is not defined` | Function deleted by mistake | Restore it — it is called by `_encode_query` and `stratified_vector_search` |
| Embed server reports `device=cpu` when GPU is available | GPU driver not visible to the process | Check `nvidia-smi` and CUDA driver installation; on WSL2 run `wsl --shutdown` and restart |
| Chainlit shows default name/theme instead of custom | Wrong working directory at launch | Run chainlit from `apps/chat/` — `.chainlit/` config and `public/` must be accessible from CWD |
| LanceDB write fails silently | Writing to a filesystem that does not support mmap | Write to a native Linux ext4 path; on WSL2 avoid `/mnt/` paths |
| Semantic search misses obvious results | Short terms filtered by length | Add short domain terms to `_KW_ALLOWLIST` in `retrieval_engine.py` |
| Co-change builder runs out of memory | Loading full repository objects into memory | Use `06_build_cochange.py` which streams at commit level, not repo level |
| Server process exits when terminal closes | Process attached to a PTY | Use a process manager, `nohup`, or `subprocess.Popen(start_new_session=True)` |
| vectors.lance is empty after stage 3 | Stage 3 wrote to a temp path and was not copied | Check `ARTIFACT_DIR` env var; ensure stage 5 ran after stage 3 |
| MCP tools not showing in Claude Code | Wrong transport or URL | Use `type: sse` in `.mcp.json`; confirm port is correct; check server log |
