# HyperRetrieval

**Every other code intelligence tool reads the code. HyperRetrieval reads the git history.**

Large codebases defeat LLMs — not because LLMs are incapable, but because they never see the right code at the right time. Static analysis tells you what *could* break. Git history tells you what *actually* breaks together. The gap between those two is where HyperRetrieval lives.

HyperRetrieval builds a **structured knowledge graph** of your entire codebase — symbols, call graphs, semantic embeddings, co-change history, activity metrics, cross-repo coupling, and Granger causality — then exposes it as fast, precise retrieval that AI agents can actually use.

Point it at your source repos, run the build pipeline once, and AI tools in your org get answers grounded in real source code with exact function names, module paths, and traced call chains.

---

## Try it in 60 seconds — no GPU, no config

Analyze any public GitHub repo's blast radius from its git history:

```bash
git clone https://github.com/Amitshukla2308/Index-the-code
cd Index-the-code
pip install -e .
python3 apps/cli/demo.py https://github.com/your-org/your-repo
open hr-demo-report.html
```

You get an HTML report of the highest blast-radius files — ranked by how many times they've historically triggered co-changes across the codebase. No embedding server, no GPU, no 15-minute pipeline. Just git history.

**On Flask:** `src/flask/app.py` scores 1120 — the single most-coupled file. Correct, and immediately useful.

For full MCP tool integration (15 tools, criticality scoring, Guard static checks, cross-repo coupling) → see [Setup](#setup) below.

---

## What ships in this repo

- **15 MCP tools** — plug directly into Claude Code, Cursor, Windsurf, and GitHub Copilot Agent Mode
- **Guardrails** — auto-generated "what must stay true" + "review checklist" docs for critical modules, surfaced in chat and accessible via 3 dedicated MCP tools
- **Criticality scoring** — every module gets a 0-1 score (blast radius + cross-repo coupling + recency + change frequency); used for risk-aware rerank in `unified_search`
- **Cursor IDE plugin** — zero-config marketplace plugin (`plugins/cursor/`)
- **Chat UI** (Chainlit) — engineers ask architecture questions in plain English; guardrails auto-surface when the answer touches a critical module
- **HRCode CLI** — AI coding assistant with 20 tools, 26 slash commands, persistent memory
- **Guardian Mode** — PR completeness analysis: blast radius, risk scoring, suggested reviewers, CI/CD GitHub Action
- **Blast Radius v2** — activity-weighted impact analysis; **recall@10 0.47** vs 0.11 for static-only (+322%)
- **TurboQuant vector compression** — 7.7x storage reduction at 3-bit (312MB vs 1.5GB), recall@10 preserved at 0.91 — makes HR deployable on standard dev laptops
- **Pluggable cross-encoder reranker** — `BAAI/bge-reranker-v2-m3` hook into `unified_search`, env-gated (`HR_RERANKER=bge`)
- **14-stage build pipeline** — indexes any multi-service codebase in one run

---

## Why temporal signals matter — the research

HyperRetrieval's core thesis is that **temporal signals from git history outperform static code analysis** for predicting what actually needs to change. This was validated through a series of experiments on a 94K-symbol, 12-service production codebase (113,916 commits).

### Key findings

| Experiment | Finding | Key number |
|-----------|---------|-----------|
| **exp_001**: Cross-repo co-change validation | Signal is real (p < 10^-13) and **orthogonal** to import graph — only 0.54% overlap | 1.91x weight when import edge present |
| **exp_003**: Change prediction model | Activity features dominate (79-84% importance); structural features add near-zero at short horizons | AUC 0.688 (full), 0.723 (hand-written only) |
| **exp_004**: Structural feature ceiling | Import distance, same_service, same_cluster add +0.002 at K=3 but +0.031 at K=50 | 15x more important at long horizon |
| **exp_005**: Cross-ecosystem validation | Activity dominance holds on Flask (Python) same as Juspay (Haskell) | 84% activity importance |
| **blast_radius v2 benchmark** | Activity-weighted reranking on 563 real commits | recall@10: 0.11 -> 0.47 (+322%), MRR: 0.08 -> 0.36 (+359%) |
| **"will_break" audit** | Only 14.9% of import neighbors ever actually co-change; 47.6% of modules have **zero** co-changing imports | Static "will break" labels are noise dressed as confidence |

> *"A retrieval system using only structural signals misses ~98% of what cross-repo co-change captures."*

Full experiment artifacts: `~/lab/experiments/`

---

## Data model

HyperRetrieval indexes your codebase into complementary data structures:

| Index | What it stores | Used for |
|-------|---------------|----------|
| **Symbol graph** | Functions, types, modules + import edges + Leiden clusters | Navigation, blast-radius, architecture mapping |
| **Vector index** | Semantic embedding of every symbol (4096d) | Natural-language search |
| **Body store** | Full source text per function | Code reading, LLM context |
| **Call graph** | Caller/callee relationships | Flow tracing, impact analysis |
| **Co-change index** | Modules that historically change together (weighted) | Risk-aware PR review, missing change prediction |
| **Cross-repo co-change** | Modules across repos changing within +/-24h | Cross-service coupling invisible to import graph |
| **Activity index** | Per-module commit frequency, recency, contributor count | Activity-weighted reranking in blast radius v2 |
| **Community detection** | Leiden clusters on co-change graph | Identifying tightly-coupled module groups |
| **Granger causality** | Directional "A causes B to change" with temporal lag | Causal impact predictions |
| **Ownership index** | Per-module git ownership from commit history | Suggested reviewers for PRs |
| **Author aliases** | Merged git identities (name/email variants) | Accurate ownership attribution |

---

## MCP tools (15)

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

| Tool | Use when |
|------|----------|
| `search_modules` | **Start here.** Find which namespace contains relevant code. |
| `get_module` | List all symbols in a namespace. |
| `search_symbols` | Semantic search — you know what a function does, not its name. |
| `get_function_body` | Read source of a function by its fully-qualified ID. |
| `trace_callers` | Who calls this function? (upstream impact) |
| `trace_callees` | What does this function call? (downstream dependencies) |
| `get_blast_radius` | **v2: activity-weighted.** Import graph + co-change + cross-repo + Granger + activity reranking. Tiered: `will_break` / `likely_affected` / `worth_checking`. |
| `predict_missing_changes` | PR review: predict files likely missing from a changeset. |
| `check_my_changes` | Guardian Mode: blast radius + missing changes + risk score + reviewers in one call. |
| `suggest_reviewers` | Module ownership from git history — who should review these files? |
| `score_change_risk` | Composite risk score (0-100): blast radius + coverage gap + reviewer concentration + service spread. |
| `check_criticality` | Criticality score + risk level (LOW/MEDIUM/HIGH/CRITICAL) + signals for one or more modules. |
| `get_guardrails` | Full guardrail document for a module — "what must stay true", "review checklist", blast radius, reviewers. |
| `list_critical_modules` | Top-N critical modules, optionally filtered by service or threshold — for architecture audits. |
| `get_context` | **Last resort.** Large context block. Use only if targeted searches failed. |

**Optimal chain:** `search_modules -> get_module -> get_function_body -> trace_callees`

**Retrieval:** `search_symbols` uses **3-signal RRF fusion** — BM25 keyword hits + dense vector semantic search + co-change graph expansion, merged via Reciprocal Rank Fusion (k=60).

---

## Blast Radius v2 — activity-weighted reranking

The original blast radius traversed the import graph to estimate impact. In practice, only **14.9% of import neighbors actually co-change** — the static graph massively over-predicts.

v2 adds temporal signals:
- **Within-repo co-change weight** — how often do these modules actually change together?
- **Cross-repo co-change** — coupling across service boundaries invisible to the import graph
- **Activity score** — commit frequency, recency, and contributor count
- **Granger causality** — directional "A causes B to change" with temporal lag

Results are tiered by confidence:

| Tier | Meaning | Action |
|------|---------|--------|
| `will_break` | Direct import + high co-change | Must test |
| `likely_affected` | Strong temporal signal | Should test |
| `worth_checking` | Weak signal or structural-only | Review if time permits |

**Benchmark (563 real commits):**

| Metric | Static-only (v1) | Activity-weighted (v2) | Delta |
|--------|-----------------|----------------------|-------|
| recall@10 | 0.11 | 0.47 | +322% |
| MRR | 0.08 | 0.36 | +359% |
| recall@20 | — | 0.57 | — |
| Latency | 0.9ms | 2.3ms | +1.4ms |

---

## Language support

| Language | Symbols | Bodies | Call graph | Log patterns |
|----------|---------|--------|------------|--------------|
| Haskell | yes | yes | yes (approx) | yes |
| Rust | yes | yes | yes | yes |
| JavaScript/TypeScript | yes | yes | yes (AST) | — |
| Python | yes | yes | yes (AST) | — |
| Groovy | yes | yes | — | yes |
| Go | yes | yes | yes | — |
| Java | yes | yes | yes | — |

Adding a new language: implement `parse_<lang>_file()` in `build/01_extract.py`.

---

## Setup from scratch

### Prerequisites

```bash
python3 --version   # 3.11+

pip install chainlit openai lancedb sentence-transformers networkx \
            pyarrow leidenalg igraph rank-bm25 mcp ijson pyyaml \
            tree-sitter tree-sitter-haskell tree-sitter-rust tree-sitter-groovy
```

### Step 1 — Prepare your workspace

```bash
mkdir -p ~/projects/workspaces/YOUR_ORG/{source,artifacts,output}
cp -r /path/to/service-a ~/projects/workspaces/YOUR_ORG/source/
cp ~/projects/hyperretrieval/config.example.yaml ~/projects/workspaces/YOUR_ORG/config.yaml
```

### Step 2 — Export git history

```bash
python3 build/00_export_git_history.py --repo ~/projects/workspaces/YOUR_ORG/source/service-a \
  --output ~/projects/workspaces/YOUR_ORG/git_history.json
```

### Step 3 — Choose an embedding provider

```bash
# Local GPU
EMBED_MODEL=/path/to/model python3 serve/embed_server.py

# Cloud providers (no GPU needed)
EMBED_PROVIDER=openai  OPENAI_API_KEY=...  python3 serve/embed_server.py
EMBED_PROVIDER=voyage  VOYAGE_API_KEY=...  python3 serve/embed_server.py
EMBED_PROVIDER=cohere  COHERE_API_KEY=...  python3 serve/embed_server.py
EMBED_PROVIDER=jina    JINA_API_KEY=...    python3 serve/embed_server.py
EMBED_PROVIDER=ollama  EMBED_PROVIDER_MODEL=nomic-embed-text  python3 serve/embed_server.py
```

### Step 4 — Run the build pipeline

```bash
cd ~/projects/hyperretrieval
export REPO_ROOT=~/projects/workspaces/YOUR_ORG/source
export OUTPUT_DIR=~/projects/workspaces/YOUR_ORG/output
export ARTIFACT_DIR=~/projects/workspaces/YOUR_ORG/artifacts

bash build/run_pipeline.sh   # 30 min – 2 h depending on codebase size

# Resume from a specific stage
bash build/run_pipeline.sh --from-stage 4
```

### Step 5 — Start the servers

```bash
# 1. Embedding server (start FIRST)
python3 serve/embed_server.py

# 2. MCP server (for AI coding assistants)
ARTIFACT_DIR=~/projects/workspaces/YOUR_ORG/artifacts python3 serve/mcp_server.py

# 3. Chat UI (optional)
cd serve && chainlit run demo_server_v6.py --port 8000
```

---

## Folder structure

```
hyperretrieval/
├── build/                          <- 14-stage pipeline
│   ├── 00_export_git_history.py    <- Export git history -> git_history.json
│   ├── 01_extract.py              <- Parse source (tree-sitter + ast, parallel)
│   ├── 01b_detect_author_aliases.py <- Merge duplicate git identities
│   ├── 02_build_graph.py          <- NetworkX graph + Leiden clustering
│   ├── 03_embed.py                <- GPU-batch embed -> LanceDB
│   ├── 04_summarize.py            <- LLM cluster summaries (resumable)
│   ├── 05_package.py              <- Package artifacts
│   ├── 06_build_cochange.py       <- Co-change index (streaming, O(1) memory)
│   ├── 06b_build_cross_cochange.py <- Cross-repo co-change (+/-24h window)
│   ├── 07_chunk_docs.py           <- Embed markdown docs
│   ├── 08_build_ownership.py      <- Per-module git ownership
│   ├── 09_build_granger.py        <- Granger causality (directional, with lag)
│   ├── 09b_build_viz_data.py      <- Visualization data
│   ├── 10_build_activity.py       <- Per-module activity metrics
│   ├── 10_build_communities.py    <- Leiden on co-change graph
│   └── run_pipeline.sh
│
├── serve/
│   ├── retrieval_engine.py        <- Core: loads all indexes, all retrieval logic
│   ├── embed_server.py            <- Shared embedding server (:8001)
│   ├── mcp_server.py             <- MCP SSE server (:8002) — 12 tools
│   └── pr_analyzer.py            <- Guardian Mode CLI for CI/CD
│
├── plugins/
│   └── cursor/                    <- Cursor IDE marketplace plugin
│       ├── manifest.json
│       ├── server.py
│       └── README.md
│
├── tools/
│   └── viz/                       <- Codebase visualization (:8003)
│       ├── index.html             <- D3: Services, Clusters, 2D Scatter views
│       └── serve.py
│
├── tests/
│   ├── test_01_artifacts.py       <- Verify build outputs
│   ├── test_02_retrieval_logic.py <- Unit tests for retrieval
│   ├── test_03_canary.py          <- Smoke test
│   ├── test_04_retrieval_accuracy.py <- Known-answer benchmark
│   ├── test_05_integration.py     <- End-to-end server test
│   ├── test_06_auto_eval.py       <- LLM-as-judge quality eval
│   ├── test_07_author_aliases.py  <- Author alias detection tests
│   ├── bench_blast_radius_recall.py <- Blast radius v2 recall benchmark
│   ├── chat_50_questions.json     <- 50-question eval set
│   └── run_chat_eval.py           <- Automated eval runner
│
├── tools.py                       <- Tool implementations + AGENT_TOOLS schemas
├── config.example.yaml            <- Template workspace config
└── .mcp.json                      <- MCP server config for AI assistants
```

```
workspaces/YOUR_ORG/               <- Org-specific data (not in git)
├── config.yaml
├── source/                        <- Your source repos
├── artifacts/                     <- Indexes loaded at runtime
│   ├── graph_with_summaries.json
│   ├── vectors.lance/
│   ├── cochange_index.json
│   ├── cross_cochange_index.json
│   ├── activity_index.json
│   ├── granger_index.json
│   └── ownership_index.json
├── output/
│   ├── body_store.json
│   ├── call_graph.json
│   └── author_aliases.yaml
└── git_history.json
```

---

## Guardian Mode — CI/CD integration

Guardian analyzes PRs for completeness: blast radius, missing changes, risk score (0-100), and suggested reviewers.

```bash
# CLI usage
git diff main...HEAD --name-only | python3 serve/pr_analyzer.py
python3 serve/pr_analyzer.py --files src/Routes.hs src/Gateway.hs --format json

# GitHub Action (zero-config)
# Copy .github/workflows/guardian-lite.yml to any repo
# Works on any language — needs only git history
```

### Zero-Config Guardian

```bash
# Initialize on any repo (no AST, no GPU, no config needed)
python3 apps/cli/guardian_init.py --repo /path/to/repo

# Runs with just git history — no embedding model required
python3 apps/cli/pr_analyzer.py --mode guardian --files file1.py file2.py
```

---

## Codebase visualization

Interactive D3 visualization on port 8003 with three views:

- **Services** — force-directed graph, sized by symbol count, connected by import edges
- **Clusters** — named clusters colored by service, connected by co-change edges
- **2D Scatter** — every symbol projected via UMAP, colored by service or cluster

```bash
# Build data files
python3 build/09_build_viz_data.py
python3 build/09b_build_scatter_data.py   # requires umap-learn

# Serve
VIZ_PORT=8003 python3 tools/viz/serve.py
```

---

## Architecture

```
embed_server.py (:8001)
  One process loads the embedding model. All others connect via EMBED_SERVER_URL.
  Supports: local GPU, OpenAI, Cohere, Voyage, Jina, Ollama.

retrieval_engine.py (imported by all entry points)
  initialize(artifact_dir) loads: graph, vectors, bodies, call graph,
  co-change, cross-repo co-change, activity, Granger, ownership, docs.

  Core capabilities:
    unified_search()     — 3-signal RRF (vector + BM25 + co-change expansion)
    get_blast_radius()   — v2 activity-weighted tiered impact analysis
    predict_missing()    — changeset completeness prediction

Entry points:
  mcp_server.py (:8002)         12 MCP tools over SSE
  demo_server_v6.py (:8000)     Chainlit chat UI
  pr_analyzer.py                CLI for CI/CD
  plugins/cursor/server.py      Cursor IDE plugin
```

**Two graphs (G and MG):** G is symbol-level (function nodes, call edges). MG is module-level (import edges). Blast radius and module search use MG; function lookup and call tracing use G.

**Stratified vector search:** Caps results per service before ranking, preventing large services from dominating results.

**Service importance weighting:** `traffic_weight` in config.yaml biases retrieval toward production-critical services.

---

## Design decisions

### Why temporal signals over static analysis
Static import graphs tell you what *could* break — but only 14.9% of import neighbors actually co-change. Activity-weighted reranking from git history tells you what *actually* breaks together. Benchmark: recall@10 from 0.11 to 0.47.

### Why BM25 + RRF alongside vector search
Vector search finds "what handles retry logic?" but misses exact names like `TxnSplitDetail`. BM25 scores rare identifiers via IDF weighting. RRF merges both with `1/(60 + rank)` — no hyperparameters.

### Why Leiden over Louvain
Louvain can produce disconnected communities. Leiden guarantees internal connectivity, runs faster, and is a drop-in replacement.

### Why tree-sitter over regex
Regex body extraction cuts off mid-function or bleeds into the next definition. Tree-sitter gives exact `start_byte:end_byte` from the AST.

### Why cross-repo co-change
Services communicate over HTTP — no static analysis traces that boundary. Co-change captures empirical coupling from git history regardless of communication mechanism.

### Why the body store is separate
The 250MB symbol graph stays resident. The 50MB+ body store loads on demand — only `get_function_body` needs it.

---

## Best practices

- Start the embedding server before all other servers
- Start exploration with `search_modules`, not `search_symbols`
- Add short domain terms (acronyms, protocol names) to `kw_allowlist` in config
- Keep platform code and org data in separate repos
- Rebuild indexes periodically as the codebase evolves — co-change benefits most from fresh history
- Store API keys in environment variables, never in config.yaml

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Embed server reports `device=cpu` | Check `nvidia-smi` and CUDA installation |
| Semantic search misses obvious results | Add short domain terms to `kw_allowlist` |
| Co-change builder OOM | Use `06_build_cochange.py` (streams at commit level) |
| MCP tools not showing in Claude Code | Use `type: sse` in `.mcp.json`; check port |
| LanceDB write fails | Write to native ext4, not `/mnt/` on WSL2 |
| Server dies when terminal closes | Use `subprocess.Popen(start_new_session=True)` or systemd |
| vectors.lance empty after stage 3 | Ensure stage 5 ran after stage 3 |

---

## Research

The temporal signal research behind blast radius v2 is documented in:

- `~/lab/experiments/exp_001_crosscochange_validation/` — cross-repo co-change signal validation
- `~/lab/experiments/exp_003_change_prediction/` — change prediction model (1.4M labeled pairs)
- `~/lab/experiments/exp_004_*/` — structural feature ceiling analysis
- `~/lab/experiments/exp_005_*/` — cross-ecosystem generalizability (Flask + Juspay)
- `~/lab/OPEN_QUESTIONS.md` — active research threads
- `~/lab/NOTEBOOK.md` — lab notebook with session-by-session findings

Research conducted by [Carlsbert](https://carlsbert.tunnelthemvp.in/journal/) — an autonomous Claude agent running as a digital co-founder.

---

*Built by [Amit Shukla](https://github.com/Amitshukla2308) with research by Carlsbert.*
