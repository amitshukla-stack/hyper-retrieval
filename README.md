# Ripple

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![MCP Tools](https://img.shields.io/badge/MCP-15%20tools-brightgreen)](https://modelcontextprotocol.io)
[![Guard](https://img.shields.io/badge/Guard-2.4ms%2Ffile-orange)](#guard)

**Code intelligence from git history — not static analysis.**

Works inside **Claude Code · Cursor · GitHub Copilot Agent · OpenHands · Windsurf**

Static analysis tells you what *could* break. Git history tells you what *actually* breaks together. The gap between those two is where production incidents live — and where Ripple operates.

---

## Try it in 60 seconds — no GPU, no config

Analyze any public repo's blast radius from its commit history:

```bash
git clone https://github.com/Amitshukla2308/Index-the-code
cd Index-the-code
pip install -e .
python3 apps/cli/demo.py https://github.com/your-org/your-repo
open hr-demo-report.html
```

You get an HTML report of the highest-risk files ranked by co-change history. **On Flask:** `src/flask/app.py` scores 1120 — the single most-coupled file across 2000 commits. No surprise. Immediately useful.

---

## What it does

### Blast Radius — +322% recall over static import graph

Every other tool counts import edges. Ripple counts how often files *actually changed together* across your git history. The difference: only **14.9% of import neighbors ever co-change** — the static graph predicts risk for files that don't need review. Temporal signals catch the 85% the graph misses.

| Metric | Static (v1) | Temporal (v2) | Delta |
|--------|------------|--------------|-------|
| recall@10 | 0.11 | **0.47** | **+322%** |
| MRR | 0.08 | **0.36** | +359% |

### Guard — static semantic checks at 2.4ms/file

AI-generated code passes every review gate because it *looks* correct. Guard verifies what it *claims*: checks that comments match the code that follows, that locks aren't released before promised mutations complete, that auth happens before action. Catches the class of bugs where the AI wrote a plausible lie.

```bash
# Run on any Python/Haskell/Rust/Go/JS codebase
python3 -m ripple.guard path/to/changed_file.py
```

Patterns: lock scope, premature release, transaction boundaries, auth-before-action, error swallowing.

### 15 MCP Tools — plug into any AI coding assistant

One config block gives your entire team's AI assistants access to your codebase's history:

```json
{
  "mcpServers": {
    "ripple": {
      "type": "sse",
      "url": "http://127.0.0.1:8002/sse"
    }
  }
}
```

**Setup guides:** [Cursor](docs/cursor.md) · [GitHub Copilot](docs/copilot.md) · [OpenHands](docs/openhands.md)

| Tool | What it answers |
|------|----------------|
| `check_my_changes` | Full PR verdict: blast radius + Guard + risk score + reviewers |
| `get_blast_radius` | Which files co-change with these? Tiered by confidence. |
| `get_why_context` | WHY is this code the way it is? Ownership, activity trend, Granger causal direction, anti-patterns. |
| `predict_missing_changes` | What files are likely missing from this PR? |
| `score_change_risk` | Composite 0-100 risk score for a changeset |
| `suggest_reviewers` | Who owns these modules from git history? |
| `check_criticality` | How critical is this module? (blast + coupling + recency) |
| `get_guardrails` | What must stay true when touching this module? |
| `list_critical_modules` | Top-N highest-risk modules in the codebase |
| `fast_search` | Zero-GPU BM25 keyword search, ~40ms p50. No embed server needed. |
| `search_symbols` | Semantic + keyword + co-change fusion search |
| `search_modules` | Find which namespace contains relevant code |
| `get_module` | All symbols in a module |
| `get_function_body` | Source code of a function by ID |
| `trace_callers` | Who calls this? (upstream impact) |
| `trace_callees` | What does this call? (downstream deps) |
| `get_context` | Large context block — last resort |

---

## Full setup

### Prerequisites

```bash
python3 --version   # 3.11+
pip install chainlit openai lancedb sentence-transformers networkx \
            pyarrow leidenalg igraph rank-bm25 mcp ijson pyyaml \
            tree-sitter tree-sitter-haskell tree-sitter-rust
```

### Step 1 — Prepare workspace

```bash
mkdir -p ~/projects/workspaces/YOUR_ORG/{source,artifacts,output}
cp path/to/your/repos ~/projects/workspaces/YOUR_ORG/source/
cp config.example.yaml ~/projects/workspaces/YOUR_ORG/config.yaml
```

### Step 2 — Choose embedding provider

```bash
# Local GPU (no API cost)
EMBED_MODEL=/path/to/model python3 serve/embed_server.py

# Cloud (any, no GPU needed)
EMBED_PROVIDER=openai  OPENAI_API_KEY=sk-...  python3 serve/embed_server.py
EMBED_PROVIDER=voyage  VOYAGE_API_KEY=...     python3 serve/embed_server.py
EMBED_PROVIDER=cohere  COHERE_API_KEY=...     python3 serve/embed_server.py
EMBED_PROVIDER=jina    JINA_API_KEY=...       python3 serve/embed_server.py
EMBED_PROVIDER=ollama  EMBED_PROVIDER_MODEL=nomic-embed-text  python3 serve/embed_server.py
```

### Step 3 — Build the index

```bash
export REPO_ROOT=~/projects/workspaces/YOUR_ORG/source
export OUTPUT_DIR=~/projects/workspaces/YOUR_ORG/output
export ARTIFACT_DIR=~/projects/workspaces/YOUR_ORG/artifacts

bash build/run_pipeline.sh   # 30 min – 2 h depending on codebase size
```

### Step 4 — Start the servers

```bash
python3 serve/embed_server.py   # start first — other servers share it
ARTIFACT_DIR=~/projects/workspaces/YOUR_ORG/artifacts python3 serve/mcp_server.py
```

Add `.mcp.json` to your project and your AI assistant has all 15 tools.

---

## Language support

| Language | Symbols | Call graph | Guard | Co-change |
|----------|---------|------------|-------|-----------|
| Python | ✓ | ✓ | ✓ | ✓ |
| Haskell | ✓ | ✓ (approx) | ✓ | ✓ |
| Rust | ✓ | ✓ | ✓ | ✓ |
| JavaScript/TypeScript | ✓ | ✓ | ✓ | ✓ |
| Go | ✓ | ✓ | ✓ | ✓ |
| Groovy | ✓ | — | ✓ | ✓ |
| Java | ✓ | ✓ | — | ✓ |

---

## Guardian Mode — 0.2s setup, any repo

No GPU. No API key. No AST parser. Just a git repo.

```bash
# Initialize on any repo — reads only git history
python3 apps/cli/guardian_init.py --repo /path/to/your/repo
```

Output in 0.2 seconds:
```
Guardian initialized for: /path/to/your/repo
  Co-change index: 2,847 module pairs (from 4,219 commits)
  Ownership index: 187 authors, 634 modules
  Artifacts: .hyperretrieval/artifacts/
  Ready for: pr_analyzer.py, get_blast_radius, suggest_reviewers
```

**Why this matters**: Static analysis tools need AST parsers, language-specific rules, and 30+ seconds of setup per repo. Guardian reads git history — which every repo already has, in every language, with no configuration.

### Add to CI in 3 steps

```bash
# Step 1: Initialize (once per repo)
python3 apps/cli/guardian_init.py --repo .

# Step 2: Analyze a PR
git diff main...HEAD --name-only | python3 serve/pr_analyzer.py \
  --artifact-dir .hyperretrieval/artifacts

# Step 3: Add the GitHub Action (copy once, works forever)
cp .github/workflows/guardian-lite.yml /path/to/your/repo/.github/workflows/
```

The Action runs on every PR, posts a risk score comment, and flags missing changes from the co-change history. Zero secrets required.

### What Guardian catches (without reading a line of code)

| Signal | How | Example finding |
|--------|-----|----------------|
| Missing co-changes | Git history | "src/auth.py changed — src/session.py changed in 73% of prior PRs. Not in this PR." |
| Blast radius | Import + co-change | "This touches 3 services. 8 cross-service callers historically affected." |
| Reviewer suggestions | `git blame` ownership | "3 engineers own 80% of these files. None are in this PR's reviewers." |
| Granger causality | Commit sequence | "Changes to payments/ historically precede changes to fraud/ within 3 commits." |

---

## Architecture

```
embed_server.py (:8001)   — loads embedding model once; all servers connect to it
mcp_server.py   (:8002)   — 28 MCP tools over SSE
demo_server_v6.py (:8000) — Chainlit chat UI (optional)
retrieval_engine.py       — core: all indexes, all retrieval logic, imported by everything
```

Indexes built once, loaded at startup: symbol graph, vector store (LanceDB), co-change, cross-repo co-change, Granger causality, activity metrics, ownership, guardrail docs.

**TurboQuant**: optional 7.7x vector compression at 3-bit (312MB vs 1.5GB), recall@10 preserved at 0.91. Set `QUANT_BITS=3` at build time to deploy on laptops.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Embed server shows `device=cpu` | Check `nvidia-smi` and CUDA |
| Semantic search misses domain terms | Add short acronyms to `kw_allowlist` in config |
| MCP tools missing in IDE | Verify `type: sse` in `.mcp.json`, check port 8002 |
| LanceDB write fails on WSL2 | Write to `/home/`, not `/mnt/d/` (ext4 only) |
| Co-change builder OOM | Use `06_build_cochange.py` — streams at O(1) memory |

---

## Research

Ripple's temporal signals thesis was validated on a 94K-symbol, 12-service production codebase (113,916 commits):

- **Cross-repo co-change**: signal is real (p < 10⁻¹³), orthogonal to import graph (0.54% overlap), 1.91× weight when import edge present
- **Change prediction model**: activity features dominate (79–84% importance); structural features add near-zero at short horizons (K=3)
- **Cross-ecosystem**: activity dominance holds on Flask (Python) as on Haskell — 84% activity importance

Full artifacts: `~/lab/experiments/` · Active threads: `~/lab/OPEN_QUESTIONS.md`

---

## Self-hosted. Your code never leaves your machines.

*Built by [Amit Shukla](https://github.com/Amitshukla2308) · Research by [Carlsbert](https://carlsbert.tunnelthemvp.in/journal/) — an autonomous Claude agent*
