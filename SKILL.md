# HyperRetrieval — Codebase Intelligence Skill

**Trigger**: `/hyperretrieval` or when working with a large codebase indexed by HyperRetrieval.

Use this skill to answer precise questions about large codebases without hallucinating. HyperRetrieval indexes source code into a graph + vector store and exposes 32 MCP tools for semantic search, blast radius analysis, ownership lookup, and AI code guardrails.

## Prerequisites

HyperRetrieval MCP server running at `http://localhost:8002/sse` (or configured endpoint).

**Quick start:**
```bash
# Clone and start
git clone https://github.com/Amitshukla2308/use-ripple
cd use-ripple
pip install -r requirements.txt
python serve/mcp_server.py  # port 8002
```

**MCP config** (`.mcp.json` at project root):
```json
{"mcpServers":{"hyperretrieval":{"type":"sse","url":"http://127.0.0.1:8002/sse"}}}
```

---

## Optimal Tool Sequences

### "What does module X do?"
```
search_modules("X") → get_module("X.Y") → get_function_body("X.Y.fn")
```

### "Who calls function F?"
```
search_symbols("F") → trace_callers("Module.fn")
```

### "What breaks if I change file F?"
```
get_blast_radius(["path/to/file.hs"])
```
Returns: import graph expansion + co-change history + Granger causal predictions + risk score.

### "What files am I probably missing from this PR?"
```
predict_missing_changes(["file1.hs", "file2.rs"])
```

### "Is my change safe to merge?"
```
check_my_changes(["file1.hs", "file2.hs"])
```
Returns: PASS/WARN/FAIL verdict + blast radius + Guard findings + AI provenance + reviewers.

### "How does X work?" (behavioral queries)
```
search_requirements("how does payment routing work")
```
Returns: functional requirement clusters with behavior tags (retry, idempotency, locking, etc.).

### "Who owns this code?"
```
suggest_reviewers(["path/to/file.hs"])
```

### "Is this code risky?"
```
score_change_risk(["file1", "file2"])    # 0-100 composite risk score
get_guardrails("Module.name")            # org-specific invariants that must hold
```

### Fast symbol lookup (no GPU needed)
```
fast_search("PaymentGateway")           # BM25, <50ms
fast_search_reranked("payment routing") # BM25 + cross-encoder, P@10=0.90
```

---

## Tool Reference (32 tools)

| Tool | Purpose |
|------|---------|
| `search_modules` | Find modules by topic — use FIRST for any new exploration |
| `get_module` | List all symbols in a module |
| `search_symbols` | Semantic + keyword search for functions/types |
| `get_function_body` | Source code of a function |
| `trace_callers` | Who calls this function (upstream) |
| `trace_callees` | What this function calls (downstream) |
| `get_blast_radius` | Full change impact: imports + co-change + Granger causal |
| `predict_missing_changes` | PR review: files likely missing from changeset |
| `check_my_changes` | Full SDLC guardian: blast + Guard + risk + reviewers |
| `suggest_reviewers` | Ownership from git history |
| `score_change_risk` | 0-100 risk score |
| `fast_search` | Zero-GPU BM25, <50ms |
| `fast_search_reranked` | BM25 + reranker, P@10=0.90 |
| `search_requirements` | Behavioral/flow queries ("how does X work?") |
| `get_why_context` | WHY context: ownership, Granger direction, anti-patterns |
| `get_context` | Deep context dump (last resort — high token cost) |
| `check_criticality` | Criticality score for a module |
| `get_guardrails` | Org-specific invariants for a module |
| `list_critical_modules` | Modules above criticality threshold |

---

## Token Budget

- Well-scoped question: ~5K tokens / 4-6 tool calls
- `get_context`: 5K–18K tokens — call once max, never twice per turn
- **Never spawn a subagent for MCP queries** — 18× token waste

## Supported Languages

Haskell, Rust, Python, JavaScript/TypeScript, Go, Java, Groovy

## Key Rules

1. Start with `search_modules` for any new topic
2. Use `search_requirements` for behavioral/"how does X work?" queries
3. Use `fast_search` for identifier/symbol name lookups (faster, no embed server)
4. Only call `get_context` as a last resort
5. `check_my_changes` before any PR — catches what you missed
