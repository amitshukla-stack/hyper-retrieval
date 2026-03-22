# HyperRetrieval — Codebase Intelligence Platform

## Project Layout

```
~/projects/
├── hyperretrieval/           ← THIS REPO (platform code only)
│   ├── serve/                ← runtime: retrieval_engine.py, demo_server_v6.py, embed_server.py, mcp_server.py, pr_analyzer.py
│   ├── build/                ← pipeline: 01_extract.py … 07_chunk_docs.py
│   ├── tools/                ← utilities
│   ├── tests/
│   └── config.example.yaml
├── workspaces/
│   └── juspay/               ← org-specific data (NOT in git)
│       ├── config.yaml
│       ├── artifacts/        ← graph_with_summaries.json, vectors.lance, cochange_index.json
│       ├── output/           ← body_store.json, call_graph.json, log_patterns.json, docs.lance
│       ├── source/           ← 12 Juspay service repos
│       └── git_history.json
└── models/
    └── qwen3-embed-8b/       ← 15GB Qwen3 embedding model
```

## Starting Servers

**Always start in this order (embed server must be ready before others):**

```bash
~/start_embed.sh    # GPU model load ~35s, watch: tail -f ~/embed_server.log
~/start_chainlit.sh # Chainlit UI on :8000, watch: tail -f ~/chainlit.log
~/start_mcp.sh      # MCP SSE on :8002, watch: tail -f ~/mcp_server.log
```

**Check if running:**
```bash
ps aux | grep -E 'embed_server|chainlit|mcp_server'
```

**Kill a server:**
```bash
fuser -k 8000/tcp  # chainlit
fuser -k 8001/tcp  # embed server
fuser -k 8002/tcp  # mcp server
```

**After WSL restart:** GPU may need `wsl --shutdown` from Windows then restart if `/dev/nvidia*` missing.

## Environment Variables

| Var | Value | Used by |
|-----|-------|---------|
| `ARTIFACT_DIR` | `/home/beast/projects/workspaces/juspay/artifacts` | retrieval_engine.py, demo_server_v6.py |
| `EMBED_SERVER_URL` | `http://localhost:8001` | retrieval_engine.py, mcp_server.py |
| `EMBED_MODEL` | `/home/beast/projects/models/qwen3-embed-8b` | embed_server.py (fallback if no embed server) |
| `HF_HUB_OFFLINE` | `1` | prevents HuggingFace download attempts |

## Architecture

```
embed_server.py (:8001) ← GPU model loaded once, shared via HTTP
         ↑ EMBED_SERVER_URL
retrieval_engine.py ← all data + retrieval functions
    ↑                      ↑                     ↑
demo_server_v6.py    mcp_server.py         pr_analyzer.py
(Chainlit :8000)     (MCP SSE :8002)       (CLI / CI)
```

## Juspay Workspace — 12 Services

euler-api-gateway (39,806), euler-api-txns (30,673), UCS (7,787), euler-db (5,610), euler-api-order (3,652),
graphh (2,377), euler-api-pre-txn (2,364), euler-api-customer (1,231), basilisk-v3 (335), euler-drainer (233),
token_issuer_portal_backend (121), haskell-sequelize (55)

## MCP Tools (7)

search_symbols, search_modules, get_module, get_function_body, trace_callers, trace_callees, get_blast_radius, get_context

**Optimal query path:** `search_modules("X")` → `get_module("X.Y")` → `get_function_body("X.Y.fn")` → `trace_callees`

**CRITICAL:** Never spawn a general-purpose Agent for MCP queries — call tools directly (18× token savings).

## WSL Command Patterns

- **Deploy from Windows:** `cmd //c "wsl cp /mnt/d/downloads/repo/FILE ~/projects/hyperretrieval/serve/FILE"`
- **Complex WSL commands:** Write to script file with `cmd //c "wsl tee /tmp/script.sh" << 'EOF' ... EOF` then `cmd //c "wsl bash /tmp/script.sh"`
- **Python:** Always `/home/beast/miniconda3/bin/python3` (3.13)
- **Git Bash path trap:** Never `cmd //c "wsl /home/..."` — use `cmd //c "wsl bash /tmp/script.sh"` pattern

## .mcp.json (Claude Code integration)

```json
{"mcpServers":{"juspay-code":{"type":"sse","url":"http://127.0.0.1:8002/sse"}}}
```
Located at `~/projects/hyperretrieval/.mcp.json`

## Known Pitfalls

1. **OOM**: Never load data in a second process while Chainlit is running — WSL2 VM OOM requires reboot
2. **GPU blocked**: After all GPU processes killed, `wsl --shutdown` from Windows + restart fixes `/dev/nvidia*`
3. **Chainlit config lookup**: `.chainlit/` and `public/` must be in CWD at launch (i.e. `serve/`)
4. **setsid required**: All long-running WSL processes must use setsid or they die when terminal closes
5. **`_encode_queries_batch`**: DO NOT remove — used by `_encode_query` and `stratified_vector_search`

## Pending Tasks

1. Copy euler-docs markdown to `workspaces/juspay/` for internal doc embedding
2. Test pr_analyzer.py on a real webhook file change
3. Package as pyproject.toml for generic codebase use
4. Rebuild pipeline stage 2 to fix import edge propagation to node-level graph (optional — MG workaround in place)
