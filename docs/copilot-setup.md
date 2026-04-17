# HyperRetrieval + GitHub Copilot Agent Mode

Use HyperRetrieval's 12 MCP tools inside GitHub Copilot's Agent Mode for deep codebase intelligence — blast radius, change prediction, code search, and more.

## Prerequisites

1. HyperRetrieval MCP server running:
   ```bash
   cd serve && python mcp_server.py  # starts on port 8002
   ```

2. VS Code with GitHub Copilot extension (v1.111+)

## Setup (VS Code)

Create `.vscode/mcp.json` in your project root:

```json
{
  "servers": {
    "hyperretrieval": {
      "type": "sse",
      "url": "http://127.0.0.1:8002/sse"
    }
  }
}
```

That's it. Open Copilot Chat, select **Agent** mode, and HyperRetrieval's tools are available.

## Setup (JetBrains)

In Settings > Tools > AI Assistant > MCP Servers, add:

- **Name**: hyperretrieval
- **Type**: SSE
- **URL**: `http://127.0.0.1:8002/sse`

## Setup (Claude Code)

Already configured via `.mcp.json` in the project root:

```json
{"mcpServers": {"hyperretrieval": {"type": "sse", "url": "http://127.0.0.1:8002/sse"}}}
```

## Available Tools

| Tool | What it does |
|------|-------------|
| `search_modules` | Find module namespaces — use this first |
| `get_module` | List all symbols in a module |
| `search_symbols` | Semantic + keyword search for functions/types |
| `get_function_body` | Source code of a specific function |
| `trace_callers` | Who calls this function (upstream) |
| `trace_callees` | What this function calls (downstream) |
| `get_blast_radius` | Impact analysis for changed files |
| `predict_missing_changes` | Detect files likely missing from a PR |
| `check_my_changes` | Full PR guardian: blast radius + missing + risk + security |
| `suggest_reviewers` | Who should review changes to these files |
| `score_change_risk` | Composite risk score (0-100) |
| `get_context` | Full context dump (use as last resort) |

## Example Prompts in Copilot Agent Mode

- "What would break if I change `PaymentFlows.processPayment`?"
- "Who should review changes to the gateway module?"
- "Find all functions related to UPI payment processing"
- "Check if my current changes are missing any files"

## Remote Access

If HyperRetrieval runs on a different machine (e.g., a build server), replace `127.0.0.1` with the server's address. The MCP server supports any HTTP client — no special authentication required.

For Cloudflare Tunnel setups:
```json
{
  "servers": {
    "hyperretrieval": {
      "type": "sse",
      "url": "https://hr.your-domain.com/sse"
    }
  }
}
```
