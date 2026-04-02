# HyperRetrieval — Cursor Plugin

Code impact intelligence for every PR. Zero config.

## What it does

- **Blast Radius**: See which modules will break, may break, or need review when you change code
- **Missing Changes**: Detect files you likely forgot to include in your PR
- **Reviewer Suggestions**: Find the right reviewers based on git history ownership

## How it works

On workspace open, HyperRetrieval indexes your repo using git history only. No GPU, no cloud, no config. Analysis runs locally in seconds.

The plugin exposes 3 MCP tools that your AI assistant can use:
- `get_blast_radius` — tiered impact analysis with confidence scores
- `predict_missing_changes` — co-change-based missing file detection
- `suggest_reviewers` — ownership-based reviewer ranking

## Install

From Cursor Marketplace: search "HyperRetrieval"

Or manually:
```bash
# Clone the repo
git clone https://github.com/amitshukla-stack/hyper-retrieval.git

# Add to .cursor/mcp.json
{
  "mcpServers": {
    "hyperretrieval": {
      "command": "python3",
      "args": ["path/to/hyper-retrieval/plugins/cursor/server.py"]
    }
  }
}
```

## Requirements

- Python 3.10+
- `pip install mcp[cli] networkx ijson`
- A git repository

## Privacy

All analysis runs locally. No code leaves your machine.
