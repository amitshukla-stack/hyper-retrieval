# PyPI Publish Guide

Package: `ripple` v0.6.0  
Build: `ripple-0.6.0-py3-none-any.whl` (pre-built in `dist/`)

## One-command publish (once Amit provides token)

```bash
cd ~/projects/hyperretrieval

# Step 1: Install twine if needed
/home/beast/miniconda3/bin/pip install twine

# Step 2: Rebuild wheel (in case code changed)
cd /tmp && /home/beast/miniconda3/bin/python3 -m build /home/beast/projects/hyperretrieval --wheel --no-isolation
mv /tmp/dist/ripple-*.whl ~/projects/hyperretrieval/dist/

# Step 3: Upload
cd ~/projects/hyperretrieval
/home/beast/miniconda3/bin/python3 -m twine upload dist/ripple-0.6.0-py3-none-any.whl \
  --username __token__ \
  --password <PYPI_TOKEN_HERE>
```

## After publish

1. Verify: `pip install ripple` on a clean machine
2. Update `.mcp-registry/server.json` — change `install.pip` from `hyperretrieval` to `ripple`
3. Submit MCP Registry PR at https://github.com/github/github-mcp-server (or Cursor marketplace)

## Rename note

The package name is `ripple` (not `hyperretrieval`). If `ripple` is taken on PyPI, fallback options:
- `ripple-code` 
- `hyperripple`
- `hr-mcp`

Check availability: `pip index versions ripple`
