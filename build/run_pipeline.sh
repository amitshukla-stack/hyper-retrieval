#!/bin/bash
# Full pipeline orchestrator — run this once on the build machine.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")/repo}"
export REPO_ROOT

VENV="$SCRIPT_DIR/.venv"
PY="$VENV/bin/python3"

echo "============================================"
echo " Codebase Mind Map — Build Pipeline"
echo "============================================"
echo " Repo root:  $REPO_ROOT"
echo " LM Studio:  ${LM_STUDIO_URL:-http://172.18.0.1:1234/v1}"
echo " Python:     $PY"
echo "============================================"
echo ""

# Create venv if needed
if [ ! -f "$PY" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv "$VENV"
fi

echo "[setup] Installing dependencies..."
"$VENV/bin/pip" install -q \
  networkx python-louvain sentence-transformers \
  lancedb pyarrow openai torch chainlit

echo ""
echo "[stage 1] Extracting symbols from source + git..."
time "$PY" 01_extract.py
echo ""

echo "[stage 2] Building graph + Louvain clustering..."
time "$PY" 02_build_graph.py
echo ""

echo "[stage 3] GPU embedding → LanceDB..."
time "$PY" 03_embed.py
echo ""

echo "[stage 4] Cluster summarisation via LM Studio..."
time "$PY" 04_summarize.py
echo ""

echo "[stage 5] Packaging demo artifact..."
"$PY" 05_package.py
echo ""

echo "[model] Downloading GGUF for MacBook demo..."
bash demo_artifact/download_model.sh
echo ""

echo "============================================"
echo " Pipeline complete!"
echo " Transfer: copy demo_artifact/ to MacBook"
echo " Demo:     cd demo_artifact && bash run_demo.sh"
echo "============================================"
