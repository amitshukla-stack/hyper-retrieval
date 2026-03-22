"""
Stage 5 — Package the demo artifact.
Copies everything needed into demo_artifact/ and writes helper files.
Also prints GGUF download instructions.
Input:  pipeline/output/graph_with_summaries.json
        pipeline/demo_artifact/vectors.lance/
Output: pipeline/demo_artifact/  (complete, self-contained)
"""
import json, pathlib, shutil, sys, os

PIPELINE_DIR = pathlib.Path(__file__).parent
OUT_DIR      = PIPELINE_DIR / "output"
ARTIFACT_DIR = PIPELINE_DIR / "demo_artifact"
ARTIFACT_DIR.mkdir(exist_ok=True)


REQUIREMENTS_DEMO = """\
# Demo machine requirements (MacBook Air M4 16GB)
# Install: pip install -r requirements_demo.txt
chainlit>=1.0.0
llama-cpp-python>=0.2.0
lancedb>=0.5.0
sentence-transformers>=2.7.0
networkx>=3.0
pyarrow>=14.0
openai>=1.0.0
"""

REQUIREMENTS_BUILD = """\
# Build machine requirements (RTX 5090 / CUDA)
sentence-transformers>=2.7.0
lancedb>=0.5.0
pyarrow>=14.0
networkx>=3.0
python-louvain>=0.16
openai>=1.0.0
torch>=2.0.0
chainlit>=1.0.0
llama-cpp-python>=0.2.0
"""

RUN_DEMO_SH = """\
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Codebase Mind Map Demo ==="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found. Install Python 3.10+."
  exit 1
fi

# Install deps if needed
if ! python3 -c "import chainlit" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -q -r requirements_demo.txt
fi

# Check model
if [ ! -f "model.gguf" ]; then
  echo "WARNING: model.gguf not found."
  echo "Download it with:"
  echo "  huggingface-cli download Qwen/Qwen2.5-Coder-3B-Instruct-GGUF qwen2.5-coder-3b-instruct-q8_0.gguf --local-dir ."
  echo "  mv qwen2.5-coder-3b-instruct-q8_0.gguf model.gguf"
  echo ""
  echo "Continuing without local model (LM Studio mode disabled)..."
fi

echo "Starting demo server..."
echo "Opening http://localhost:8000"
echo ""

# Open browser (works on macOS)
(sleep 3 && open http://localhost:8000) &

chainlit run demo_server.py --port 8000
"""

DOWNLOAD_GGUF_SH = """\
#!/bin/bash
# Run this on the BUILD MACHINE to download the GGUF model into demo_artifact/
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_FILE="model.gguf"
HF_REPO="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
HF_FILE="qwen2.5-coder-3b-instruct-q8_0.gguf"

if [ -f "$MODEL_FILE" ]; then
  echo "model.gguf already exists ($(du -h $MODEL_FILE | cut -f1)). Skipping."
  exit 0
fi

echo "Downloading $HF_FILE from $HF_REPO..."

if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir . --local-dir-use-symlinks False
  mv "$HF_FILE" "$MODEL_FILE"
else
  pip install -q huggingface_hub
  python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='$HF_REPO', filename='$HF_FILE')
shutil.copy(path, '$MODEL_FILE')
print(f'Saved to $MODEL_FILE')
"
fi

echo "Done: $(du -h $MODEL_FILE | cut -f1)"
"""


def main():
    print("=== Stage 5: Packaging demo artifact ===")

    # 1. graph_with_summaries.json (may already be there from stage 4)
    src = OUT_DIR / "graph_with_summaries.json"
    dst = ARTIFACT_DIR / "graph_with_summaries.json"
    if src.exists() and (not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime):
        shutil.copy2(src, dst)
        print(f"  Copied graph_with_summaries.json ({src.stat().st_size // 1024}KB)")
    elif dst.exists():
        print(f"  graph_with_summaries.json already in artifact ({dst.stat().st_size // 1024}KB)")
    else:
        print("  ERROR: graph_with_summaries.json not found. Run stage 4 first.")
        sys.exit(1)

    # 2. vectors.lance — already written by stage 3
    lance_dir = ARTIFACT_DIR / "vectors.lance"
    if lance_dir.exists():
        size_mb = sum(f.stat().st_size for f in lance_dir.rglob("*") if f.is_file()) // (1024*1024)
        print(f"  vectors.lance present ({size_mb}MB)")
    else:
        print("  ERROR: vectors.lance not found. Run stage 3 first.")
        sys.exit(1)

    # 3. demo_server.py — copy from pipeline root if not already in artifact
    server_src = PIPELINE_DIR / "demo_server.py"
    server_dst = ARTIFACT_DIR / "demo_server.py"
    shutil.copy2(server_src, server_dst)
    print("  Copied demo_server.py")

    # 4. Write requirements and shell scripts
    (ARTIFACT_DIR / "requirements_demo.txt").write_text(REQUIREMENTS_DEMO)
    (ARTIFACT_DIR / "requirements_build.txt").write_text(REQUIREMENTS_BUILD)
    print("  Wrote requirements_demo.txt")

    run_sh = ARTIFACT_DIR / "run_demo.sh"
    run_sh.write_text(RUN_DEMO_SH)
    run_sh.chmod(0o755)
    print("  Wrote run_demo.sh")

    dl_sh = ARTIFACT_DIR / "download_model.sh"
    dl_sh.write_text(DOWNLOAD_GGUF_SH)
    dl_sh.chmod(0o755)
    print("  Wrote download_model.sh")

    # 5. Print model status
    model_path = ARTIFACT_DIR / "model.gguf"
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"  model.gguf present ({size_gb:.1f}GB)")
    else:
        print("\n  [ACTION REQUIRED] model.gguf not yet downloaded.")
        print("  Run: bash pipeline/demo_artifact/download_model.sh")

    # 6. Summary
    total_size = sum(
        f.stat().st_size for f in ARTIFACT_DIR.rglob("*") if f.is_file()
    )
    print(f"\n✓ demo_artifact/ is ready ({total_size / (1024**2):.0f}MB without model)")
    print(f"\n  Contents:")
    for f in sorted(ARTIFACT_DIR.iterdir()):
        if f.is_file():
            print(f"    {f.name:40s} {f.stat().st_size // 1024:6d}KB")
        elif f.is_dir():
            sz = sum(x.stat().st_size for x in f.rglob("*") if x.is_file())
            print(f"    {f.name + '/':40s} {sz // 1024:6d}KB")

    print(f"\n  Transfer to MacBook: copy the entire demo_artifact/ folder")
    print(f"  Demo command: cd demo_artifact && bash run_demo.sh")


if __name__ == "__main__":
    main()
