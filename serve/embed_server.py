"""
embed_server.py — Shared GPU embedding service

Loads the Qwen3-Embedding-8B model ONCE and serves embeddings over HTTP.
Both the Chainlit server and MCP server call this instead of loading the
model themselves — one GPU load, zero OOM conflicts.

Start it first, before any other service:
  PYTHONUNBUFFERED=1 /home/beast/miniconda3/bin/python3 embed_server.py

Then set in other processes:
  export EMBED_SERVER_URL=http://localhost:8001

The Chainlit server and MCP server will automatically use it when that
env var is set, skipping their own local model load entirely.

API:
  POST /embed    {"texts": ["query1", ...], "instruction": "optional prefix"}
                 → {"embeddings": [[f32, ...], ...], "model": "...", "dim": 4096}
  GET  /health   → {"status": "ok", "model": "...", "device": "cuda", "loaded": true}
  GET  /status   → same as /health plus request count
"""
import json, os, pathlib, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL",
    str(pathlib.Path(__file__).parent / "models" / "qwen3-embed-8b")
)
PORT = int(os.environ.get("EMBED_SERVER_PORT", "8001"))
HOST = os.environ.get("EMBED_SERVER_HOST", "127.0.0.1")

DEFAULT_INSTRUCTION = (
    "Instruct: Represent this code module for finding semantically similar "
    "components across microservices. Query: "
)

# ── State ─────────────────────────────────────────────────────────────────────
_model       = None
_device      = None
_model_name  = os.path.basename(EMBED_MODEL)
_lock        = Lock()   # one encode at a time (GPU is not thread-safe for batching)
_request_count = 0
_loaded_at   = None


def _load_model():
    global _model, _device, _loaded_at
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    print(f"[embed_server] Loading {_model_name} on {_device}...")
    _model = SentenceTransformer(
        EMBED_MODEL, device=_device, trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16} if _device == "cuda" else {},
    )
    _loaded_at = time.time()
    # Warm up with one encode
    _model.encode(["warmup"], normalize_embeddings=True)
    print(f"[embed_server] Ready on {HOST}:{PORT}")


class EmbedHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass   # suppress default access log (too noisy)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg: str, status: int = 400):
        self._send_json({"error": msg}, status)

    def do_GET(self):
        global _request_count
        if self.path in ("/health", "/status"):
            self._send_json({
                "status":  "ok" if _model is not None else "loading",
                "model":   _model_name,
                "device":  _device or "unknown",
                "loaded":  _model is not None,
                "loaded_at": _loaded_at,
                "requests": _request_count,
            })
        else:
            self._send_error("Not found", 404)

    def do_POST(self):
        global _request_count
        if self.path != "/embed":
            self._send_error("Not found", 404)
            return

        if _model is None:
            self._send_error("Model not loaded yet", 503)
            return

        try:
            length  = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
        except Exception as e:
            self._send_error(f"Bad request: {e}")
            return

        texts       = payload.get("texts", [])
        instruction = payload.get("instruction", DEFAULT_INSTRUCTION)

        if not texts:
            self._send_error("'texts' list is required")
            return
        if not isinstance(texts, list):
            self._send_error("'texts' must be a list of strings")
            return

        try:
            with _lock:
                prefixed = [instruction + t for t in texts]
                vecs = _model.encode(
                    prefixed, normalize_embeddings=True, convert_to_numpy=True
                )
            _request_count += 1
            self._send_json({
                "embeddings": vecs.tolist(),
                "model": _model_name,
                "dim":   vecs.shape[1] if vecs.ndim > 1 else len(vecs[0]),
                "count": len(texts),
            })
        except Exception as e:
            self._send_error(f"Encode error: {e}", 500)


def main():
    _load_model()
    server = HTTPServer((HOST, PORT), EmbedHandler)
    print(f"[embed_server] Listening on http://{HOST}:{PORT}")
    print(f"[embed_server] Set EMBED_SERVER_URL=http://{HOST}:{PORT} in other processes")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[embed_server] Shutting down")


if __name__ == "__main__":
    main()
