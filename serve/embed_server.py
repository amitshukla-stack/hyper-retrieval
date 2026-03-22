"""
embed_server.py — Shared embedding service (local GPU or cloud provider)

Serves a unified HTTP interface so all other servers (Chainlit, MCP) never
load an embedding model themselves.  Switch providers via env vars — no code
changes needed elsewhere.

PROVIDERS
  local   (default) — Qwen3-Embedding-8B loaded on GPU/CPU
  openai            — OpenAI text-embedding-3-large (3072d) or -small (1536d)
  cohere            — Cohere embed-english-v3.0 (1024d)
  voyage            — Voyage voyage-code-3 (1024d) — best for code
  jina              — Jina jina-embeddings-v3 (1024d)
  ollama            — any Ollama model, e.g. nomic-embed-text (768d)

IMPORTANT: embedding dimension depends on the provider/model.
  If you change providers you must re-run build/03_embed.py to rebuild vectors.lance.

STARTUP (choose one):

  # Local GPU (default):
  EMBED_MODEL=/path/to/qwen3-embed-8b python3 embed_server.py

  # OpenAI:
  EMBED_PROVIDER=openai OPENAI_API_KEY=sk-... python3 embed_server.py

  # Voyage (recommended for code, no GPU needed):
  EMBED_PROVIDER=voyage VOYAGE_API_KEY=pa-... python3 embed_server.py

  # Ollama (fully local, no GPU required):
  EMBED_PROVIDER=ollama EMBED_PROVIDER_MODEL=nomic-embed-text python3 embed_server.py

Then set in all other processes:
  export EMBED_SERVER_URL=http://localhost:8001

API:
  POST /embed    {"texts": ["q1", ...], "instruction": "optional"}
                 → {"embeddings": [[f32,...]], "model": "...", "dim": N}
  GET  /health   → {"status": "ok", "provider": "...", "model": "...", "dim": N, "loaded": true}
  GET  /status   → same as /health plus request count
"""
import json, os, pathlib, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "local").lower()
PORT = int(os.environ.get("EMBED_SERVER_PORT", "8001"))
HOST = os.environ.get("EMBED_SERVER_HOST", "127.0.0.1")

# Local model path (only used when EMBED_PROVIDER=local)
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL",
    str(pathlib.Path(__file__).parent / "models" / "qwen3-embed-8b")
)

# Provider-specific model override
EMBED_PROVIDER_MODEL = os.environ.get("EMBED_PROVIDER_MODEL", "")

# API keys
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY",  "")
COHERE_API_KEY  = os.environ.get("COHERE_API_KEY",  "")
VOYAGE_API_KEY  = os.environ.get("VOYAGE_API_KEY",  "")
JINA_API_KEY    = os.environ.get("JINA_API_KEY",    "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Default models per provider
_PROVIDER_DEFAULTS = {
    "local":  (os.path.basename(EMBED_MODEL), 4096),
    "openai": ("text-embedding-3-large",      3072),
    "cohere": ("embed-english-v3.0",          1024),
    "voyage": ("voyage-code-3",               1024),
    "jina":   ("jina-embeddings-v3",          1024),
    "ollama": ("nomic-embed-text",             768),
}

_default_model, _default_dim = _PROVIDER_DEFAULTS.get(EMBED_PROVIDER, ("unknown", 0))
_model_name = EMBED_PROVIDER_MODEL or _default_model
_embed_dim  = _default_dim

# Instruction prefix — used only for local Qwen3 (providers have native query/doc types)
DEFAULT_INSTRUCTION = (
    "Instruct: Represent this code module for finding semantically similar "
    "components across microservices. Query: "
)

# ── State ──────────────────────────────────────────────────────────────────────
_local_model   = None
_device        = None
_client        = None        # API client for cloud providers
_lock          = Lock()
_request_count = 0
_loaded_at     = None


# ── Backend initialisation ─────────────────────────────────────────────────────

def _init_local():
    global _local_model, _device, _loaded_at, _embed_dim
    import torch
    from sentence_transformers import SentenceTransformer

    _device = "cuda" if torch.cuda.is_available() \
         else "mps"  if torch.backends.mps.is_available() \
         else "cpu"

    print(f"[embed_server] Loading {_model_name} on {_device}...")
    _local_model = SentenceTransformer(
        EMBED_MODEL, device=_device, trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16} if _device == "cuda" else {},
    )
    warmup = _local_model.encode(["warmup"], normalize_embeddings=True)
    _embed_dim = warmup.shape[1]
    _loaded_at = time.time()
    print(f"[embed_server] Ready — {_model_name} @ {_embed_dim}d on {_device}")


def _init_openai():
    global _client, _loaded_at, _embed_dim
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for EMBED_PROVIDER=openai")
    _client = OpenAI(api_key=OPENAI_API_KEY)
    # Probe dimension with a test call
    resp = _client.embeddings.create(input=["test"], model=_model_name)
    _embed_dim = len(resp.data[0].embedding)
    _loaded_at = time.time()
    print(f"[embed_server] Ready — OpenAI {_model_name} @ {_embed_dim}d")


def _init_cohere():
    global _client, _loaded_at, _embed_dim
    import cohere
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is required for EMBED_PROVIDER=cohere")
    _client = cohere.Client(COHERE_API_KEY)
    resp = _client.embed(texts=["test"], model=_model_name,
                         input_type="search_query")
    _embed_dim = len(resp.embeddings[0])
    _loaded_at = time.time()
    print(f"[embed_server] Ready — Cohere {_model_name} @ {_embed_dim}d")


def _init_voyage():
    global _client, _loaded_at, _embed_dim
    import voyageai
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY is required for EMBED_PROVIDER=voyage")
    _client = voyageai.Client(api_key=VOYAGE_API_KEY)
    resp = _client.embed(["test"], model=_model_name, input_type="query")
    _embed_dim = len(resp.embeddings[0])
    _loaded_at = time.time()
    print(f"[embed_server] Ready — Voyage {_model_name} @ {_embed_dim}d")


def _init_jina():
    global _loaded_at, _embed_dim
    import requests as _req
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY is required for EMBED_PROVIDER=jina")
    resp = _req.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}"},
        json={"input": ["test"], "model": _model_name},
        timeout=30,
    )
    resp.raise_for_status()
    _embed_dim = len(resp.json()["data"][0]["embedding"])
    _loaded_at = time.time()
    print(f"[embed_server] Ready — Jina {_model_name} @ {_embed_dim}d")


def _init_ollama():
    global _loaded_at, _embed_dim
    import requests as _req
    resp = _req.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": _model_name, "prompt": "test"},
        timeout=60,
    )
    resp.raise_for_status()
    _embed_dim = len(resp.json()["embedding"])
    _loaded_at = time.time()
    print(f"[embed_server] Ready — Ollama {_model_name} @ {_embed_dim}d (no GPU needed)")


def init_backend():
    """Load/connect the configured embedding backend."""
    init_fn = {
        "local":  _init_local,
        "openai": _init_openai,
        "cohere": _init_cohere,
        "voyage": _init_voyage,
        "jina":   _init_jina,
        "ollama": _init_ollama,
    }.get(EMBED_PROVIDER)
    if init_fn is None:
        raise RuntimeError(
            f"Unknown EMBED_PROVIDER={EMBED_PROVIDER!r}. "
            f"Valid: local, openai, cohere, voyage, jina, ollama"
        )
    init_fn()


# ── Encode dispatch ────────────────────────────────────────────────────────────

def _encode(texts: list[str], instruction: str) -> list[list[float]]:
    """Encode texts using the configured backend. Returns list of float lists."""

    if EMBED_PROVIDER == "local":
        import numpy as np
        prefixed = [instruction + t for t in texts]
        with _lock:
            vecs = _local_model.encode(
                prefixed, normalize_embeddings=True, convert_to_numpy=True
            )
        return vecs.tolist()

    if EMBED_PROVIDER == "openai":
        # OpenAI distinguishes query vs document via the text itself — no prefix needed
        resp = _client.embeddings.create(input=texts, model=_model_name)
        return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]

    if EMBED_PROVIDER == "cohere":
        # Use search_query for single texts (query-time), search_document for batches (index-time)
        input_type = "search_query" if len(texts) == 1 else "search_document"
        resp = _client.embed(texts=texts, model=_model_name, input_type=input_type)
        return resp.embeddings

    if EMBED_PROVIDER == "voyage":
        input_type = "query" if len(texts) == 1 else "document"
        resp = _client.embed(texts, model=_model_name, input_type=input_type)
        return resp.embeddings

    if EMBED_PROVIDER == "jina":
        import requests as _req
        resp = _req.post(
            "https://api.jina.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {JINA_API_KEY}"},
            json={"input": texts, "model": _model_name},
            timeout=60,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [d["embedding"] for d in data]

    if EMBED_PROVIDER == "ollama":
        import requests as _req
        results = []
        for text in texts:
            resp = _req.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": _model_name, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results

    raise RuntimeError(f"No encoder for provider {EMBED_PROVIDER!r}")


# ── HTTP handler ───────────────────────────────────────────────────────────────

class EmbedHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass   # suppress noisy access log

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
        if self.path in ("/health", "/status"):
            self._send_json({
                "status":    "ok" if _loaded_at else "loading",
                "provider":  EMBED_PROVIDER,
                "model":     _model_name,
                "device":    _device or "api",
                "dim":       _embed_dim,
                "loaded":    _loaded_at is not None,
                "loaded_at": _loaded_at,
                "requests":  _request_count,
            })
        else:
            self._send_error("Not found", 404)

    def do_POST(self):
        global _request_count
        if self.path != "/embed":
            self._send_error("Not found", 404)
            return
        if _loaded_at is None:
            self._send_error("Backend not ready yet", 503)
            return
        try:
            length  = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length))
        except Exception as e:
            self._send_error(f"Bad request: {e}")
            return

        texts       = payload.get("texts", [])
        instruction = payload.get("instruction", DEFAULT_INSTRUCTION)

        if not texts or not isinstance(texts, list):
            self._send_error("'texts' must be a non-empty list of strings")
            return

        try:
            embeddings = _encode(texts, instruction)
            _request_count += 1
            self._send_json({
                "embeddings": embeddings,
                "model":      _model_name,
                "provider":   EMBED_PROVIDER,
                "dim":        len(embeddings[0]) if embeddings else _embed_dim,
                "count":      len(texts),
            })
        except Exception as e:
            self._send_error(f"Encode error: {e}", 500)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"[embed_server] Provider: {EMBED_PROVIDER} | Model: {_model_name}")
    init_backend()
    server = HTTPServer((HOST, PORT), EmbedHandler)
    print(f"[embed_server] Listening on http://{HOST}:{PORT}")
    print(f"[embed_server] Set EMBED_SERVER_URL=http://{HOST}:{PORT} in other processes")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[embed_server] Shutting down")


if __name__ == "__main__":
    main()
