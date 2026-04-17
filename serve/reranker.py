"""Pluggable cross-encoder reranker for HyperRetrieval.

Interface: `Reranker.rerank(query, candidates) -> list[(candidate, score)]`
  - `query` is the user's original natural-language question.
  - `candidates` is a list of dicts (nodes from RRF/co-change fusion).
  - Returns the same list in new order plus a float score per node.

Two implementations ship here:
  - `NoopReranker` — identity, score = 0.0. Always safe fallback.
  - `BGEReranker` — wraps `BAAI/bge-reranker-v2-m3` via `sentence-transformers`.
    Loaded lazily on first call so import is cheap; fails silently to Noop
    if the package or weights are unavailable.

The active reranker is chosen via env vars:
  HR_RERANKER           - "bge" | "noop" | "" (default empty = off entirely)
  HR_RERANKER_TOPK      - rerank only top-N candidates (default 50)
  HR_RERANKER_ALPHA     - blend factor vs original rrf score (default 1.0 = fully replaces)
  HR_RERANKER_MODEL     - override model id (default "BAAI/bge-reranker-v2-m3")

`get_reranker()` returns the active instance (or None if HR_RERANKER is off).
"""
from __future__ import annotations
import os
from typing import Callable


def _candidate_text(node: dict) -> str:
    """Build a short text representation of a candidate for the reranker."""
    parts = []
    for key in ("name", "module", "id", "summary", "body", "doc", "signature"):
        v = node.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
            if sum(len(p) for p in parts) > 1200:
                break
    return " ".join(parts)[:1500]


class Reranker:
    name = "base"

    def available(self) -> bool:
        return True

    def rerank(self, query: str, candidates: list) -> list:
        raise NotImplementedError


class NoopReranker(Reranker):
    name = "noop"

    def rerank(self, query: str, candidates: list) -> list:
        return [(c, 0.0) for c in candidates]


class BGEReranker(Reranker):
    name = "bge"

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or os.environ.get(
            "HR_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self._model = None
        self._load_err = None

    def _load(self):
        if self._model is not None or self._load_err is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_id, max_length=512)
        except Exception as e:
            self._load_err = e
            print(f"[reranker] BGE load failed: {e!r} — will fall back to noop")

    def available(self) -> bool:
        self._load()
        return self._model is not None

    def rerank(self, query: str, candidates: list) -> list:
        self._load()
        if self._model is None:
            return [(c, 0.0) for c in candidates]
        pairs = [(query, _candidate_text(c)) for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            print(f"[reranker] BGE inference failed: {e!r}")
            return [(c, 0.0) for c in candidates]
        out = list(zip(candidates, [float(s) for s in scores]))
        out.sort(key=lambda cs: -cs[1])
        return out


_RERANKER_INSTANCE: Reranker | None = None


def get_reranker() -> Reranker | None:
    """Return the active reranker, or None if HR_RERANKER is off."""
    global _RERANKER_INSTANCE
    choice = os.environ.get("HR_RERANKER", "").lower()
    if not choice or choice == "off":
        return None
    if _RERANKER_INSTANCE is not None and _RERANKER_INSTANCE.name == choice:
        return _RERANKER_INSTANCE
    if choice == "bge":
        _RERANKER_INSTANCE = BGEReranker()
    elif choice == "noop":
        _RERANKER_INSTANCE = NoopReranker()
    else:
        _RERANKER_INSTANCE = NoopReranker()
    return _RERANKER_INSTANCE


def apply_reranker(query: str, merged: dict,
                    reranker: Reranker | None = None,
                    score_getter: Callable[[dict], float] | None = None) -> dict:
    """Rerank the top-K candidates in each service bucket of a merged result dict.

    - `merged` is {service: [node, ...]}  (output of rrf_merge or criticality boost).
    - Only the top `HR_RERANKER_TOPK` candidates (by existing RRF score) are rescored.
    - The final score is `alpha * rerank_score + (1-alpha) * original_rrf_score`.
    - Nodes get `_rerank_score` and `_final_rank_score` stamped for observability.
    """
    rr = reranker if reranker is not None else get_reranker()
    if rr is None or not rr.available():
        return merged
    try:
        topk = int(os.environ.get("HR_RERANKER_TOPK", "50"))
    except ValueError:
        topk = 50
    try:
        alpha = float(os.environ.get("HR_RERANKER_ALPHA", "1.0"))
    except ValueError:
        alpha = 1.0
    alpha = max(0.0, min(1.0, alpha))

    _score = score_getter or (lambda n: n.get("_rrf_boosted", n.get("_rrf_score", 0.0)))
    out: dict = {}
    for svc, nodes in merged.items():
        if not nodes:
            out[svc] = nodes
            continue
        head = nodes[:topk]
        tail = nodes[topk:]
        scored = rr.rerank(query, head)
        rescored = []
        for node, rr_score in scored:
            orig = float(_score(node))
            final = alpha * float(rr_score) + (1.0 - alpha) * orig
            n2 = {**node, "_rerank_score": float(rr_score), "_final_rank_score": final}
            rescored.append(n2)
        rescored.sort(key=lambda n: -n.get("_final_rank_score", 0.0))
        out[svc] = rescored + tail
    return out
