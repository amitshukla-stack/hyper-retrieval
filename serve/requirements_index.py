"""
Requirement-graph retrieval layer for search_requirements MCP tool (RFC 015).

Loads requirements.lance (vector index) and requirements.json (metadata) at startup.
Provides search_requirements(query, k) → List[RequirementCluster].

A RequirementCluster groups semantically-similar modules that together implement
a behavioral requirement. Designed for "how does X work?" queries, not symbol lookups.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests

# ── Config ───────────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path(os.environ.get(
    "ARTIFACT_DIR",
    os.path.expanduser("~/projects/workspaces/juspay/artifacts")
))
EMBED_SERVER_URL = os.environ.get("EMBED_SERVER_URL", "http://localhost:8001")
CLUSTER_THRESHOLD = float(os.environ.get("REQ_CLUSTER_THRESHOLD", "0.80"))

REQUIREMENTS_JSON = ARTIFACT_DIR / "requirements.json"
REQUIREMENTS_LANCE = ARTIFACT_DIR / "requirements.lance"


# ── Types ─────────────────────────────────────────────────────────────────────
@dataclass
class RequirementCluster:
    requirement: str
    confidence: float
    modules: list[str] = field(default_factory=list)
    key_functions: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "requirement": self.requirement,
            "confidence": round(self.confidence, 3),
            "modules": self.modules,
            "key_functions": self.key_functions,
            "summary": self.summary,
        }


# ── Index state ───────────────────────────────────────────────────────────────
_lance_tbl = None
_req_meta: dict = {}   # module → {requirement, activity_score}
_loaded = False


def _load_lance():
    try:
        import lancedb
        db = lancedb.connect(str(REQUIREMENTS_LANCE.parent))
        return db.open_table(REQUIREMENTS_LANCE.stem)  # "requirements"
    except Exception as e:
        return None


def initialize() -> bool:
    """Load requirements index. Returns True if ready, False if index not built yet."""
    global _lance_tbl, _req_meta, _loaded

    if _loaded:
        return True

    if not REQUIREMENTS_JSON.exists():
        return False

    _req_meta = json.loads(REQUIREMENTS_JSON.read_text())

    # Warn if requirements index is stale relative to the main graph
    graph_path = ARTIFACT_DIR / "graph_with_summaries.json"
    if graph_path.exists():
        req_age = REQUIREMENTS_JSON.stat().st_mtime
        graph_age = graph_path.stat().st_mtime
        if graph_age > req_age + 3600:  # graph rebuilt > 1h after requirements
            import warnings
            warnings.warn(
                "requirements.json is older than graph_with_summaries.json by "
                f"{(graph_age - req_age)/3600:.1f}h — rebuild with build/11_build_requirements.py",
                stacklevel=2,
            )

    if REQUIREMENTS_LANCE.exists():
        _lance_tbl = _load_lance()

    _loaded = True
    return bool(_req_meta)


def is_ready() -> bool:
    return _loaded and bool(_req_meta)


def stats() -> dict:
    return {
        "modules": len(_req_meta),
        "lance_loaded": _lance_tbl is not None,
        "lance_path": str(REQUIREMENTS_LANCE),
        "json_path": str(REQUIREMENTS_JSON),
    }


# ── Embedding ─────────────────────────────────────────────────────────────────
def _embed_query(query: str) -> Optional[list[float]]:
    try:
        r = requests.post(
            f"{EMBED_SERVER_URL}/embed",
            json={"texts": [query]},
            timeout=10
        )
        r.raise_for_status()
        return r.json()["embeddings"][0]
    except Exception:
        return None


# ── Clustering ────────────────────────────────────────────────────────────────
def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _cluster_hits(hits: list[dict], threshold: float) -> list[list[dict]]:
    """Greedy clustering: group hits with cosine similarity ≥ threshold."""
    clusters: list[list[dict]] = []
    for hit in hits:
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            if _cosine_sim(rep["vector"], hit["vector"]) >= threshold:
                cluster.append(hit)
                placed = True
                break
        if not placed:
            clusters.append([hit])
    return clusters


# ── Key functions lookup ───────────────────────────────────────────────────────
def _get_key_functions(module_name: str, n: int = 3) -> list[str]:
    """Retrieve top function names from the graph for a module (best-effort)."""
    try:
        from serve.retrieval_engine import _nodes  # type: ignore
        funcs = [
            node.get("name", "")
            for nid, node in _nodes.items()
            if node.get("module") == module_name and node.get("name")
        ]
        return funcs[:n]
    except Exception:
        return []


# ── Main search ───────────────────────────────────────────────────────────────
def search_requirements(query: str, k: int = 5) -> list[RequirementCluster]:
    """
    Search for functional requirement clusters matching a behavioral query.

    Uses vector similarity on requirements.lance. Falls back to keyword scan
    of requirements.json if lance index not available.
    """
    if not is_ready():
        initialize()
    if not is_ready():
        return []

    if _lance_tbl is not None:
        return _vector_search(query, k)
    else:
        return _keyword_search(query, k)


def _vector_search(query: str, k: int) -> list[RequirementCluster]:
    """Full vector search via requirements.lance using lancedb ANN."""
    qvec = _embed_query(query)
    if qvec is None:
        return _keyword_search(query, k)

    try:
        hits = _lance_tbl.search(qvec).limit(k * 3).to_list()
        # lancedb returns dicts with _distance; convert to scored hits
        scored = []
        for h in hits:
            # _distance is L2 — convert to cosine-like score (lower = better → invert)
            dist = h.get("_distance", 1.0)
            score = max(0.0, 1.0 - dist / 2.0)
            scored.append({
                "name": h["name"],
                "requirement": h["requirement"],
                "vector": h.get("vector", []),
                "score": score,
            })
        top = scored[:k * 3]
    except Exception:
        return _keyword_search(query, k)

    clusters = _cluster_hits(top, CLUSTER_THRESHOLD)
    return _build_clusters(clusters, k)


def _keyword_search(query: str, k: int) -> list[RequirementCluster]:
    """Fallback: simple keyword match on requirement text."""
    terms = set(query.lower().split())
    scored = []
    for module, meta in _req_meta.items():
        req_text = meta.get("requirement", "").lower()
        score = sum(1 for t in terms if t in req_text) / max(len(terms), 1)
        if score > 0:
            scored.append({
                "name": module,
                "requirement": meta["requirement"],
                "vector": [],
                "score": score,
            })
    scored.sort(key=lambda x: -x["score"])
    top = scored[:k * 3]
    clusters = [[h] for h in top]  # no clustering without vectors
    return _build_clusters(clusters, k)


def _build_clusters(raw_clusters: list[list[dict]], k: int) -> list[RequirementCluster]:
    results = []
    for cluster in raw_clusters[:k]:
        rep = cluster[0]
        modules = [h["name"] for h in cluster]
        key_fns = _get_key_functions(modules[0]) if modules else []
        summary = (
            f"{len(modules)} module(s) implement this requirement."
            if len(modules) > 1
            else ""
        )
        results.append(RequirementCluster(
            requirement=rep["requirement"],
            confidence=rep["score"],
            modules=modules,
            key_functions=key_fns,
            summary=summary,
        ))
    return results


def format_for_mcp(clusters: list[RequirementCluster]) -> str:
    """Format clusters as a readable MCP tool response."""
    if not clusters:
        return "No requirement clusters found for this query."

    lines = []
    for i, c in enumerate(clusters, 1):
        lines.append(f"**{i}. {c.requirement}**")
        lines.append(f"   Confidence: {c.confidence:.2f}")
        lines.append(f"   Modules: {', '.join(c.modules)}")
        if c.key_functions:
            lines.append(f"   Key functions: {', '.join(c.key_functions)}")
        if c.summary:
            lines.append(f"   {c.summary}")
        lines.append("")
    return "\n".join(lines).strip()
