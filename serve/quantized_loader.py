"""
Quantized vector search backend for HyperRetrieval.

Loads TurboQuant-compressed vectors from vectors_quantized.npz,
reconstructs them, and provides a LanceDB-compatible search interface.

Usage:
    ARTIFACT_DIR=/path/to/artifacts USE_QUANTIZED=1 python retrieval_engine.py

Disk savings: 1.88GB (fp32 lance) -> 312MB (4-bit npz)
Quality: 94.6% recall@10 vs fp32 at 4-bit
"""
import numpy as np
from pathlib import Path


class QuantizedSearchTable:
    """Drop-in replacement for LanceDB table with quantized vectors."""

    def __init__(self, npz_path: str, metadata_df=None):
        """Load quantized vectors and reconstruct.

        Args:
            npz_path: Path to vectors_quantized.npz
            metadata_df: Optional pandas DataFrame with id, name, service, etc.
                         If None, loads metadata from LanceDB (if available).
        """
        data = np.load(npz_path)
        self.indices = data["indices"]  # (n, d) uint8
        self.Pi = data["Pi"]            # (d, d) float32 rotation matrix
        self.centroids = data["centroids"]  # (2^bits,) float32 scalar codebook
        self.norms = data["norms"]      # (n,) float32 original norms
        self.bits = int(data["bits"])
        self.n_vectors = int(data["n_vectors"])
        self.dim = int(data["dim"])
        self.metadata_df = metadata_df

        # Reconstruct vectors: lookup centroids, then unrotate
        import time
        t0 = time.time()
        print(f"  Reconstructing {self.n_vectors:,} vectors from {self.bits}-bit quantized...")

        try:
            import torch
            if torch.cuda.is_available():
                self._reconstruct_gpu(torch)
            else:
                self._reconstruct_cpu()
        except ImportError:
            self._reconstruct_cpu()

        elapsed = time.time() - t0
        size_mb = Path(npz_path).stat().st_size / 1e6
        print(f"  Loaded {self.n_vectors:,} x {self.dim}d from {size_mb:.0f}MB in {elapsed:.1f}s")

    def _reconstruct_gpu(self, torch):
        """GPU-accelerated reconstruction (~5s for 114K vectors)."""
        print(f"  Using CUDA for reconstruction...")
        Pi_t = torch.from_numpy(self.Pi).cuda()
        centroids_t = torch.from_numpy(self.centroids).cuda()
        norms_t = torch.from_numpy(self.norms).cuda()
        idx_t = torch.from_numpy(self.indices.astype(np.int64)).cuda()

        x_rot = centroids_t[idx_t]            # (n, d)
        vectors = x_rot @ Pi_t                 # unrotate
        vectors *= norms_t.unsqueeze(1)        # restore norms

        nrm = vectors.norm(dim=1, keepdim=True)
        nrm[nrm == 0] = 1.0
        vectors_normed = vectors / nrm

        self.vectors = vectors.cpu().numpy()
        self.vectors_normed = vectors_normed.cpu().numpy()

        # Free GPU memory
        del Pi_t, centroids_t, norms_t, idx_t, x_rot, vectors, vectors_normed, nrm
        torch.cuda.empty_cache()

    def _reconstruct_cpu(self):
        """CPU fallback reconstruction (~13min for 114K vectors)."""
        print(f"  Using CPU for reconstruction (slow — use GPU if available)...")
        x_rotated = self.centroids[self.indices.astype(np.int64)]
        self.vectors = x_rotated @ self.Pi
        self.vectors *= self.norms[:, np.newaxis]

        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors_normed = self.vectors / norms

    def __len__(self):
        return self.n_vectors

    def search(self, query_vec):
        """Start a search query. Returns a SearchBuilder for chaining."""
        return _SearchBuilder(self, np.asarray(query_vec, dtype=np.float32))


class _SearchBuilder:
    """Mimics LanceDB's search().limit().to_list() chain."""

    def __init__(self, table: QuantizedSearchTable, query_vec: np.ndarray):
        self._table = table
        q = query_vec.flatten()
        norm = np.linalg.norm(q)
        self._query = q / norm if norm > 0 else q
        self._limit = 10

    def limit(self, k: int):
        self._limit = k
        return self

    def to_list(self):
        # Cosine similarity = dot product of normalized vectors
        sims = self._table.vectors_normed @ self._query  # (n,)
        # LanceDB returns _distance (lower = better), so use 1 - sim
        distances = 1.0 - sims

        top_k_idx = np.argpartition(distances, self._limit)[:self._limit]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

        results = []
        for idx in top_k_idx:
            row = {"_distance": float(distances[idx])}
            if self._table.metadata_df is not None:
                meta = self._table.metadata_df.iloc[idx]
                for col in ["id", "name", "service", "module", "kind",
                            "lang", "cluster", "cluster_name", "file", "text"]:
                    if col in meta.index:
                        row[col] = meta[col]
            results.append(row)
        return results
