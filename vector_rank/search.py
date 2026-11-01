"""Batched exact retrieval; zero vectors have zero similarity to every vector."""
from pathlib import Path
import numpy as np


def _matrix(values, name):
    with np.errstate(over="ignore", invalid="ignore"):
        data = np.array(values, dtype=np.float32, order="C", copy=True)
    if data.ndim != 2 or min(data.shape) < 1:
        raise ValueError(f"{name} must be a nonempty 2-D matrix")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must contain finite float32 values")
    if max(data.shape) > np.iinfo(np.int32).max:
        raise ValueError("matrix dimensions exceed kernel limits")
    return data


def _normalize(data):
    # Accumulate in float64 so finite float32 inputs cannot overflow the norm.
    norms = np.sqrt(np.sum(data.astype(np.float64) ** 2, axis=1, keepdims=True))
    return np.ascontiguousarray(data / np.where(norms == 0, 1, norms), dtype=np.float32)


class VectorIndex:
    """Immutable dense index with CPU or explicit CUDA execution.

    Ranking is exact exhaustive cosine search. Ties use the smaller document ID.
    CUDA mode never silently falls back to CPU.
    """

    def __init__(self, documents, backend="cpu"):
        data = _matrix(documents, "documents")
        self.count, self.dimensions = data.shape
        self.backend = backend
        if backend == "cpu":
            self._documents = _normalize(data)
        elif backend == "cuda":
            try:
                import cupy as cp
            except ImportError as exc:
                raise RuntimeError("CUDA requires a CuPy package matching your CUDA runtime") from exc
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError("No CUDA device available")
            self._cp = cp
            source = Path(__file__).with_name("kernels.cu").read_text()
            self._norm = cp.RawKernel(source, "normalize_rows")
            self._dot = cp.RawKernel(source, "cosine_tiles")
            self._documents = self._normalize_gpu(cp.asarray(data))
        else:
            raise ValueError("backend must be 'cpu' or 'cuda'")

    def _normalize_gpu(self, data):
        output = self._cp.empty_like(data)
        self._norm((data.shape[0],), (256,),
                   (data, output, np.int32(data.shape[0]), np.int32(data.shape[1])))
        return output

    def search(self, queries, k=5, batch_size=64):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= self.count:
            raise ValueError("k must be an integer between 1 and document count")
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        queries = _matrix(queries, "queries")
        if queries.shape[1] != self.dimensions:
            raise ValueError("query and document dimensions must match")
        ids = np.empty((len(queries), k), dtype=np.int64)
        values = np.empty((len(queries), k), dtype=np.float32)
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            if self.backend == "cpu":
                scores = _normalize(batch) @ self._documents.T
            else:
                cp = self._cp
                normalized = self._normalize_gpu(cp.asarray(batch))
                gpu_scores = cp.empty((len(batch), self.count), dtype=cp.float32)
                self._dot(((self.count + 15) // 16, (len(batch) + 15) // 16), (16, 16),
                          (normalized, self._documents, gpu_scores, np.int32(len(batch)),
                           np.int32(self.count), np.int32(self.dimensions)))
                # Includes device synchronization; ranking is intentionally on the host.
                scores = cp.asnumpy(gpu_scores)
            scores = np.clip(scores, -1.0, 1.0)
            order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            ids[start:start + len(batch)] = order
            values[start:start + len(batch)] = np.take_along_axis(scores, order, axis=1)
        return ids, values
