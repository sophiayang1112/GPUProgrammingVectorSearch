# Vector Rank Lab

Exact cosine-similarity retrieval for dense embeddings, with a NumPy reference implementation and custom CUDA kernels. Build an index, submit a batch of query vectors, and retrieve ranked document IDs and scores.

This independently implemented portfolio project explores GPU numerical programming in a practical retrieval workload. It uses synthetic vectors and contains no external datasets or model credentials.

## What it demonstrates

- Row-wise reductions and normalization, including a defined zero-vector policy.
- A shared-memory tiled matrix product computing query/document similarities.
- Elementwise arithmetic, transposition, slicing, and batched execution.
- Deterministic ranking for equal scores and validation against a CPU reference.
- Input ownership, finite-value checks, and explicit CPU/GPU execution paths.

## Quick start

```sh
python -m pip install -e .
python -m vector_rank.cli
python -m unittest discover -s tests -v
```

The default demo creates 2,000 synthetic document vectors and 32 noisy queries with 128 dimensions. It checks results against the CPU reference before reporting measured end-to-end search latency. No pretrained embedding model is bundled.

```python
from vector_rank import VectorIndex

index = VectorIndex([[1, 0], [0, 1], [-1, 0]])
ids, scores = index.search([[1, 0]], k=2)
# ids: [[0, 1]], scores: [[1.0, 0.0]]
```

## Optional NVIDIA GPU execution

Install a CuPy distribution that matches your supported NVIDIA driver and CUDA runtime, for example `cupy-cuda12x` for a compatible CUDA 12 environment. Consult [CuPy installation guidance](https://docs.cupy.dev/en/stable/install.html) for your platform.

```sh
python -m vector_rank.cli --backend cuda --documents 10000 --queries 64 --dimensions 256
```

CuPy compiles the included `kernels.cu` through its runtime compiler. CUDA mode fails explicitly if GPU support is unavailable. The test suite includes optional GPU parity tests; skips do not establish GPU correctness.

## How retrieval works

1. Copy and validate document vectors as contiguous float32 arrays.
2. Normalize each row; zero vectors remain zero. Norm accumulation uses float64.
3. Process queries in bounded batches and multiply normalized queries by normalized document rows.
4. Copy GPU scores to the host when applicable, then perform a stable descending sort.
5. Return the top `k` IDs and cosine scores; equal scores retain ascending document ID.

The CUDA normalization kernel uses a 256-thread shared-memory reduction. The dot-product kernel uses 16-by-16 tiles and handles dimensions that are not tile multiples.

## Tradeoffs and boundaries

This is exhaustive retrieval, not an approximate nearest-neighbor database. Arithmetic costs O(QND); a score batch uses O(BN) memory, and full ranking costs O(B N log N). All document vectors remain resident on the selected device. Query batching limits score storage but does not make arbitrarily large document collections fit in GPU memory.

Ranking runs on the CPU. Benchmark timings include query normalization, dot products, GPU-to-host transfer when used, and ranking; they exclude index construction and initial kernel compilation. CPU mode may be faster for small workloads. No GPU speedup is claimed without measurements on specified hardware. Float32 arithmetic can reorder nearly tied scores across devices; exact ties on a given backend use document ID order.

No automatic differentiation, embedding generation, persistent database, distributed execution, or production-serving layer is included. See [validation notes](VALIDATION.md) for checks performed in the development environment.
