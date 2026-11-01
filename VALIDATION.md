# Validation

CPU checks cover known vector geometry, zero vectors, ties, scale invariance, large finite inputs, noncontiguous arrays, query batching, index ownership, and invalid inputs.

GPU parity checks cover odd dimensions, partial tiles, zero rows, and batches smaller than the query collection. They require CuPy and an NVIDIA CUDA device. No GPU execution or performance result is asserted for an environment without those prerequisites.

Reproduce:

```sh
python -m unittest discover -s tests -v
python -m vector_rank.cli --documents 257 --queries 17 --dimensions 35 --batch-size 6
```

## Development check (November 1, 2026)

Python 3.13 / NumPy 2.3.2 on Windows: all 10 CPU tests passed. The CUDA parity test class was skipped because CuPy was not installed. The 257-document, 17-query, 35-dimension demo completed with CPU-reference agreement. CUDA compilation and device execution remain unverified.
