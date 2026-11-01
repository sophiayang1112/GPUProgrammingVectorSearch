"""Reproducible synthetic retrieval demonstration and end-to-end benchmark."""
import argparse
import json
import time
import numpy as np
from .search import VectorIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.documents, args.queries, args.dimensions, args.repeats, args.batch_size) < 1:
        parser.error("sizes and repeats must be positive")
    if not 1 <= args.k <= args.documents:
        parser.error("k must be between 1 and document count")
    rng = np.random.default_rng(42)
    documents = rng.standard_normal((args.documents, args.dimensions), dtype=np.float32)
    queries = documents[np.arange(args.queries) % args.documents].copy()
    queries += 0.01 * rng.standard_normal(queries.shape, dtype=np.float32)
    index = VectorIndex(documents, args.backend)
    reference = VectorIndex(documents)
    expected_ids, expected_scores = reference.search(queries, args.k, args.batch_size)
    ids, scores = index.search(queries, args.k, args.batch_size)  # warm-up / JIT
    np.testing.assert_allclose(scores, expected_scores, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(ids, expected_ids)
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        index.search(queries, args.k, args.batch_size)
        times.append(time.perf_counter() - start)
    median = float(np.median(times))
    print(json.dumps({"backend": args.backend, "documents": args.documents,
        "queries": args.queries, "dimensions": args.dimensions,
        "batch_size": args.batch_size, "repeats": args.repeats,
        "median_seconds": median, "queries_per_second": args.queries / median,
        "cpu_reference_match": True, "first_query_ids": ids[0].tolist(),
        "first_query_scores": scores[0].tolist()}, indent=2))


if __name__ == "__main__":
    main()
