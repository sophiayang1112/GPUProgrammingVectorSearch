import unittest
import numpy as np
from vector_rank import VectorIndex


class SearchTests(unittest.TestCase):
    def test_known_geometry(self):
        ids, scores = VectorIndex([[1, 0], [0, 1], [-1, 0]]).search([[1, 0]], 3)
        np.testing.assert_array_equal(ids, [[0, 1, 2]])
        np.testing.assert_allclose(scores, [[1, 0, -1]])

    def test_zero_vectors_and_ties(self):
        ids, scores = VectorIndex([[0, 0], [1, 0], [1, 0]]).search([[0, 0]], 3)
        np.testing.assert_array_equal(ids, [[0, 1, 2]])
        np.testing.assert_array_equal(scores, [[0, 0, 0]])

    def test_scaled_vectors(self):
        ids, scores = VectorIndex([[3, 4], [30, 40]]).search([[6, 8]], 2)
        np.testing.assert_array_equal(ids, [[0, 1]])
        np.testing.assert_allclose(scores, [[1, 1]], atol=1e-6)

    def test_large_finite_values(self):
        _, scores = VectorIndex([[3e38, 3e38]]).search([[3e38, 3e38]], 1)
        np.testing.assert_allclose(scores, [[1]], atol=1e-6)

    def test_noncontiguous_input(self):
        data = np.arange(60, dtype=np.float32).reshape(6, 10)[:, ::2]
        a = VectorIndex(data).search(data, 3)
        b = VectorIndex(data.copy()).search(data.copy(), 3)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_batching(self):
        rng = np.random.default_rng(8)
        docs, queries = rng.normal(size=(23, 19)), rng.normal(size=(7, 19))
        index = VectorIndex(docs)
        a, b = index.search(queries, 4, 1), index.search(queries, 4, 5)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1], atol=1e-6)

    def test_index_owns_input(self):
        docs = np.eye(2, dtype=np.float32)
        index = VectorIndex(docs)
        docs[:] = 0
        ids, _ = index.search([[0, 1]], 1)
        self.assertEqual(ids[0, 0], 1)

    def test_invalid_documents(self):
        for docs in ([], [1, 2], [[np.nan]], [[np.inf]], [[1e100]]):
            with self.subTest(docs=docs), self.assertRaises(ValueError):
                VectorIndex(docs)

    def test_invalid_search(self):
        index = VectorIndex([[1, 2]])
        for kwargs in ({"k": 0}, {"k": 2}, {"k": True}, {"batch_size": 0}, {"batch_size": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                index.search([[1, 2]], **kwargs)
        with self.assertRaises(ValueError):
            index.search([[1]], 1)
        with self.assertRaises(ValueError):
            index.search([[np.nan, 1]], 1)

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            VectorIndex([[1]], "unknown")


class CudaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest("No CUDA device")
        except ImportError:
            raise unittest.SkipTest("CuPy not installed")

    def test_non_tile_aligned_and_zero_rows(self):
        rng = np.random.default_rng(12)
        for n, d, q in ((23, 19, 7), (1, 1, 1), (33, 257, 17)):
            docs = rng.normal(size=(n, d)).astype(np.float32)
            queries = rng.normal(size=(q, d)).astype(np.float32)
            docs[0] = 0
            queries[0] = 0
            cpu = VectorIndex(docs).search(queries, min(n, 5), 3)
            gpu = VectorIndex(docs, "cuda").search(queries, min(n, 5), 3)
            np.testing.assert_array_equal(cpu[0], gpu[0])
            np.testing.assert_allclose(cpu[1], gpu[1], atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
