"""Unit tests for face matcher (cosine similarity + top-K search)."""
from __future__ import annotations

import numpy as np
import pytest

from src.ml.faces.matcher import cosine_similarity, find_matches


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)  # L2 normalize
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors(self):
        v = np.random.randn(512).astype(np.float32)
        v = v / np.linalg.norm(v)
        assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_returns_float(self):
        v = np.ones(512, dtype=np.float32) / np.sqrt(512)
        result = cosine_similarity(v, v)
        assert isinstance(result, float)


class TestFindMatches:
    @pytest.fixture
    def database(self) -> np.ndarray:
        """Create a small normalized database of 10 embeddings."""
        rng = np.random.default_rng(42)
        db = rng.standard_normal((10, 512)).astype(np.float32)
        norms = np.linalg.norm(db, axis=1, keepdims=True)
        return db / norms

    def test_empty_database(self):
        query = np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)
        db = np.empty((0, 512), dtype=np.float32)
        results = find_matches(query, db, threshold=0.0)
        assert results == []

    def test_finds_exact_match(self, database: np.ndarray):
        """Querying with a vector from the database should return it as top match."""
        query = database[3].copy()
        results = find_matches(query, database, threshold=0.5, top_k=5)
        assert len(results) > 0
        assert results[0]["index"] == 3
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)

    def test_respects_threshold(self, database: np.ndarray):
        query = database[0].copy()
        results = find_matches(query, database, threshold=0.99, top_k=10)
        # Only the exact match (index 0) should pass threshold 0.99
        assert all(r["score"] >= 0.99 for r in results)

    def test_respects_top_k(self, database: np.ndarray):
        query = database[0].copy()
        results = find_matches(query, database, threshold=0.0, top_k=3)
        assert len(results) <= 3

    def test_sorted_descending(self, database: np.ndarray):
        query = database[0].copy()
        results = find_matches(query, database, threshold=0.0, top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_format(self, database: np.ndarray):
        query = database[0].copy()
        results = find_matches(query, database, threshold=0.0, top_k=1)
        assert len(results) >= 1
        assert "index" in results[0]
        assert "score" in results[0]
        assert isinstance(results[0]["index"], int)
        assert isinstance(results[0]["score"], float)

    def test_high_threshold_returns_empty(self, database: np.ndarray):
        """A threshold of 2.0 (impossible for cosine) should return nothing."""
        query = np.random.randn(512).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = find_matches(query, database, threshold=2.0, top_k=10)
        assert results == []
