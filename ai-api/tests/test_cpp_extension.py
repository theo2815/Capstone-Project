"""Tests for the _eventai_cpp C++ extension module.

Skipped automatically if the extension is not built.
"""
from __future__ import annotations

import numpy as np
import pytest

cpp = pytest.importorskip("_eventai_cpp", reason="C++ extension not built")


# ──────────────────────────────────────────────────────────────
# face_ops
# ──────────────────────────────────────────────────────────────

class TestCosineSimlarity:
    def test_identical_vectors(self):
        v = np.random.randn(512).astype(np.float32)
        v /= np.linalg.norm(v)
        assert cpp.cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors(self):
        v = np.random.randn(512).astype(np.float32)
        v /= np.linalg.norm(v)
        assert cpp.cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert cpp.cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_matches_numpy(self):
        a = np.random.randn(512).astype(np.float32)
        b = np.random.randn(512).astype(np.float32)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        expected = float(np.dot(a, b))
        assert cpp.cosine_similarity(a, b) == pytest.approx(expected, abs=1e-5)

    def test_dimension_mismatch_raises(self):
        a = np.ones(512, dtype=np.float32)
        b = np.ones(256, dtype=np.float32)
        with pytest.raises(Exception):
            cpp.cosine_similarity(a, b)


class TestBatchCosineTopk:
    @pytest.fixture()
    def setup_data(self):
        np.random.seed(42)
        query = np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)
        database = np.random.randn(100, 512).astype(np.float32)
        database /= np.linalg.norm(database, axis=1, keepdims=True)
        return query, database

    def test_returns_topk_result(self, setup_data):
        query, db = setup_data
        result = cpp.batch_cosine_topk(query, db, 0.0, 5)
        assert hasattr(result, "indices")
        assert hasattr(result, "scores")
        assert len(result.indices) == 5
        assert len(result.scores) == 5

    def test_scores_descending(self, setup_data):
        query, db = setup_data
        result = cpp.batch_cosine_topk(query, db, 0.0, 10)
        for i in range(len(result.scores) - 1):
            assert result.scores[i] >= result.scores[i + 1]

    def test_threshold_filters(self, setup_data):
        query, db = setup_data
        result = cpp.batch_cosine_topk(query, db, 0.99, 100)
        for s in result.scores:
            assert s >= 0.99

    def test_matches_numpy(self, setup_data):
        query, db = setup_data
        result = cpp.batch_cosine_topk(query, db, 0.0, 5)
        # Verify with numpy
        sims = db @ query
        top_np = np.argsort(sims)[::-1][:5]
        # C++ top-1 should match numpy top-1
        assert result.indices[0] == top_np[0]
        assert result.scores[0] == pytest.approx(float(sims[top_np[0]]), abs=1e-5)

    def test_empty_database(self):
        query = np.ones(512, dtype=np.float32) / np.sqrt(512)
        db = np.empty((0, 512), dtype=np.float32)
        result = cpp.batch_cosine_topk(query, db, 0.0, 5)
        assert len(result.indices) == 0
        assert len(result.scores) == 0

    def test_top_k_zero(self, setup_data):
        query, db = setup_data
        result = cpp.batch_cosine_topk(query, db, 0.0, 0)
        assert len(result.indices) == 0

    def test_dimension_mismatch_raises(self, setup_data):
        _, db = setup_data
        bad_query = np.ones(256, dtype=np.float32)
        with pytest.raises(Exception):
            cpp.batch_cosine_topk(bad_query, db, 0.0, 5)


# ──────────────────────────────────────────────────────────────
# blur_ops
# ──────────────────────────────────────────────────────────────

class TestLaplacianVariance:
    def test_uniform_image_zero_variance(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        var = cpp.laplacian_variance(gray)
        assert var == pytest.approx(0.0, abs=1e-6)

    def test_noisy_image_high_variance(self):
        np.random.seed(42)
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        var = cpp.laplacian_variance(gray)
        assert var > 1000  # Random noise has very high Laplacian variance

    def test_small_image_returns_zero(self):
        gray = np.zeros((2, 2), dtype=np.uint8)
        assert cpp.laplacian_variance(gray) == 0.0

    def test_positive_result(self):
        gray = np.zeros((50, 50), dtype=np.uint8)
        gray[25, 25] = 255  # Single bright pixel
        var = cpp.laplacian_variance(gray)
        assert var > 0


class TestFftHfRatio:
    def test_uniform_image_low_hf(self):
        gray = np.full((64, 64), 128, dtype=np.uint8)
        hf = cpp.fft_hf_ratio(gray)
        # Uniform image: all energy in DC, HF ratio should be 0
        assert hf == pytest.approx(0.0, abs=0.01)

    def test_noisy_image_high_hf(self):
        np.random.seed(42)
        gray = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        hf = cpp.fft_hf_ratio(gray)
        assert hf > 0.5

    def test_small_image_returns_zero(self):
        gray = np.zeros((3, 3), dtype=np.uint8)
        assert cpp.fft_hf_ratio(gray) == 0.0

    def test_result_between_0_and_1(self):
        np.random.seed(42)
        gray = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        hf = cpp.fft_hf_ratio(gray)
        assert 0.0 <= hf <= 1.0


class TestBatchBlurMetrics:
    def test_batch_returns_correct_count(self):
        images = [np.random.randint(0, 256, (64, 64), dtype=np.uint8) for _ in range(5)]
        results = cpp.batch_blur_metrics(images)
        assert len(results) == 5

    def test_batch_struct_fields(self):
        images = [np.random.randint(0, 256, (64, 64), dtype=np.uint8)]
        results = cpp.batch_blur_metrics(images)
        assert hasattr(results[0], "laplacian_var")
        assert hasattr(results[0], "hf_ratio")

    def test_empty_batch(self):
        results = cpp.batch_blur_metrics([])
        assert len(results) == 0


# ──────────────────────────────────────────────────────────────
# preprocess_ops
# ──────────────────────────────────────────────────────────────

class TestBgrToGray:
    def test_output_shape(self):
        bgr = np.random.randint(0, 256, (100, 80, 3), dtype=np.uint8)
        gray = cpp.bgr_to_gray(bgr)
        assert gray.shape == (100, 80)
        assert gray.dtype == np.uint8

    def test_pure_red(self):
        # BGR: (0, 0, 255) => gray = 0.114*0 + 0.587*0 + 0.299*255 ≈ 76
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255
        gray = cpp.bgr_to_gray(bgr)
        assert gray[0, 0] == pytest.approx(76, abs=1)

    def test_pure_green(self):
        # BGR: (0, 255, 0) => gray = 0.114*0 + 0.587*255 + 0.299*0 ≈ 150
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 1] = 255
        gray = cpp.bgr_to_gray(bgr)
        assert gray[0, 0] == pytest.approx(150, abs=1)

    def test_invalid_channels_raises(self):
        bgr = np.zeros((10, 10, 4), dtype=np.uint8)
        with pytest.raises(Exception):
            cpp.bgr_to_gray(bgr)


class TestResizeGray:
    def test_output_shape(self):
        gray = np.random.randint(0, 256, (100, 80), dtype=np.uint8)
        resized = cpp.resize_gray(gray, 50, 40)
        assert resized.shape == (50, 40)
        assert resized.dtype == np.uint8

    def test_upscale(self):
        gray = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        resized = cpp.resize_gray(gray, 20, 20)
        assert resized.shape == (20, 20)

    def test_invalid_dimensions_raises(self):
        gray = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(Exception):
            cpp.resize_gray(gray, 0, 10)

    def test_identity_resize(self):
        gray = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        resized = cpp.resize_gray(gray, 32, 32)
        # Not exact due to bilinear interpolation center offset, but close
        assert resized.shape == (32, 32)
