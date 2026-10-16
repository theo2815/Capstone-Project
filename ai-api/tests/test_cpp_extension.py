"""Tests for the _quickpitik_cpp C++ extension module.

Skipped automatically if the extension is not built.
"""
from __future__ import annotations

import numpy as np
import pytest

cpp = pytest.importorskip("_quickpitik_cpp", reason="C++ extension not built")


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


class TestClassifyPreprocess:
    """Pins the exported `classify_preprocess` binding's own contract.

    BlurClassifier no longer calls this. It does a bilinear crop-and-resize, and
    the classifier needs an area-averaged one — bilinear aliases on a large
    downscale and makes blurry photos read as `sharp` (2026-08-14 ADR). It is
    also not faster than the OpenCV path, and it ships in no deployment (the
    package builds with the setuptools backend and no ext_modules).

    These tests still earn their place: the binding is exported and documented,
    so shape / range / channel-order regressions should be caught. What they no
    longer imply is that a compiled extension changes how photos classify.
    """

    @staticmethod
    def _bilinear_reference(image: np.ndarray, size: int = 224) -> np.ndarray:
        """OpenCV equivalent of what the C++ claims to do: bilinear crop+resize."""
        import cv2

        h, w = image.shape[:2]
        m = min(h, w)
        top, left = (h - m) // 2, (w - m) // 2
        cropped = image[top:top + m, left:left + m]
        resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.expand_dims(np.transpose(rgb, (2, 0, 1)), axis=0)

    def test_output_shape_and_dtype(self):
        bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        out = cpp.classify_preprocess(bgr, 224)
        assert out.shape == (1, 3, 224, 224)
        assert out.dtype == np.float32

    def test_normalised_to_unit_range(self):
        bgr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        out = cpp.classify_preprocess(bgr, 224)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_channels_are_rgb_not_bgr(self):
        # Pure blue in BGR (255, 0, 0) must land in the LAST channel after the
        # BGR->RGB swap, not the first.
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255
        out = cpp.classify_preprocess(bgr, 224)
        assert out[0, 2].mean() == pytest.approx(1.0, abs=1e-3)
        assert out[0, 0].mean() == pytest.approx(0.0, abs=1e-3)

    def test_matches_bilinear_reference(self):
        """The binding really is a bilinear crop-and-resize, within rounding.

        It interpolates in double and normalises straight from the float, while
        cv2.resize rounds to uint8 first (and uses fixed-point weights), so the
        two differ by under one uint8 step — measured max ~0.76/255 on real
        photos. That gap is why the two paths were never bit-identical; it is no
        longer a classification risk now that BlurClassifier uses only OpenCV.
        """
        rng = np.random.default_rng(7)
        for shape in ((480, 640, 3), (640, 480, 3), (512, 512, 3)):
            bgr = rng.integers(0, 256, shape, dtype=np.uint8)
            got = cpp.classify_preprocess(bgr, 224)
            expected = self._bilinear_reference(bgr, 224)
            assert got.shape == expected.shape
            # 1.5/255 leaves headroom for rounding without admitting a real
            # algorithmic change (a wrong crop or channel order blows past this).
            assert np.abs(got - expected).max() < 1.5 / 255
