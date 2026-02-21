"""Unit tests for BlurDetector."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.ml.blur.detector import BlurDetector


@pytest.fixture
def detector() -> BlurDetector:
    return BlurDetector(laplacian_threshold=100.0)


def _make_sharp_image(size: int = 256) -> np.ndarray:
    """Create a sharp image with high-frequency content (random noise)."""
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def _make_blurry_image(size: int = 256) -> np.ndarray:
    """Create a blurry image (uniform color with minimal variance)."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    noise = np.random.randint(-2, 3, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_gaussian_blurred(size: int = 256, ksize: int = 31) -> np.ndarray:
    """Create a real-world blurry image by Gaussian-blurring a sharp one."""
    sharp = _make_sharp_image(size)
    return cv2.GaussianBlur(sharp, (ksize, ksize), 0)


class TestBlurDetectorInit:
    def test_default_threshold(self):
        d = BlurDetector()
        assert d.laplacian_threshold == 100.0

    def test_custom_threshold(self):
        d = BlurDetector(laplacian_threshold=50.0)
        assert d.laplacian_threshold == 50.0


class TestBlurDetectorDetect:
    def test_returns_required_keys(self, detector: BlurDetector):
        image = _make_sharp_image()
        result = detector.detect(image)
        assert "is_blurry" in result
        assert "confidence" in result
        assert "laplacian_variance" in result
        assert "hf_ratio" in result

    def test_sharp_image_not_blurry(self, detector: BlurDetector):
        image = _make_sharp_image()
        result = detector.detect(image)
        assert result["is_blurry"] is False
        assert result["confidence"] > 0.5
        assert result["laplacian_variance"] > detector.laplacian_threshold

    def test_blurry_image_is_blurry(self, detector: BlurDetector):
        image = _make_blurry_image()
        result = detector.detect(image)
        assert result["is_blurry"] is True
        assert result["confidence"] > 0.5
        assert result["laplacian_variance"] < detector.laplacian_threshold

    def test_gaussian_blurred_is_blurry(self, detector: BlurDetector):
        image = _make_gaussian_blurred(ksize=51)
        result = detector.detect(image)
        assert result["is_blurry"] is True

    def test_heavier_blur_has_lower_laplacian(self, detector: BlurDetector):
        """More blur should yield lower laplacian variance."""
        light = _make_gaussian_blurred(ksize=11)
        heavy = _make_gaussian_blurred(ksize=51)
        result_light = detector.detect(light)
        result_heavy = detector.detect(heavy)
        assert result_heavy["laplacian_variance"] < result_light["laplacian_variance"]

    def test_confidence_range(self, detector: BlurDetector):
        """Confidence must always be between 0 and 1."""
        for img_fn in [_make_sharp_image, _make_blurry_image, _make_gaussian_blurred]:
            result = detector.detect(img_fn())
            assert 0.0 <= result["confidence"] <= 1.0

    def test_hf_ratio_range(self, detector: BlurDetector):
        """High-frequency ratio must be between 0 and 1."""
        for img_fn in [_make_sharp_image, _make_blurry_image]:
            result = detector.detect(img_fn())
            assert 0.0 <= result["hf_ratio"] <= 1.0

    def test_sharp_has_higher_hf_ratio(self, detector: BlurDetector):
        """Sharp images should have more high-frequency energy than blurry ones."""
        sharp_result = detector.detect(_make_sharp_image())
        blurry_result = detector.detect(_make_blurry_image())
        assert sharp_result["hf_ratio"] > blurry_result["hf_ratio"]

    def test_laplacian_variance_is_positive(self, detector: BlurDetector):
        result = detector.detect(_make_sharp_image())
        assert result["laplacian_variance"] >= 0.0

    def test_custom_threshold_changes_classification(self):
        """A very high threshold should classify everything as blurry."""
        strict = BlurDetector(laplacian_threshold=999999.0)
        result = strict.detect(_make_sharp_image())
        assert result["is_blurry"] is True

    def test_very_low_threshold_classifies_as_sharp(self):
        """A very low threshold should classify even blurry images as sharp."""
        lenient = BlurDetector(laplacian_threshold=0.001)
        result = lenient.detect(_make_blurry_image())
        assert result["is_blurry"] is False

    def test_different_image_sizes(self, detector: BlurDetector):
        """Detector should work with various image sizes."""
        for size in [64, 128, 512]:
            result = detector.detect(_make_sharp_image(size=size))
            assert "is_blurry" in result

    def test_non_square_image(self, detector: BlurDetector):
        """Detector should work with non-square images."""
        image = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
        result = detector.detect(image)
        assert "is_blurry" in result
        assert result["is_blurry"] is False
