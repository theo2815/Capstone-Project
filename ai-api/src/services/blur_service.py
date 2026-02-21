from __future__ import annotations

import numpy as np

from src.ml.blur.detector import BlurDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BlurService:
    """Business logic for blur detection."""

    def __init__(self, detector: BlurDetector) -> None:
        self.detector = detector

    def detect(self, image: np.ndarray, threshold: float | None = None) -> dict:
        """Run blur detection on an image.

        Args:
            image: BGR numpy array.
            threshold: Optional override for the Laplacian threshold.
        """
        if threshold is not None:
            original = self.detector.laplacian_threshold
            self.detector.laplacian_threshold = threshold
            result = self.detector.detect(image)
            self.detector.laplacian_threshold = original
        else:
            result = self.detector.detect(image)

        return result
