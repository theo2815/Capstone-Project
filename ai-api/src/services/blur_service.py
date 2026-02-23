from __future__ import annotations

import numpy as np

from src.ml.blur.classifier import BlurClassifier
from src.ml.blur.detector import BlurDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BlurService:
    """Business logic for blur detection and classification."""

    def __init__(
        self,
        detector: BlurDetector,
        classifier: BlurClassifier | None = None,
    ) -> None:
        self.detector = detector
        self.classifier = classifier

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

    def classify(self, image: np.ndarray) -> dict | None:
        """Classify an image into blur categories.

        Args:
            image: BGR numpy array.

        Returns:
            Dict with predicted_class, confidence, and probabilities.
            Returns None if classifier is not available.
        """
        if self.classifier is None:
            return None
        return self.classifier.classify(image)

    def detect_blur_type(self, image: np.ndarray, blur_type: str) -> dict | None:
        """Detect whether a specific blur type is present in an image.

        Args:
            image: BGR numpy array.
            blur_type: One of "defocused_object_portrait", "defocused_blurred",
                       "motion_blurred".

        Returns:
            Dict with detected, confidence, blur_type, and blur_type_probability.
            Returns None if classifier is not available.
        """
        if self.classifier is None:
            return None
        return self.classifier.detect_blur_type(image, blur_type)
