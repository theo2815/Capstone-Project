from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BibDetector:
    """Detect bib number regions in images using YOLOv8."""

    def __init__(self, model_path: str = "./models/bib_detection/yolov8n_bib.onnx") -> None:
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            logger.info("BibDetector loaded", model_path=self.model_path)
        except Exception as e:
            logger.warning(
                "BibDetector model not found, detection will be unavailable",
                model_path=self.model_path,
                error=str(e),
            )
            self.model = None

    def detect(self, image: np.ndarray, confidence: float = 0.5) -> list[dict]:
        """Detect bib regions in an image.

        Args:
            image: BGR numpy array.
            confidence: Minimum detection confidence.

        Returns:
            List of dicts with 'bbox' and 'confidence' keys.
        """
        if self.model is None:
            logger.warning("BibDetector model not loaded, returning empty results")
            return []

        results = self.model(image, conf=confidence, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                    "confidence": float(box.conf[0]),
                })
        return detections
