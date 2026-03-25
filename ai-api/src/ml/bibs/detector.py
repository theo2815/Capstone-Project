from __future__ import annotations

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BibDetector:
    """Detect bib number regions in images using YOLOv8."""

    # Class names that correspond to bib detections.
    _BIB_CLASS_NAMES = {"bib"}

    def __init__(self, model_path: str = "./models/bib_detection/yolov8n_bib.onnx") -> None:
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.endswith(".onnx"):
            logger.error(
                "BibDetector only accepts .onnx models for security (no pickle)",
                model_path=self.model_path,
            )
            self.model = None
            return

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

        Only returns detections whose class name is in ``_BIB_CLASS_NAMES``
        so that a multi-class model (face+bib) only produces bib crops.

        Args:
            image: BGR numpy array.
            confidence: Minimum detection confidence.

        Returns:
            List of dicts with 'bbox', 'confidence', and 'class_name' keys.
        """
        if self.model is None:
            logger.warning("BibDetector model not loaded, returning empty results")
            return []

        results = self.model(image, conf=confidence, verbose=False)
        detections = []
        for result in results:
            names = result.names  # {0: "face", 1: "bib"} or {0: "bib"}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, "")
                if class_name not in self._BIB_CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                    "confidence": float(box.conf[0]),
                    "class_name": class_name,
                })
        return detections
