from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BibService:
    """Business logic for bib number detection and recognition."""

    def __init__(self, detector, recognizer) -> None:
        self.detector = detector
        self.recognizer = recognizer

    def recognize(self, image, confidence: float = 0.5) -> list[dict]:
        """Detect bib regions and run OCR on each.

        If detector is available, runs detect->crop->OCR pipeline.
        Otherwise, runs OCR on the full image as fallback.
        """
        if self.detector is not None and self.detector.model is not None:
            detections = self.detector.detect(image, confidence=confidence)
            results = []
            for det in detections:
                bbox = det["bbox"]
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                ocr_result = self.recognizer.recognize(cropped)
                if ocr_result["bib_number"]:
                    results.append({**ocr_result, "bbox": bbox})
            return results

        # Fallback: OCR on whole image
        h, w = image.shape[:2]
        ocr_result = self.recognizer.recognize(image)
        if ocr_result["bib_number"]:
            bbox = {"x1": 0.0, "y1": 0.0, "x2": float(w), "y2": float(h)}
            return [{**ocr_result, "bbox": bbox}]
        return []
