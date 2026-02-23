from __future__ import annotations

from src.utils.logging import get_logger
from src.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="bibs.recognize_batch")
def bib_recognize_batch(self, job_id: str, image_data_list: list[str]):
    """Process a batch of images for bib number recognition.

    Args:
        job_id: UUID of the job record.
        image_data_list: List of base64-encoded image data.
    """
    from src.workers.helpers import (
        complete_job,
        decode_base64_image,
        fail_job,
        update_job_progress,
    )
    from src.workers.model_loader import get_bib_detector, get_bib_recognizer

    bib_ocr = get_bib_recognizer()
    if bib_ocr is None:
        fail_job(job_id, "Bib OCR model not loaded in worker")
        return

    bib_detector = get_bib_detector()
    total = len(image_data_list)
    results = []

    for i, b64_data in enumerate(image_data_list):
        try:
            image = decode_base64_image(b64_data)
            if image is None:
                results.append({"index": i, "bibs": [], "error": "Failed to decode image"})
            else:
                bibs = _recognize_single(image, bib_detector, bib_ocr)
                results.append({"index": i, "bibs": bibs})
        except Exception as e:
            logger.error("Bib recognition failed for image", index=i, error=str(e))
            results.append({"index": i, "bibs": [], "error": str(e)})

        update_job_progress(job_id, i + 1, total)

    complete_job(job_id, results)
    logger.info("Bib batch job completed", job_id=job_id, total=total)


def _recognize_single(image, bib_detector, bib_ocr) -> list[dict]:
    """Run the bib recognition pipeline on a single image."""
    h, w = image.shape[:2]

    if bib_detector is not None and bib_detector.model is not None:
        detections = bib_detector.detect(image)
        bib_results = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            ocr_result = bib_ocr.recognize(cropped)
            if ocr_result["bib_number"]:
                bib_results.append({
                    "bib_number": ocr_result["bib_number"],
                    "confidence": ocr_result["confidence"],
                    "bbox": bbox,
                    "all_candidates": ocr_result["all_candidates"],
                })
        return bib_results

    # Fallback: OCR on whole image
    ocr_result = bib_ocr.recognize(image)
    if ocr_result["bib_number"]:
        return [{
            "bib_number": ocr_result["bib_number"],
            "confidence": ocr_result["confidence"],
            "bbox": {"x1": 0.0, "y1": 0.0, "x2": float(w), "y2": float(h)},
            "all_candidates": ocr_result["all_candidates"],
        }]
    return []
