from __future__ import annotations

from src.utils.logging import get_logger
from src.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="bibs.recognize_batch")
def bib_recognize_batch(self, job_id: str, image_paths: list[str]):
    """Process a batch of images for bib number recognition.

    Uses sub-batching for the decode+detect phase to bound memory, while
    accumulating all bib crops (small, ~50 KB each) for batch OCR.

    Args:
        job_id: UUID of the job record.
        image_paths: List of blob-store file paths for each image.
    """
    from src.config import get_settings
    from src.workers.helpers import (
        complete_job,
        decode_image_from_path,
        decode_images_from_paths,
        fail_job,
        update_job_progress,
    )
    from src.workers.model_loader import get_bib_detector, get_bib_recognizer

    bib_ocr = get_bib_recognizer()
    if bib_ocr is None:
        fail_job(job_id, "Bib OCR model not loaded in worker")
        return

    bib_detector = get_bib_detector()
    settings = get_settings()
    sub_batch = settings.INFERENCE_SUB_BATCH_SIZE
    total = len(image_paths)
    results: list[dict] = [{} for _ in range(total)]

    # --- Phase 1+2: Decode + YOLO detect in sub-batches ---
    # Accumulate crops across all sub-batches (crops are small).
    # Structure: (global_image_index, detection_dict, cropped_ndarray)
    crop_tasks: list[tuple[int, dict]] = []
    # Track images with no detections for full-image OCR fallback
    no_detection_indices: list[int] = []
    # We need images for fallback OCR, but only keep those with no detections
    fallback_paths: dict[int, str] = {}

    for chunk_start in range(0, total, sub_batch):
        chunk_paths = image_paths[chunk_start:chunk_start + sub_batch]
        # Parallel decode: overlaps disk I/O with CPU decompression
        decoded = decode_images_from_paths(chunk_paths, max_dim=1024)
        chunk_images: list = []
        chunk_global_indices: list[int] = []

        for j, image in enumerate(decoded):
            idx = chunk_start + j
            if image is None:
                results[idx] = {"index": idx, "bibs": [], "error": "Failed to decode image"}
            else:
                chunk_images.append(image)
                chunk_global_indices.append(idx)

        if not chunk_images:
            update_job_progress(job_id, min(chunk_start + len(chunk_paths), total), total)
            continue

        # Batch YOLO detection for this sub-batch
        if bib_detector is not None and bib_detector.model is not None:
            try:
                batch_detections = bib_detector.detect_batch(chunk_images)
            except Exception as e:
                logger.error("Bib batch detection failed, falling back to per-image", error=str(e))
                batch_detections = [bib_detector.detect(img) for img in chunk_images]
        else:
            batch_detections = [[] for _ in chunk_images]

        # Extract crops from this sub-batch
        for local_i, (img, detections) in enumerate(zip(chunk_images, batch_detections)):
            global_i = chunk_global_indices[local_i]
            h, w = img.shape[:2]
            if not detections:
                no_detection_indices.append(global_i)
                fallback_paths[global_i] = image_paths[global_i]
                continue
            for det in detections:
                bbox = det["bbox"]
                x1 = max(0, int(bbox["x1"]))
                y1 = max(0, int(bbox["y1"]))
                x2 = min(w, int(bbox["x2"]))
                y2 = min(h, int(bbox["y2"]))
                if x2 <= x1 or y2 <= y1:
                    continue
                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                crop_tasks.append((global_i, det, cropped))

        # chunk_images freed after this iteration
        update_job_progress(job_id, min(chunk_start + len(chunk_paths), total), total)

    # --- Phase 3: Batch OCR on all accumulated crops ---
    ocr_results_map: dict[int, list[tuple[dict, dict]]] = {}
    if crop_tasks:
        crops_only = [t[2] for t in crop_tasks]
        try:
            ocr_outputs = bib_ocr.recognize_batch(crops_only)
        except Exception as e:
            logger.error("Bib batch OCR failed, falling back to per-crop", error=str(e))
            ocr_outputs = [bib_ocr.recognize(c) for c in crops_only]

        for (global_i, det, _), ocr_out in zip(crop_tasks, ocr_outputs):
            ocr_results_map.setdefault(global_i, []).append((ocr_out, det))

    # --- Phase 4: Assemble per-image results ---
    for i in range(total):
        if results[i]:
            # Already set (decode error)
            continue

        try:
            if i in ocr_results_map:
                bib_results = []
                for ocr_result, det in ocr_results_map[i]:
                    if ocr_result["bib_number"]:
                        bib_results.append({
                            "bib_number": ocr_result["bib_number"],
                            "confidence": ocr_result["confidence"],
                            "bbox": det["bbox"],
                            "all_candidates": ocr_result["all_candidates"],
                        })
                results[i] = {"index": i, "bibs": bib_results}
            elif i in no_detection_indices:
                # No crop detections — OCR on full image as fallback
                image = decode_image_from_path(fallback_paths[i], max_dim=1024)
                if image is not None:
                    h, w = image.shape[:2]
                    ocr_result = bib_ocr.recognize(image)
                    if ocr_result["bib_number"]:
                        results[i] = {"index": i, "bibs": [{
                            "bib_number": ocr_result["bib_number"],
                            "confidence": ocr_result["confidence"],
                            "bbox": {"x1": 0.0, "y1": 0.0, "x2": float(w), "y2": float(h)},
                            "all_candidates": ocr_result["all_candidates"],
                            "warning": "full_image_fallback",
                        }]}
                    else:
                        results[i] = {"index": i, "bibs": []}
                else:
                    results[i] = {"index": i, "bibs": []}
            else:
                results[i] = {"index": i, "bibs": []}
        except Exception as e:
            logger.error("Bib result assembly failed for image", index=i, error=str(e))
            results[i] = {"index": i, "bibs": [], "error": str(e)}

    if not job_id:
        return results
    complete_job(job_id, results)
    logger.info("Bib batch job completed", job_id=job_id, total=total)
