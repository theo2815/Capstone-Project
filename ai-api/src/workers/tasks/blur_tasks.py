from __future__ import annotations

from src.utils.logging import get_logger
from src.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="blur.detect_batch")
def blur_detect_batch(self, job_id: str, image_paths: list[str]):
    """Process a batch of images for blur detection.

    Uses sub-batching with parallel grayscale decode and Laplacian-only
    detection (no FFT). Mirrors the fast path in the streaming endpoint
    and the sub-batch pattern from blur_classify_batch.

    Args:
        job_id: UUID of the job record.
        image_paths: List of blob-store file paths for each image.
    """
    from src.config import get_settings
    from src.workers.helpers import (
        complete_job,
        decode_grays_from_paths,
        fail_job,
        update_job_progress,
    )
    from src.workers.model_loader import get_blur_detector

    detector = get_blur_detector()
    if detector is None:
        fail_job(job_id, "Blur detector model not loaded in worker")
        return

    settings = get_settings()
    sub_batch = settings.INFERENCE_SUB_BATCH_SIZE
    total = len(image_paths)
    results: list[dict] = [{} for _ in range(total)]

    # Process in sub-batches: parallel grayscale decode + fast Laplacian-only detection.
    for chunk_start in range(0, total, sub_batch):
        chunk_paths = image_paths[chunk_start:chunk_start + sub_batch]
        # Parallel decode: overlaps disk I/O with CPU decompression (grayscale, no EXIF)
        decoded = decode_grays_from_paths(chunk_paths, max_dim=640)

        for j, gray in enumerate(decoded):
            idx = chunk_start + j
            if gray is None:
                results[idx] = {"index": idx, "error": "Failed to decode image"}
                continue
            try:
                detection = detector.detect_fast(gray)
                results[idx] = {"index": idx, **detection}
            except Exception as e:
                logger.error("Blur detection failed for image", index=idx, error=str(e))
                results[idx] = {"index": idx, "error": str(e)}

        update_job_progress(job_id, min(chunk_start + len(chunk_paths), total), total)

    if not job_id:
        # Mega-batch sub-task: return results for chord callback
        return results
    complete_job(job_id, results)
    logger.info("Blur batch job completed", job_id=job_id, total=total)


@celery_app.task(bind=True, name="blur.classify_batch")
def blur_classify_batch(
    self, job_id: str, image_paths: list[str], blur_type: str | None = None
):
    """Process a batch of images for blur classification.

    Uses sub-batching (INFERENCE_SUB_BATCH_SIZE) to keep peak memory bounded
    while still leveraging ONNX batch inference within each sub-batch.

    Args:
        job_id: UUID of the job record.
        image_paths: List of blob-store file paths for each image.
        blur_type: Optional specific blur type to detect. When provided,
            returns Detected/Not Detected per image instead of full classification.
    """
    from src.config import get_settings
    from src.workers.helpers import (
        complete_job,
        decode_images_from_paths,
        fail_job,
        update_job_progress,
    )
    from src.workers.model_loader import get_blur_classifier

    classifier = get_blur_classifier()
    if classifier is None:
        fail_job(job_id, "Blur classifier model not loaded in worker")
        return

    settings = get_settings()
    sub_batch = settings.INFERENCE_SUB_BATCH_SIZE
    total = len(image_paths)
    results: list[dict] = [{} for _ in range(total)]

    # Process in sub-batches to bound memory at O(sub_batch) instead of O(total).
    for chunk_start in range(0, total, sub_batch):
        chunk_paths = image_paths[chunk_start:chunk_start + sub_batch]
        # Parallel decode: overlaps disk I/O with CPU decompression
        decoded = decode_images_from_paths(chunk_paths, max_dim=640)
        chunk_images: list = []
        chunk_indices: list[int] = []

        for j, image in enumerate(decoded):
            idx = chunk_start + j
            if image is None:
                results[idx] = {"index": idx, "error": "Failed to decode image"}
            else:
                chunk_images.append(image)
                chunk_indices.append(idx)

        if chunk_images:
            try:
                batch_classifications = classifier.classify_batch(chunk_images)
                for img_i, classification in zip(chunk_indices, batch_classifications):
                    if classification is None:
                        results[img_i] = {"index": img_i, "error": "Classifier returned None"}
                        continue
                    if blur_type is not None:
                        blur_type_probability = classification["probabilities"].get(blur_type, 0.0)
                        detected = classification["predicted_class"] == blur_type
                        results[img_i] = {
                            "index": img_i,
                            "detected": detected,
                            "confidence": classification["confidence"],
                            "blur_type": blur_type,
                            "blur_type_probability": blur_type_probability,
                            "predicted_class": classification["predicted_class"],
                            "probabilities": classification["probabilities"],
                        }
                    else:
                        results[img_i] = {"index": img_i, **classification}
            except Exception as e:
                logger.error("Blur classification sub-batch failed", error=str(e))
                # Fall back to per-image on sub-batch failure
                for img_i, image in zip(chunk_indices, chunk_images):
                    try:
                        if blur_type is not None:
                            r = classifier.detect_blur_type(image, blur_type)
                        else:
                            r = classifier.classify(image)
                        results[img_i] = {"index": img_i, **(r or {"error": "Classifier returned None"})}
                    except Exception as e2:
                        results[img_i] = {"index": img_i, "error": str(e2)}

        update_job_progress(job_id, min(chunk_start + len(chunk_paths), total), total)

    if not job_id:
        return results
    complete_job(job_id, results)
    logger.info("Blur classify batch job completed", job_id=job_id, total=total)
