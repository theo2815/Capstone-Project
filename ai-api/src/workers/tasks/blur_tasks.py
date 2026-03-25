from __future__ import annotations

from src.utils.logging import get_logger
from src.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="blur.detect_batch")
def blur_detect_batch(self, job_id: str, image_data_list: list[str]):
    """Process a batch of images for blur detection.

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
    from src.workers.model_loader import get_blur_detector

    detector = get_blur_detector()
    if detector is None:
        fail_job(job_id, "Blur detector model not loaded in worker")
        return

    total = len(image_data_list)
    results: list[dict] = [{}] * total

    # PERF-8: Pre-decode all images upfront (fail fast, better memory locality)
    images = []
    for i, b64_data in enumerate(image_data_list):
        image = decode_base64_image(b64_data)
        if image is None:
            results[i] = {"index": i, "error": "Failed to decode image"}
        images.append(image)

    # Run inference on successfully decoded images
    for i, image in enumerate(images):
        if image is None:
            update_job_progress(job_id, i + 1, total)
            continue
        try:
            detection = detector.detect(image)
            results[i] = {"index": i, **detection}
        except Exception as e:
            logger.error("Blur detection failed for image", index=i, error=str(e))
            results[i] = {"index": i, "error": str(e)}
        update_job_progress(job_id, i + 1, total)

    complete_job(job_id, results)
    logger.info("Blur batch job completed", job_id=job_id, total=total)


@celery_app.task(bind=True, name="blur.classify_batch")
def blur_classify_batch(
    self, job_id: str, image_data_list: list[str], blur_type: str | None = None
):
    """Process a batch of images for blur classification.

    Args:
        job_id: UUID of the job record.
        image_data_list: List of base64-encoded image data.
        blur_type: Optional specific blur type to detect. When provided,
            returns Detected/Not Detected per image instead of full classification.
    """
    from src.workers.helpers import (
        complete_job,
        decode_base64_image,
        fail_job,
        update_job_progress,
    )
    from src.workers.model_loader import get_blur_classifier

    classifier = get_blur_classifier()
    if classifier is None:
        fail_job(job_id, "Blur classifier model not loaded in worker")
        return

    total = len(image_data_list)
    results: list[dict] = [{}] * total

    # PERF-8: Pre-decode all images upfront
    images = []
    for i, b64_data in enumerate(image_data_list):
        image = decode_base64_image(b64_data)
        if image is None:
            results[i] = {"index": i, "error": "Failed to decode image"}
        images.append(image)

    # Run inference on successfully decoded images
    for i, image in enumerate(images):
        if image is None:
            update_job_progress(job_id, i + 1, total)
            continue
        try:
            if blur_type is not None:
                detection = classifier.detect_blur_type(image, blur_type)
                if detection is None:
                    results[i] = {"index": i, "error": "Classifier returned None"}
                else:
                    results[i] = {"index": i, **detection}
            else:
                classification = classifier.classify(image)
                if classification is None:
                    results[i] = {"index": i, "error": "Classifier returned None"}
                else:
                    results[i] = {"index": i, **classification}
        except Exception as e:
            logger.error("Blur classification failed for image", index=i, error=str(e))
            results[i] = {"index": i, "error": str(e)}
        update_job_progress(job_id, i + 1, total)

    complete_job(job_id, results)
    logger.info("Blur classify batch job completed", job_id=job_id, total=total)
