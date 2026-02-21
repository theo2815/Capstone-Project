from __future__ import annotations

from src.workers.celery_app import celery_app


@celery_app.task(bind=True, name="blur.detect_batch")
def blur_detect_batch(self, job_id: str, image_data_list: list[str]):
    """Process a batch of images for blur detection.

    Args:
        job_id: UUID of the job record.
        image_data_list: List of base64-encoded image data.
    """
    # Placeholder: full implementation in Phase 5
    pass
