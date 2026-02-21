from __future__ import annotations

from src.workers.celery_app import celery_app


@celery_app.task(bind=True, name="faces.process_batch")
def face_process_batch(self, job_id: str, image_data_list: list[str], operation: str):
    """Process a batch of images for face recognition.

    Args:
        job_id: UUID of the job record.
        image_data_list: List of base64-encoded image data.
        operation: One of 'detect', 'search', 'enroll'.
    """
    # Placeholder: full implementation in Phase 5
    pass
