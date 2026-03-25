from __future__ import annotations

from src.utils.logging import get_logger
from src.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="faces.process_batch")
def face_process_batch(
    self,
    job_id: str,
    image_data_list: list[str],
    operation: str,
    api_key_id: str | None = None,
):
    """Process a batch of images for face recognition.

    Args:
        job_id: UUID of the job record.
        image_data_list: List of base64-encoded image data.
        operation: One of 'detect', 'search'.
        api_key_id: Tenant API key ID for search isolation.
    """
    from src.workers.helpers import (
        complete_job,
        decode_base64_image,
        fail_job,
        update_job_progress,
    )
    from src.workers.model_loader import get_face_embedder

    embedder = get_face_embedder()
    if embedder is None:
        fail_job(job_id, "Face embedder model not loaded in worker")
        return

    total = len(image_data_list)
    results: list[dict] = [{} for _ in range(total)]

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
            if operation == "detect":
                faces = embedder.detect_faces(image)
                results[i] = {"index": i, "faces_detected": len(faces), "faces": faces}
            elif operation == "search":
                result = _search_single(image, embedder, api_key_id=api_key_id)
                results[i] = {"index": i, **result}
            else:
                results[i] = {"index": i, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            logger.error("Face processing failed for image", index=i, error=str(e))
            results[i] = {"index": i, "error": str(e)}
        update_job_progress(job_id, i + 1, total)

    complete_job(job_id, results)
    logger.info("Face batch job completed", job_id=job_id, total=total)


def _search_single(image, embedder, api_key_id: str | None = None) -> dict:
    """Detect faces, extract embeddings, and search the database."""
    from src.config import get_settings
    from src.db.repositories.sync_face_repo import SyncFaceRepository
    from src.db.sync_session import get_sync_session

    settings = get_settings()

    faces = embedder.get_embeddings(image)
    if not faces:
        return {"faces_detected": 0, "matches": []}

    matches = []
    with get_sync_session() as session:
        repo = SyncFaceRepository(session)
        for face in faces:
            results = repo.search_similar(
                query_embedding=face["embedding"],
                threshold=settings.FACE_SIMILARITY_THRESHOLD,
                top_k=10,
                api_key_id=api_key_id,
            )
            for r in results:
                matches.append({
                    "person_id": r["person_id"],
                    "person_name": r["person_name"],
                    "similarity": r["similarity"],
                    "bbox": face["bbox"],
                })

    return {"faces_detected": len(faces), "matches": matches}
