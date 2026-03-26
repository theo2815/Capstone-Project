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
    event_id: str | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
):
    """Process a batch of images for face recognition.

    Args:
        job_id: UUID of the job record.
        image_data_list: List of base64-encoded image data.
        operation: One of 'detect', 'search'.
        api_key_id: Tenant API key ID for search isolation.
        event_id: Event ID for event-scoped face search.
        threshold: Similarity threshold override (default uses config).
        top_k: Max results per face override (default 10).
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
                result = _search_single(
                    image, embedder,
                    api_key_id=api_key_id,
                    event_id=event_id,
                    threshold=threshold,
                    top_k=top_k,
                )
                results[i] = {"index": i, **result}
            else:
                results[i] = {"index": i, "error": f"Unknown operation: {operation}"}
        except Exception as e:
            logger.error("Face processing failed for image", index=i, error=str(e))
            results[i] = {"index": i, "error": str(e)}
        update_job_progress(job_id, i + 1, total)

    complete_job(job_id, results)
    logger.info("Face batch job completed", job_id=job_id, total=total)


def _search_single(
    image,
    embedder,
    api_key_id: str | None = None,
    event_id: str | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
) -> dict:
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
                threshold=threshold if threshold is not None else settings.FACE_SIMILARITY_THRESHOLD,
                top_k=top_k if top_k is not None else 10,
                api_key_id=api_key_id,
                event_id=event_id,
            )
            for r in results:
                matches.append({
                    "person_id": r["person_id"],
                    "person_name": r["person_name"],
                    "similarity": r["similarity"],
                    "bbox": face["bbox"],
                })

    return {"faces_detected": len(faces), "matches": matches}


@celery_app.task(bind=True, name="faces.enroll_batch")
def face_enroll_batch(
    self,
    job_id: str,
    image_data_list: list[str],
    person_name: str,
    person_id: str | None = None,
    api_key_id: str | None = None,
    event_id: str | None = None,
):
    """Batch enroll faces from multiple images.

    Two-phase approach:
      Phase 1 (no DB): decode images + run ML inference
      Phase 2 (DB only): store embeddings per image with per-image error handling

    This avoids holding a DB session open during inference (PR2-9) and
    prevents a single bad image from rolling back all prior work (PR2-10).
    """
    import base64
    import hashlib
    import uuid as _uuid

    from src.config import get_settings
    from src.db.repositories.sync_face_repo import SyncFaceRepository
    from src.db.sync_session import get_sync_session
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

    settings = get_settings()
    min_conf = settings.FACE_MIN_ENROLLMENT_CONFIDENCE
    total = len(image_data_list)
    results: list[dict] = [{} for _ in range(total)]

    # --- Phase 1: Decode + ML inference (no DB session held) ---
    inference_results: list[tuple[list[dict], str] | None] = [None] * total
    for i, b64_data in enumerate(image_data_list):
        try:
            raw_bytes = base64.b64decode(b64_data)
            image = decode_base64_image(b64_data, raw_bytes=raw_bytes)
            if image is None:
                results[i] = {"index": i, "error": "Failed to decode image"}
                continue
            faces = embedder.get_embeddings(image)
            image_hash = hashlib.sha256(raw_bytes).hexdigest()
            inference_results[i] = (faces, image_hash)
        except Exception as e:
            logger.error("Face inference failed for image", index=i, error=str(e))
            results[i] = {"index": i, "error": str(e)}
        update_job_progress(job_id, i + 1, total)

    # --- Phase 2: DB operations only (short session, per-image commits) ---
    # Create or get person
    with get_sync_session() as session:
        repo = SyncFaceRepository(session)
        if person_id:
            try:
                pid = _uuid.UUID(person_id)
            except ValueError:
                fail_job(job_id, f"Invalid person_id format: {person_id}")
                return
            # Verify person exists (mirrors the sync enroll endpoint check)
            from sqlalchemy import select
            from src.db.models import Person
            result = session.execute(
                select(Person).where(Person.id == pid)
            )
            if result.scalar_one_or_none() is None:
                fail_job(job_id, f"Person not found: {person_id}")
                return
        else:
            person = repo.create_person(
                name=person_name, api_key_id=api_key_id, event_id=event_id
            )
            pid = person.id

    # Store embeddings per image — each in its own session so partial
    # failures don't roll back prior successful images.
    for i, inf in enumerate(inference_results):
        if inf is None:
            continue
        faces, image_hash = inf
        try:
            with get_sync_session() as session:
                repo = SyncFaceRepository(session)
                stored = 0
                skipped = 0
                for face in faces:
                    conf = face["bbox"]["confidence"]
                    if conf < min_conf:
                        skipped += 1
                        continue
                    emb_result = repo.store_embedding(
                        person_id=pid,
                        embedding=face["embedding"],
                        source_image_hash=image_hash,
                        quality_score=conf,
                    )
                    if emb_result is None:
                        skipped += 1
                        continue
                    stored += 1
                results[i] = {
                    "index": i,
                    "faces_detected": len(faces),
                    "faces_enrolled": stored,
                    "skipped": skipped,
                }
        except Exception as e:
            logger.error("Face enrollment failed for image", index=i, error=str(e))
            results[i] = {"index": i, "error": str(e)}

    complete_job(job_id, results)
    logger.info(
        "Face enroll batch job completed",
        job_id=job_id,
        total=total,
        person_id=str(pid),
    )
