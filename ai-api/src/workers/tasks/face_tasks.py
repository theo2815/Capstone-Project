from __future__ import annotations

from src.utils.logging import get_logger
from src.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, name="faces.process_batch")
def face_process_batch(
    self,
    job_id: str,
    image_paths: list[str],
    operation: str,
    api_key_id: str | None = None,
    event_id: str | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
):
    """Process a batch of images for face recognition.

    Args:
        job_id: UUID of the job record.
        image_paths: List of blob-store file paths for each image.
        operation: One of 'detect', 'search'.
        api_key_id: Tenant API key ID for search isolation.
        event_id: Event ID for event-scoped face search.
        threshold: Similarity threshold override (default uses config).
        top_k: Max results per face override (default 10).
    """
    from src.workers.helpers import (
        complete_job,
        decode_image_from_path,
        fail_job,
        update_job_progress,
    )
    from src.workers.model_loader import get_face_embedder

    embedder = get_face_embedder()
    if embedder is None:
        fail_job(job_id, "Face embedder model not loaded in worker")
        return

    total = len(image_paths)
    results: list[dict] = [{} for _ in range(total)]

    if operation == "search":
        # Open ONE session for all images instead of per-image
        _search_batch(
            image_paths, embedder, results, job_id, total,
            api_key_id=api_key_id, event_id=event_id,
            threshold=threshold, top_k=top_k,
        )
    else:
        # detect: per-image, no DB needed
        for i, path in enumerate(image_paths):
            image = decode_image_from_path(path, max_dim=768)
            if image is None:
                results[i] = {"index": i, "error": "Failed to decode image"}
                update_job_progress(job_id, i + 1, total)
                continue
            try:
                faces = embedder.detect_faces(image)
                results[i] = {"index": i, "faces_detected": len(faces), "faces": faces}
            except Exception as e:
                logger.error("Face detection failed for image", index=i, error=str(e))
                results[i] = {"index": i, "error": str(e)}
            update_job_progress(job_id, i + 1, total)

    if not job_id:
        return results
    complete_job(job_id, results)
    logger.info("Face batch job completed", job_id=job_id, total=total)


def _search_batch(
    image_paths: list[str],
    embedder,
    results: list[dict],
    job_id: str,
    total: int,
    api_key_id: str | None = None,
    event_id: str | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
) -> None:
    """Detect + search for all images using a single DB session.

    Uses batch_search_similar() to issue a single LATERAL JOIN query for
    all faces in each image instead of N separate search_similar() round-trips.
    One DB session for the entire task instead of one per image.
    """
    from src.config import get_settings
    from src.db.repositories.sync_face_repo import SyncFaceRepository
    from src.db.sync_session import get_sync_session
    from src.workers.helpers import decode_image_from_path, update_job_progress

    settings = get_settings()
    eff_threshold = threshold if threshold is not None else settings.FACE_SIMILARITY_THRESHOLD
    eff_top_k = top_k if top_k is not None else 10

    with get_sync_session() as session:
        repo = SyncFaceRepository(session)

        for i, path in enumerate(image_paths):
            image = decode_image_from_path(path, max_dim=768)
            if image is None:
                results[i] = {"index": i, "error": "Failed to decode image"}
                update_job_progress(job_id, i + 1, total)
                continue
            try:
                faces = embedder.get_embeddings(image)
                if not faces:
                    results[i] = {"index": i, "faces_detected": 0, "matches": []}
                    update_job_progress(job_id, i + 1, total)
                    continue

                embeddings = [face["embedding"] for face in faces]
                per_face_results = repo.batch_search_similar(
                    embeddings=embeddings,
                    threshold=eff_threshold,
                    top_k=eff_top_k,
                    api_key_id=api_key_id,
                    event_id=event_id,
                )

                matches = []
                for face, face_results in zip(faces, per_face_results):
                    for r in face_results:
                        matches.append({
                            "person_id": r["person_id"],
                            "person_name": r["person_name"],
                            "similarity": r["similarity"],
                            "bbox": face["bbox"],
                        })
                results[i] = {"index": i, "faces_detected": len(faces), "matches": matches}
            except Exception as e:
                logger.error("Face search failed for image", index=i, error=str(e))
                results[i] = {"index": i, "error": str(e)}
            update_job_progress(job_id, i + 1, total)


@celery_app.task(bind=True, name="faces.enroll_batch")
def face_enroll_batch(
    self,
    job_id: str,
    image_paths: list[str],
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
    import hashlib
    import uuid as _uuid

    from src.config import get_settings
    from src.db.repositories.sync_face_repo import SyncFaceRepository
    from src.db.sync_session import get_sync_session
    from src.utils.blob_store import load_blob
    from src.workers.helpers import (
        _decode_raw_bytes,
        complete_job,
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
    total = len(image_paths)
    results: list[dict] = [{} for _ in range(total)]

    # --- Phase 1: Decode + ML inference (no DB session held) ---
    inference_results: list[tuple[list[dict], str] | None] = [None] * total
    for i, path in enumerate(image_paths):
        try:
            raw_bytes = load_blob(path)
            image = _decode_raw_bytes(raw_bytes, max_dim=768)
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

    # Store embeddings — single session with savepoints for per-image
    # isolation (1 connection instead of N).
    with get_sync_session() as session:
        repo = SyncFaceRepository(session)
        for i, inf in enumerate(inference_results):
            if inf is None:
                continue
            faces, image_hash = inf
            try:
                for face in faces:
                    face["source_image_hash"] = image_hash
                savepoint = session.begin_nested()
                stored, skipped = repo.bulk_store_embeddings(
                    person_id=pid,
                    faces=faces,
                    min_conf=min_conf,
                )
                savepoint.commit()
                results[i] = {
                    "index": i,
                    "faces_detected": len(faces),
                    "faces_enrolled": stored,
                    "skipped": skipped,
                }
            except Exception as e:
                savepoint.rollback()
                logger.error("Face enrollment failed for image", index=i, error=str(e))
                results[i] = {"index": i, "error": str(e)}

    complete_job(job_id, results)
    logger.info(
        "Face enroll batch job completed",
        job_id=job_id,
        total=total,
        person_id=str(pid),
    )
