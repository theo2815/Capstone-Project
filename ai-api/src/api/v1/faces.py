from __future__ import annotations

import asyncio
import hashlib
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from src.api.v1.batch_utils import (
    batch_accepted_response,
    create_batch_job,
    store_blobs_and_get_paths,
    validate_batch_files,
)
from src.api.v1.mega_batch import dispatch_mega_batch
from src.middleware.auth import check_scope, verify_api_key
from src.schemas.common import APIResponse
from src.schemas.faces import (
    BoundingBox,
    FaceCompareResponse,
    FaceDetectResponse,
    FaceDetection,
    FaceEnrollResponse,
    FaceSearchResponse,
    FaceSearchResult,
    PersonListResponse,
    PersonResponse,
)
from src.utils.image_utils import get_image_dimensions, validate_and_decode
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/faces", tags=["Face Recognition"])


@router.post("/detect", response_model=APIResponse)
async def detect_faces(
    request: Request,
    file: UploadFile = File(...),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect faces in an image and return bounding boxes + landmarks."""
    check_scope("faces:read", key_meta)
    start = time.perf_counter()
    settings = request.app.state.settings
    _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    embedder = request.app.state.model_registry.get("face")
    if embedder is None:
        return JSONResponse(
            status_code=503,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "MODEL_UNAVAILABLE", "message": "Face model not loaded"},
            ).model_dump(mode="json"),
        )

    faces = await asyncio.to_thread(embedder.detect_faces, image)
    elapsed_ms = (time.perf_counter() - start) * 1000
    w, h = get_image_dimensions(image)

    face_detections = [
        FaceDetection(
            bbox=BoundingBox(**f["bbox"]),
            landmarks=f.get("landmarks"),
        )
        for f in faces
    ]

    data = FaceDetectResponse(
        faces_detected=len(faces),
        faces=face_detections,
        image_dimensions=(w, h),
        processing_time_ms=round(elapsed_ms, 2),
    )
    return APIResponse(
        success=True,
        request_id=getattr(request.state, "request_id", ""),
        data=data.model_dump(),
    )


@router.post("/enroll", response_model=APIResponse)
async def enroll_face(
    request: Request,
    file: UploadFile = File(...),
    person_name: str = Form(..., min_length=1, max_length=255),
    person_id: str | None = Form(default=None),
    event_id: str = Form(..., min_length=1, max_length=255),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect faces, extract embeddings, and store in the database.

    event_id is required: enrollment is fail-closed event isolation (root rule
    5), the same guarantee /faces/search already enforces. An embedding stored
    without an event is not merely unscoped — it is unreachable, because every
    search path now requires an event_id and ``event_id IS NULL`` matches none
    of them, and ``DELETE /faces/persons?event_id=`` cannot erase it either. So
    a missing scope produced silent, permanently un-erasable orphans rather
    than a leak. FastAPI rejects the request with 422 instead.
    """
    check_scope("faces:write", key_meta)
    start = time.perf_counter()
    settings = request.app.state.settings
    raw_bytes, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    embedder = request.app.state.model_registry.get("face")
    if embedder is None:
        return JSONResponse(
            status_code=503,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "MODEL_UNAVAILABLE", "message": "Face model not loaded"},
            ).model_dump(mode="json"),
        )

    faces = await asyncio.to_thread(embedder.get_embeddings, image)
    if not faces:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "NO_FACES", "message": "No faces detected in image"},
        )

    image_hash = hashlib.sha256(raw_bytes).hexdigest()
    caller_key_id = key_meta.get("key_id")

    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = FaceRepository(session)

        if person_id:
            try:
                pid = uuid.UUID(person_id)
            except ValueError:
                return APIResponse(
                    success=False,
                    request_id=getattr(request.state, "request_id", ""),
                    error={"code": "INVALID_INPUT", "message": "Invalid person_id format"},
                )
            # Scoped by event as well as tenant: adding faces to a person that
            # belongs to a DIFFERENT event would file the embedding under that
            # other event while the caller believed it was enrolling into this
            # one — the same cross-event mixing event_id exists to prevent.
            person = await repo.get_person(
                pid, api_key_id=caller_key_id, event_id=event_id
            )
            if person is None:
                return APIResponse(
                    success=False,
                    request_id=getattr(request.state, "request_id", ""),
                    error={"code": "NOT_FOUND", "message": "Person not found"},
                )
        else:
            person = await repo.create_person(
                name=person_name, api_key_id=caller_key_id, event_id=event_id
            )
            pid = person.id

        min_conf = settings.FACE_MIN_ENROLLMENT_CONFIDENCE
        stored = 0
        skipped = 0
        for face_index, face in enumerate(faces):
            conf = face["bbox"]["confidence"]
            if conf < min_conf:
                skipped += 1
                logger.warning(
                    "Skipping low-confidence face during enrollment",
                    confidence=conf,
                    threshold=min_conf,
                    person_id=str(pid),
                )
                continue
            result = await repo.store_embedding(
                person_id=pid,
                embedding=face["embedding"],
                # Per-face-distinct hash. A single source image can contain
                # several runners; the (person_id, source_image_hash) unique
                # index would otherwise collapse every face from one image to a
                # single stored embedding, leaving only one runner findable in a
                # crowd shot. The face-index suffix keeps every face while
                # preserving re-enroll idempotency (stable detection order).
                source_image_hash=f"{image_hash}:{face_index}",
                quality_score=conf,
            )
            if result is None:
                skipped += 1
                continue
            stored += 1

        if stored == 0:
            return APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={
                    "code": "LOW_QUALITY",
                    "message": (
                        f"All {len(faces)} detected face(s) were below the minimum "
                        f"enrollment confidence of {min_conf}"
                    ),
                },
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        data = FaceEnrollResponse(
            person_id=pid,
            person_name=person.name,
            event_id=person.event_id,
            faces_enrolled=stored,
            embeddings_stored=stored,
            processing_time_ms=round(elapsed_ms, 2),
        )
        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(),
        )


@router.post("/search", response_model=APIResponse)
async def search_faces(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Query(default=0.4, ge=0.0, le=1.0),
    top_k: int = Query(default=10, ge=1, le=100),
    event_id: str = Query(..., min_length=1, max_length=255),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect faces in an image and search the database for matches.

    event_id is required: search is fail-closed event isolation (root rule 5).
    Omitting it would match faces across every event for this key, so FastAPI
    rejects the request with 422 instead.
    """
    check_scope("faces:read", key_meta)
    start = time.perf_counter()
    settings = request.app.state.settings
    _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    embedder = request.app.state.model_registry.get("face")
    if embedder is None:
        return JSONResponse(
            status_code=503,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "MODEL_UNAVAILABLE", "message": "Face model not loaded"},
            ).model_dump(mode="json"),
        )

    faces = await asyncio.to_thread(embedder.get_embeddings, image)

    matches = []
    unmatched = []
    caller_key_id = key_meta.get("key_id")

    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = FaceRepository(session)

        if faces:
            # Batch search: single DB round-trip for all faces instead of N queries
            all_embeddings = [face["embedding"] for face in faces]
            all_results = await repo.batch_search_similar(
                embeddings=all_embeddings,
                threshold=threshold,
                top_k=top_k,
                api_key_id=caller_key_id,
                event_id=event_id,
            )

            for face, results in zip(faces, all_results):
                bbox = BoundingBox(**face["bbox"])
                if results:
                    for r in results:
                        matches.append(FaceSearchResult(
                            person_id=r["person_id"],
                            person_name=r["person_name"],
                            similarity=r["similarity"],
                            bbox=bbox,
                        ))
                else:
                    unmatched.append(FaceDetection(
                        bbox=bbox,
                        landmarks=face.get("landmarks"),
                    ))

    elapsed_ms = (time.perf_counter() - start) * 1000
    data = FaceSearchResponse(
        faces_detected=len(faces),
        matches=matches,
        unmatched_faces=unmatched,
        processing_time_ms=round(elapsed_ms, 2),
    )
    return APIResponse(
        success=True,
        request_id=getattr(request.state, "request_id", ""),
        data=data.model_dump(),
    )


@router.post("/compare", response_model=APIResponse)
async def compare_faces(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Compare two images for 1:1 face verification."""
    check_scope("faces:read", key_meta)
    start = time.perf_counter()
    settings = request.app.state.settings

    _, image1 = await validate_and_decode(file1, max_file_size=settings.MAX_FILE_SIZE)
    _, image2 = await validate_and_decode(file2, max_file_size=settings.MAX_FILE_SIZE)

    embedder = request.app.state.model_registry.get("face")
    if embedder is None:
        return JSONResponse(
            status_code=503,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "MODEL_UNAVAILABLE", "message": "Face model not loaded"},
            ).model_dump(mode="json"),
        )

    faces1, faces2 = await asyncio.gather(
        asyncio.to_thread(embedder.get_embeddings, image1),
        asyncio.to_thread(embedder.get_embeddings, image2),
    )

    if not faces1 or not faces2:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "NO_FACES", "message": "No face detected in one or both images"},
        )

    import numpy as np

    from src.ml.faces.matcher import cosine_similarity

    emb1 = np.array(faces1[0]["embedding"])
    emb2 = np.array(faces2[0]["embedding"])
    similarity = min(1.0, max(0.0, cosine_similarity(emb1, emb2)))

    threshold = settings.FACE_SIMILARITY_THRESHOLD
    elapsed_ms = (time.perf_counter() - start) * 1000

    data = FaceCompareResponse(
        is_match=similarity >= threshold,
        similarity=round(similarity, 4),
        face1=FaceDetection(bbox=BoundingBox(**faces1[0]["bbox"])),
        face2=FaceDetection(bbox=BoundingBox(**faces2[0]["bbox"])),
        processing_time_ms=round(elapsed_ms, 2),
    )
    return APIResponse(
        success=True,
        request_id=getattr(request.state, "request_id", ""),
        data=data.model_dump(),
    )


@router.get("/persons", response_model=APIResponse)
async def list_persons(
    request: Request,
    event_id: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """List enrolled persons with pagination (tenant-isolated)."""
    check_scope("faces:read", key_meta)
    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session_ctx

    caller_key_id = key_meta.get("key_id")

    async with get_session_ctx() as session:
        repo = FaceRepository(session)
        persons, total = await repo.list_persons_with_counts(
            api_key_id=caller_key_id,
            event_id=event_id,
            offset=offset,
            limit=limit,
        )

        person_list = []
        for p, count in persons:
            person_list.append(PersonResponse(
                person_id=p.id,
                person_name=p.name,
                event_id=p.event_id,
                embeddings_count=count,
                created_at=p.created_at,
                updated_at=p.updated_at,
            ))

        data = PersonListResponse(
            persons=person_list,
            total=total,
            offset=offset,
            limit=limit,
        )
        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(),
        )


@router.get("/persons/{person_id}", response_model=APIResponse)
async def get_person(
    request: Request,
    person_id: uuid.UUID,
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Get an enrolled person's metadata (tenant-isolated)."""
    check_scope("faces:read", key_meta)
    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session_ctx

    caller_key_id = key_meta.get("key_id")

    async with get_session_ctx() as session:
        repo = FaceRepository(session)
        person = await repo.get_person(person_id, api_key_id=caller_key_id)
        if person is None:
            return APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "NOT_FOUND", "message": "Person not found"},
            )

        count = await repo.get_embeddings_count(person_id)
        data = PersonResponse(
            person_id=person.id,
            person_name=person.name,
            event_id=person.event_id,
            embeddings_count=count,
            created_at=person.created_at,
            updated_at=person.updated_at,
        )
        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(),
        )


@router.delete("/persons/{person_id}", response_model=APIResponse)
async def delete_person(
    request: Request,
    person_id: uuid.UUID,
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Remove an enrolled person and all their embeddings (GDPR erasure, tenant-isolated)."""
    check_scope("faces:delete", key_meta)
    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session_ctx

    caller_key_id = key_meta.get("key_id")

    async with get_session_ctx() as session:
        repo = FaceRepository(session)
        deleted = await repo.delete_person(person_id, api_key_id=caller_key_id)
        if not deleted:
            return APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "NOT_FOUND", "message": "Person not found"},
            )

        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data={"deleted": True, "person_id": str(person_id)},
        )


@router.delete("/persons", response_model=APIResponse)
async def delete_persons_by_event(
    request: Request,
    event_id: str = Query(
        ..., min_length=1, max_length=255,
        description="Erase every person + embeddings enrolled under this event",
    ),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Bulk GDPR erasure: remove every person + their embeddings for one event
    (tenant-isolated). A backend calls this when it deletes an event so the
    biometric data tied to that event's photos is erased in a single call,
    instead of one delete per photo. event_id is required — there is no
    delete-all-for-tenant path here.
    """
    check_scope("faces:delete", key_meta)
    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session_ctx

    caller_key_id = key_meta.get("key_id")

    async with get_session_ctx() as session:
        repo = FaceRepository(session)
        deleted = await repo.delete_persons_by_event(event_id, api_key_id=caller_key_id)
        return APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data={"deleted": deleted, "event_id": event_id},
        )


def _require_event_for_search(
    request: Request, operation: str, event_id: str | None
) -> JSONResponse | None:
    """Fail-closed event isolation for the batch/mega search endpoints. A search
    MUST be event-scoped (root rule 5) or it would match across every event for
    the key; ``detect`` does no DB lookup, so event_id is irrelevant there.
    Returns a 422 JSONResponse when a search is missing its scope, else None.
    """
    if operation == "search" and not (event_id and event_id.strip()):
        return JSONResponse(
            status_code=422,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={
                    "code": "EVENT_ID_REQUIRED",
                    "message": "event_id is required when operation=search",
                },
            ).model_dump(mode="json"),
        )
    return None


@router.post("/search/batch", status_code=202)
async def search_faces_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (JPEG, PNG, WebP)"),
    operation: str = Query(default="search", pattern="^(detect|search)$"),
    event_id: str | None = Query(default=None),
    threshold: float = Query(default=0.4, ge=0.0, le=1.0),
    top_k: int = Query(default=10, ge=1, le=100),
    key_meta: dict = Depends(verify_api_key),
):
    """Submit a batch of images for async face processing.

    Supported operations:
    - **detect**: Detect faces and return bounding boxes.
    - **search**: Detect faces and search the database for matches.

    Returns a job ID immediately. Poll GET /api/v1/jobs/{job_id} for results.
    """
    check_scope("faces:read", key_meta)
    if (guard := _require_event_for_search(request, operation, event_id)) is not None:
        return guard
    settings = request.app.state.settings

    result = await validate_batch_files(
        request, files, settings.MAX_BATCH_SIZE, settings.MAX_FILE_SIZE
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, f"face_{operation}_batch", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    image_paths = store_blobs_and_get_paths(job_id, raw_bytes_list)

    from src.workers.tasks.face_tasks import face_process_batch

    face_process_batch.delay(
        job_id, image_paths, operation,
        api_key_id=key_meta.get("key_id"),
        event_id=event_id,
        threshold=threshold,
        top_k=top_k,
    )

    return batch_accepted_response(request, job_id, len(files))


@router.post("/enroll/batch", status_code=202)
async def enroll_faces_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (JPEG, PNG, WebP)"),
    person_name: str = Form(..., min_length=1, max_length=255),
    person_id: str | None = Form(default=None),
    event_id: str = Form(..., min_length=1, max_length=255),
    key_meta: dict = Depends(verify_api_key),
):
    """Submit a batch of images for async face enrollment.

    All images are enrolled under the same person. If `person_id` is provided,
    embeddings are added to that existing person; otherwise a new person is created.

    event_id is required, for the same reason as /faces/enroll: enrollment is
    fail-closed event isolation (root rule 5). /faces/enroll/mega already
    required it; these two were the inconsistent, fail-open surfaces.

    Returns a job ID immediately. Poll GET /api/v1/jobs/{job_id} for results.
    """
    check_scope("faces:write", key_meta)
    settings = request.app.state.settings

    result = await validate_batch_files(
        request, files, settings.MAX_BATCH_SIZE, settings.MAX_FILE_SIZE
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, "face_enroll_batch", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    image_paths = store_blobs_and_get_paths(job_id, raw_bytes_list)

    from src.workers.tasks.face_tasks import face_enroll_batch

    face_enroll_batch.delay(
        job_id,
        image_paths,
        person_name=person_name,
        person_id=person_id,
        api_key_id=key_meta.get("key_id"),
        event_id=event_id,
    )

    return batch_accepted_response(request, job_id, len(files))


@router.post("/search/mega", status_code=202)
async def search_faces_mega(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (up to 500)"),
    operation: str = Query(default="search", pattern="^(detect|search)$"),
    event_id: str | None = Query(default=None),
    threshold: float = Query(default=0.4, ge=0.0, le=1.0),
    top_k: int = Query(default=10, ge=1, le=100),
    key_meta: dict = Depends(verify_api_key),
):
    """Submit a mega-batch of images for async face processing.

    Accepts up to 500 images per request. The server automatically splits
    them into sub-tasks and merges results into a single job.
    """
    check_scope("faces:read", key_meta)
    if (guard := _require_event_for_search(request, operation, event_id)) is not None:
        return guard
    settings = request.app.state.settings

    result = await validate_batch_files(
        request, files, settings.MEGA_BATCH_MAX_SIZE, settings.MAX_FILE_SIZE
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, f"face_{operation}_mega", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    from src.workers.tasks.face_tasks import face_process_batch

    dispatch_mega_batch(
        job_id, raw_bytes_list, face_process_batch,
        extra_kwargs={
            "operation": operation,
            "api_key_id": key_meta.get("key_id"),
            "event_id": event_id,
            "threshold": threshold,
            "top_k": top_k,
        },
    )

    return batch_accepted_response(request, job_id, len(files))


@router.post("/enroll/mega", status_code=202)
async def enroll_faces_mega(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (up to 500)"),
    event_id: str = Form(..., min_length=1, max_length=255),
    key_meta: dict = Depends(verify_api_key),
):
    """Submit a mega-batch of photos for async indexing — ONE person per image.

    This is the bulk photo-indexing primitive. Unlike ``/enroll/batch`` (which
    lumps every image under a single shared person, for many selfies of one
    runner), each uploaded image is enrolled as its OWN person and EVERY
    detected face in it is stored (per-face hash). A photo therefore becomes one
    ai person; any runner appearing in it is findable at search time.

    Each file's filename is echoed back as ``ref`` in the per-image result, so
    the caller can map results to its own photo IDs without relying on order.
    Per-image result: ``{index, ref, person_id, faces_detected, faces_enrolled,
    skipped}`` (``person_id`` is null when no enrollable face was found — a
    benign "no runner in this shot" outcome). Poll GET /api/v1/jobs/{job_id}.
    """
    check_scope("faces:write", key_meta)
    settings = request.app.state.settings

    result = await validate_batch_files(
        request, files, settings.MEGA_BATCH_MAX_SIZE, settings.MAX_FILE_SIZE
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    refs = [f.filename or str(i) for i, f in enumerate(files)]

    job_id = await create_batch_job(
        request, "face_enroll_mega", len(files),
        key_meta.get("key_id"), rate_tier=key_meta.get("rate_tier"),
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    from src.workers.tasks.face_tasks import face_enroll_mega_batch

    dispatch_mega_batch(
        job_id, raw_bytes_list, face_enroll_mega_batch,
        extra_kwargs={
            "api_key_id": key_meta.get("key_id"),
            "event_id": event_id,
        },
        refs=refs,
    )

    return batch_accepted_response(request, job_id, len(files))
