from __future__ import annotations

import hashlib
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile

from src.middleware.auth import verify_api_key
from src.schemas.common import APIResponse
from src.schemas.faces import (
    BoundingBox,
    FaceCompareResponse,
    FaceDetectResponse,
    FaceDetection,
    FaceEnrollResponse,
    FaceSearchResponse,
    FaceSearchResult,
    PersonResponse,
)
from src.utils.image_utils import get_image_dimensions, validate_and_decode

router = APIRouter(prefix="/faces", tags=["Face Recognition"])


@router.post("/detect", response_model=APIResponse)
async def detect_faces(
    request: Request,
    file: UploadFile = File(...),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect faces in an image and return bounding boxes + landmarks."""
    start = time.perf_counter()
    settings = request.app.state.settings
    _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    embedder = request.app.state.model_registry.get("face")
    if embedder is None:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "MODEL_UNAVAILABLE", "message": "Face model not loaded"},
        )

    faces = embedder.detect_faces(image)
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
    person_name: str = Form(...),
    person_id: str | None = Form(default=None),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect faces, extract embeddings, and store in the database."""
    start = time.perf_counter()
    settings = request.app.state.settings
    raw_bytes, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    embedder = request.app.state.model_registry.get("face")
    if embedder is None:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "MODEL_UNAVAILABLE", "message": "Face model not loaded"},
        )

    faces = embedder.get_embeddings(image)
    if not faces:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "NO_FACES", "message": "No faces detected in image"},
        )

    image_hash = hashlib.sha256(raw_bytes).hexdigest()

    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = FaceRepository(session)

        if person_id:
            pid = uuid.UUID(person_id)
            person = await repo.get_person(pid)
            if person is None:
                return APIResponse(
                    success=False,
                    request_id=getattr(request.state, "request_id", ""),
                    error={"code": "NOT_FOUND", "message": "Person not found"},
                )
        else:
            person = await repo.create_person(name=person_name)
            pid = person.id

        stored = 0
        for face in faces:
            await repo.store_embedding(
                person_id=pid,
                embedding=face["embedding"],
                source_image_hash=image_hash,
                quality_score=face["bbox"]["confidence"],
            )
            stored += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        data = FaceEnrollResponse(
            person_id=pid,
            person_name=person_name,
            faces_enrolled=len(faces),
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
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect faces in an image and search the database for matches."""
    start = time.perf_counter()
    settings = request.app.state.settings
    _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    embedder = request.app.state.model_registry.get("face")
    if embedder is None:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "MODEL_UNAVAILABLE", "message": "Face model not loaded"},
        )

    faces = embedder.get_embeddings(image)

    matches = []
    unmatched = []

    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = FaceRepository(session)
        for face in faces:
            results = await repo.search_similar(
                query_embedding=face["embedding"],
                threshold=threshold,
                top_k=top_k,
            )
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
    start = time.perf_counter()
    settings = request.app.state.settings

    _, image1 = await validate_and_decode(file1, max_file_size=settings.MAX_FILE_SIZE)
    _, image2 = await validate_and_decode(file2, max_file_size=settings.MAX_FILE_SIZE)

    embedder = request.app.state.model_registry.get("face")
    if embedder is None:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "MODEL_UNAVAILABLE", "message": "Face model not loaded"},
        )

    faces1 = embedder.get_embeddings(image1)
    faces2 = embedder.get_embeddings(image2)

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
    similarity = cosine_similarity(emb1, emb2)

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


@router.get("/persons/{person_id}", response_model=APIResponse)
async def get_person(
    request: Request,
    person_id: uuid.UUID,
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Get an enrolled person's metadata."""
    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = FaceRepository(session)
        person = await repo.get_person(person_id)
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
            embeddings_count=count,
            created_at=person.created_at.isoformat(),
            updated_at=person.updated_at.isoformat(),
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
    """Remove an enrolled person and all their embeddings (GDPR erasure)."""
    from src.db.repositories.face_repo import FaceRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = FaceRepository(session)
        deleted = await repo.delete_person(person_id)
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
