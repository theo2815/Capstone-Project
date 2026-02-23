from __future__ import annotations

import base64
import time

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from src.middleware.auth import verify_api_key
from src.schemas.blur import (
    BlurClassProbabilities,
    BlurClassificationResponse,
    BlurDetectionResponse,
    BlurMetrics,
    BlurType,
    BlurTypeDetectionResponse,
)
from src.schemas.common import APIResponse
from src.schemas.jobs import JobCreateResponse
from src.utils.image_utils import get_image_dimensions, validate_and_decode

router = APIRouter(prefix="/blur", tags=["Blur Detection"])


@router.post("/detect", response_model=APIResponse)
async def detect_blur(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    threshold: float = Query(default=100.0, ge=1.0, le=10000.0),
    include_metrics: bool = Query(default=True),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect if an image is blurry."""
    start = time.perf_counter()

    settings = request.app.state.settings
    _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    registry = request.app.state.model_registry
    detector = registry.get("blur")
    if detector is None:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "MODEL_UNAVAILABLE", "message": "Blur detector not loaded"},
        )

    result = detector.detect(image)
    elapsed_ms = (time.perf_counter() - start) * 1000
    w, h = get_image_dimensions(image)

    metrics = None
    if include_metrics:
        metrics = BlurMetrics(
            laplacian_variance=result["laplacian_variance"],
            hf_ratio=result["hf_ratio"],
            confidence=result["confidence"],
        )

    response_data = BlurDetectionResponse(
        is_blurry=result["is_blurry"],
        confidence=result["confidence"],
        metrics=metrics,
        image_dimensions=(w, h),
        processing_time_ms=round(elapsed_ms, 2),
    )

    return APIResponse(
        success=True,
        request_id=getattr(request.state, "request_id", ""),
        data=response_data.model_dump(),
    )


@router.post("/detect/batch", status_code=202)
async def detect_blur_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (JPEG, PNG, WebP)"),
    key_meta: dict = Depends(verify_api_key),
):
    """Submit a batch of images for async blur detection.

    Returns a job ID immediately. Poll GET /api/v1/jobs/{job_id} for results.
    """
    settings = request.app.state.settings

    if len(files) == 0:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "EMPTY_BATCH", "message": "No files provided"},
            ).model_dump(mode="json"),
        )

    if len(files) > settings.MAX_BATCH_SIZE:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={
                    "code": "BATCH_TOO_LARGE",
                    "message": f"Maximum {settings.MAX_BATCH_SIZE} files per batch",
                },
            ).model_dump(mode="json"),
        )

    # Read and base64 encode all files
    image_data_list = []
    for f in files:
        raw = await f.read()
        image_data_list.append(base64.b64encode(raw).decode("ascii"))

    # Create job record
    from src.db.repositories.job_repo import JobRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = JobRepository(session)
        job = await repo.create(job_type="blur_detect_batch", total_items=len(files))
        job_id = str(job.id)

    # Queue Celery task
    from src.workers.tasks.blur_tasks import blur_detect_batch

    blur_detect_batch.delay(job_id, image_data_list)

    data = JobCreateResponse(
        job_id=job_id,
        status="pending",
        total_items=len(files),
        poll_url=f"/api/v1/jobs/{job_id}",
    )
    return JSONResponse(
        status_code=202,
        content=APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(mode="json"),
        ).model_dump(mode="json"),
    )


@router.post("/classify", response_model=APIResponse)
async def classify_blur(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    blur_type: BlurType | None = Query(
        default=None,
        description="Specific blur type to detect. Returns Detected/Not Detected. "
        "Options: defocused_object_portrait, defocused_blurred, motion_blurred",
    ),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Classify an image into blur categories.

    When blur_type is provided, returns a targeted Detected/Not Detected response
    for the selected blur type only.

    When blur_type is omitted, returns full classification with predicted class
    (sharp, defocused_object_portrait, defocused_blurred, motion_blurred)
    and confidence scores.
    """
    start = time.perf_counter()

    settings = request.app.state.settings
    _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    registry = request.app.state.model_registry
    classifier = registry.get("blur_classifier")
    if classifier is None:
        return JSONResponse(
            status_code=503,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={
                    "code": "MODEL_UNAVAILABLE",
                    "message": "Blur classifier not loaded. Train and export the model first.",
                },
            ).model_dump(mode="json"),
        )

    if blur_type is not None:
        # Targeted detection mode
        result = classifier.detect_blur_type(image, blur_type.value)
        if result is None:
            return JSONResponse(
                status_code=503,
                content=APIResponse(
                    success=False,
                    request_id=getattr(request.state, "request_id", ""),
                    error={
                        "code": "MODEL_UNAVAILABLE",
                        "message": "Blur classifier session not available",
                    },
                ).model_dump(mode="json"),
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        w, h = get_image_dimensions(image)

        response_data = BlurTypeDetectionResponse(
            blur_type=result["blur_type"],
            detected=result["detected"],
            confidence=result["confidence"],
            blur_type_probability=result["blur_type_probability"],
            image_dimensions=(w, h),
            processing_time_ms=round(elapsed_ms, 2),
        )
    else:
        # Full classification mode (backward compatible)
        result = classifier.classify(image)
        if result is None:
            return JSONResponse(
                status_code=503,
                content=APIResponse(
                    success=False,
                    request_id=getattr(request.state, "request_id", ""),
                    error={
                        "code": "MODEL_UNAVAILABLE",
                        "message": "Blur classifier session not available",
                    },
                ).model_dump(mode="json"),
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        w, h = get_image_dimensions(image)

        response_data = BlurClassificationResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=BlurClassProbabilities(**result["probabilities"]),
            image_dimensions=(w, h),
            processing_time_ms=round(elapsed_ms, 2),
        )

    return APIResponse(
        success=True,
        request_id=getattr(request.state, "request_id", ""),
        data=response_data.model_dump(),
    )


@router.post("/classify/batch", status_code=202)
async def classify_blur_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (JPEG, PNG, WebP)"),
    blur_type: BlurType | None = Query(
        default=None,
        description="Specific blur type to detect in batch mode",
    ),
    key_meta: dict = Depends(verify_api_key),
):
    """Submit a batch of images for async blur classification.

    When blur_type is provided, each image is evaluated for the selected blur type
    and returns Detected/Not Detected per image.

    Returns a job ID immediately. Poll GET /api/v1/jobs/{job_id} for results.
    """
    settings = request.app.state.settings

    if len(files) == 0:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "EMPTY_BATCH", "message": "No files provided"},
            ).model_dump(mode="json"),
        )

    if len(files) > settings.MAX_BATCH_SIZE:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={
                    "code": "BATCH_TOO_LARGE",
                    "message": f"Maximum {settings.MAX_BATCH_SIZE} files per batch",
                },
            ).model_dump(mode="json"),
        )

    image_data_list = []
    for f in files:
        raw = await f.read()
        image_data_list.append(base64.b64encode(raw).decode("ascii"))

    from src.db.repositories.job_repo import JobRepository
    from src.db.session import get_session

    async for session in get_session():
        repo = JobRepository(session)
        job = await repo.create(job_type="blur_classify_batch", total_items=len(files))
        job_id = str(job.id)

    from src.workers.tasks.blur_tasks import blur_classify_batch

    blur_classify_batch.delay(
        job_id, image_data_list, blur_type.value if blur_type else None
    )

    data = JobCreateResponse(
        job_id=job_id,
        status="pending",
        total_items=len(files),
        poll_url=f"/api/v1/jobs/{job_id}",
    )
    return JSONResponse(
        status_code=202,
        content=APIResponse(
            success=True,
            request_id=getattr(request.state, "request_id", ""),
            data=data.model_dump(mode="json"),
        ).model_dump(mode="json"),
    )
