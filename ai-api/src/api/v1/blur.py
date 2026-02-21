from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile

from src.middleware.auth import verify_api_key
from src.schemas.blur import BlurDetectionResponse, BlurMetrics
from src.schemas.common import APIResponse
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
