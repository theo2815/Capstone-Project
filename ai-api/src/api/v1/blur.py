from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from src.api.v1.batch_utils import (
    batch_accepted_response,
    create_batch_job,
    store_blobs_and_get_paths,
    validate_batch_files,
)
from src.api.v1.mega_batch import dispatch_mega_batch
from src.middleware.auth import check_scope, verify_api_key
from src.schemas.blur import (
    BlurClassProbabilities,
    BlurClassificationResponse,
    BlurDetectionResponse,
    BlurMetrics,
    BlurType,
    BlurTypeDetectionResponse,
)
from src.schemas.common import APIResponse
from src.utils.image_utils import get_image_dimensions, validate_and_decode
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/blur", tags=["Blur Detection"])


# ---------------------------------------------------------------------------
# Fast decode + detect for streaming batch (bypasses full PIL pipeline)
# ---------------------------------------------------------------------------

def _fast_decode_and_detect(
    data: bytes,
    filename: str,
    index: int,
    detector: Any,
    threshold: float | None,
    include_hf_ratio: bool = False,
) -> dict:
    """Decode an image with OpenCV and run blur detection in one call.

    Optimized for throughput:
    - Decodes as grayscale directly (30% faster than color decode)
    - Downscales to 640px (blur normalizes to this reference anyway)
    - Skips FFT hf_ratio by default (saves ~8ms per image)
    - Skips EXIF rotation (irrelevant for blur metrics)
    """
    try:
        arr = np.frombuffer(data, dtype=np.uint8)

        if include_hf_ratio:
            # Need BGR for full detector.detect() which computes both metrics
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if image is None:
                return {"index": index, "filename": filename, "error": "Failed to decode image"}
            h, w = image.shape[:2]
            if max(h, w) > 640:
                scale = 640 / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            result = detector.detect(image, threshold_override=threshold)
            return {"index": index, "filename": filename, **result}

        # Fast path: grayscale decode + Laplacian only (~18ms vs ~70ms)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return {"index": index, "filename": filename, "error": "Failed to decode image"}

        h, w = gray.shape
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Normalize to 640px reference resolution
        h2, w2 = gray.shape
        if max(h2, w2) > 0:
            laplacian_var = laplacian_var * 640 / max(h2, w2)

        thr = threshold if threshold is not None else detector.laplacian_threshold
        is_blurry = laplacian_var < thr
        if is_blurry:
            confidence = 1.0 - min(1.0, laplacian_var / thr)
        else:
            confidence = min(1.0, laplacian_var / thr)

        return {
            "index": index,
            "filename": filename,
            "is_blurry": is_blurry,
            "confidence": round(confidence, 4),
            "laplacian_variance": round(laplacian_var, 2),
        }
    except Exception as e:
        return {"index": index, "filename": filename, "error": str(e)}


# ---------------------------------------------------------------------------
# Fast decode for streaming classify batch (bypasses full PIL pipeline)
# ---------------------------------------------------------------------------

def _decode_bgr_for_classify(data: bytes, max_dim: int) -> np.ndarray | None:
    """Decode image bytes to a BGR array, downscaled to *max_dim* on the long edge.

    OpenCV BGR decode (no PIL, no EXIF — desktop pre-rotates via Sharp), then an
    INTER_AREA downscale to BLUR_CLASSIFY_DECODE_DIM (the classifier resizes to
    224px internally). That downscale is accuracy-relevant, not just a speed
    trick — see the setting's note in config.py. Returns None if the bytes cannot
    be decoded. Inference is done in batch by the caller via
    BlurClassifier.classify_batch().
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


@router.post("/detect", response_model=APIResponse)
async def detect_blur(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    threshold: float = Query(default=100.0, ge=1.0, le=10000.0),
    include_metrics: bool = Query(default=True),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect if an image is blurry."""
    check_scope("blur:read", key_meta)
    start = time.perf_counter()

    settings = request.app.state.settings
    _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    registry = request.app.state.model_registry
    detector = registry.get("blur")
    if detector is None:
        return JSONResponse(
            status_code=503,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "MODEL_UNAVAILABLE", "message": "Blur detector not loaded"},
            ).model_dump(mode="json"),
        )

    result = await asyncio.to_thread(detector.detect, image, threshold_override=threshold)
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


@router.post(
    "/detect/stream",
    summary="High-throughput batch blur detection (streaming)",
    response_class=StreamingResponse,
)
async def detect_blur_stream(
    request: Request,
    files: list[UploadFile] = File(..., description="Images to scan (max 500 per request)"),
    threshold: float | None = Query(default=None, ge=1.0, le=10000.0),
    include_hf_ratio: bool = Query(
        default=False,
        description="Include FFT high-frequency ratio (slower, adds ~8ms per image)",
    ),
    key_meta: dict = Depends(verify_api_key),
):
    """High-throughput synchronous blur detection for desktop applications.

    Processes up to 500 images concurrently and streams results as NDJSON
    (one JSON object per line). No Celery, no polling — results stream back
    as each image is processed.

    For 5k-10k images, send multiple concurrent requests of 200-500 images each.

    Response format (one per line):
    ```
    {"index":0,"filename":"IMG_001.jpg","is_blurry":false,"confidence":0.82,...}
    {"index":1,"filename":"IMG_002.jpg","is_blurry":true,"confidence":0.95,...}
    ```
    """
    check_scope("blur:read", key_meta)
    settings = request.app.state.settings
    max_size = settings.STREAM_BATCH_MAX_SIZE
    concurrency = settings.STREAM_CONCURRENCY

    if len(files) > max_size:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={
                    "code": "BATCH_TOO_LARGE",
                    "message": f"Maximum {max_size} images per stream request. "
                    f"Send multiple concurrent requests for larger sets.",
                },
            ).model_dump(mode="json"),
        )

    registry = request.app.state.model_registry
    detector = registry.get("blur")
    if detector is None:
        return JSONResponse(
            status_code=503,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={"code": "MODEL_UNAVAILABLE", "message": "Blur detector not loaded"},
            ).model_dump(mode="json"),
        )

    # Read all file data into memory (UploadFile is async)
    items: list[tuple[int, str, bytes]] = []
    for i, f in enumerate(files):
        data = await f.read()
        items.append((i, f.filename or f"image_{i}", data))

    total = len(items)
    start = time.perf_counter()

    async def generate():
        sem = asyncio.Semaphore(concurrency)

        async def process(index: int, filename: str, data: bytes) -> str:
            async with sem:
                result = await asyncio.to_thread(
                    _fast_decode_and_detect, data, filename, index, detector,
                    threshold, include_hf_ratio,
                )
                return json.dumps(result, separators=(",", ":")) + "\n"

        tasks = [process(i, name, data) for i, name, data in items]
        for coro in asyncio.as_completed(tasks):
            yield await coro

        # Final summary line
        elapsed = (time.perf_counter() - start) * 1000
        summary = {"_summary": True, "total": total, "processing_time_ms": round(elapsed, 2)}
        yield json.dumps(summary, separators=(",", ":")) + "\n"

    logger.info("Stream blur detection started", total=total, request_id=getattr(request.state, "request_id", ""))

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Total-Images": str(total)},
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
    check_scope("blur:read", key_meta)
    settings = request.app.state.settings

    result = await validate_batch_files(
        request, files, settings.MAX_BATCH_SIZE, settings.MAX_FILE_SIZE
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, "blur_detect_batch", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    image_paths = store_blobs_and_get_paths(job_id, raw_bytes_list)

    from src.workers.tasks.blur_tasks import blur_detect_batch

    blur_detect_batch.delay(job_id, image_paths)

    return batch_accepted_response(request, job_id, len(files))


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
    check_scope("blur:read", key_meta)
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
        result = await asyncio.to_thread(classifier.detect_blur_type, image, blur_type.value)
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
        result = await asyncio.to_thread(classifier.classify, image)
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


@router.post(
    "/classify/stream",
    summary="High-throughput batch blur classification (streaming)",
    response_class=StreamingResponse,
)
async def classify_blur_stream(
    request: Request,
    files: list[UploadFile] = File(..., description="Images to classify (max 500 per request)"),
    blur_type: BlurType | None = Query(
        default=None,
        description="Specific blur type to detect. Returns detected/not-detected per image.",
    ),
    key_meta: dict = Depends(verify_api_key),
):
    """High-throughput synchronous blur classification for desktop applications.

    Processes up to 500 images concurrently and streams results as NDJSON
    (one JSON object per line). No Celery, no polling — results stream back
    as each image is classified.

    For 5k-10k images, send multiple concurrent requests of 200-500 images each.

    Response format (one per line):
    ```
    {"index":0,"filename":"IMG_001.jpg","predicted_class":"sharp","confidence":0.97,...}
    {"index":1,"filename":"IMG_002.jpg","detected":true,"blur_type":"motion_blurred",...}
    ```
    """
    check_scope("blur:read", key_meta)
    settings = request.app.state.settings
    max_size = settings.STREAM_CLASSIFY_MAX_SIZE
    concurrency = settings.STREAM_CLASSIFY_CONCURRENCY

    if len(files) > max_size:
        return JSONResponse(
            status_code=400,
            content=APIResponse(
                success=False,
                request_id=getattr(request.state, "request_id", ""),
                error={
                    "code": "BATCH_TOO_LARGE",
                    "message": f"Maximum {max_size} images per stream request. "
                    f"Send multiple concurrent requests for larger sets.",
                },
            ).model_dump(mode="json"),
        )

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

    # Read all file data into memory (UploadFile is async)
    items: list[tuple[int, str, bytes]] = []
    for i, f in enumerate(files):
        data = await f.read()
        items.append((i, f.filename or f"image_{i}", data))

    total = len(items)
    bt = blur_type.value if blur_type else None
    sub_batch = settings.INFERENCE_SUB_BATCH_SIZE
    decode_dim = settings.BLUR_CLASSIFY_DECODE_DIM
    min_conf = classifier.min_detection_confidence
    start = time.perf_counter()

    def _build_line(index: int, filename: str, image, result) -> dict:
        if image is None:
            return {"index": index, "filename": filename, "error": "Failed to decode image"}
        if result is None:
            return {"index": index, "filename": filename, "error": "Classifier returned None"}
        if bt is not None:
            # Targeted detection — mirror BlurClassifier.detect_blur_type (incl. the
            # min-confidence floor) so the streaming response contract is unchanged.
            return {
                "index": index,
                "filename": filename,
                "blur_type": bt,
                "detected": result["predicted_class"] == bt and result["confidence"] >= min_conf,
                "confidence": result["confidence"],
                "blur_type_probability": result["probabilities"].get(bt, 0.0),
                "predicted_class": result["predicted_class"],
                "probabilities": result["probabilities"],
            }
        return {
            "index": index,
            "filename": filename,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        }

    async def generate():
        sem = asyncio.Semaphore(concurrency)

        async def decode_one(data: bytes):
            async with sem:
                return await asyncio.to_thread(_decode_bgr_for_classify, data, decode_dim)

        # Sequential sub-batches (mirrors the Celery blur_classify_batch worker):
        # decode each sub-batch's images in parallel, then ONE batched inference.
        # classify_batch runs a true (N,3,224,224) ONNX call when the model has a
        # dynamic batch axis, and a per-image loop otherwise — identical results.
        for chunk_start in range(0, total, sub_batch):
            sub = items[chunk_start:chunk_start + sub_batch]
            images = await asyncio.gather(*[decode_one(d) for _, _, d in sub])
            try:
                results = await asyncio.to_thread(classifier.classify_batch, images)
            except Exception as e:
                # The status line and headers are already on the wire, so we
                # cannot turn this into an HTTP error. Letting it propagate would
                # kill the generator mid-stream and hand the desktop a truncated
                # body with no _summary — indistinguishable from success. Emit a
                # per-image error line for this sub-batch instead and keep going,
                # so the stream stays well-formed and every index is accounted for.
                logger.error(
                    "Stream classify sub-batch failed",
                    chunk_start=chunk_start,
                    size=len(sub),
                    error=str(e),
                )
                for idx, name, _ in sub:
                    line = {"index": idx, "filename": name, "error": str(e)}
                    yield json.dumps(line, separators=(",", ":")) + "\n"
                continue
            for (idx, name, _), image, result in zip(sub, images, results):
                yield json.dumps(_build_line(idx, name, image, result), separators=(",", ":")) + "\n"

        # Final summary line
        elapsed = (time.perf_counter() - start) * 1000
        summary = {"_summary": True, "total": total, "processing_time_ms": round(elapsed, 2)}
        yield json.dumps(summary, separators=(",", ":")) + "\n"

    logger.info(
        "Stream blur classification started",
        total=total,
        blur_type=bt,
        request_id=getattr(request.state, "request_id", ""),
    )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"X-Total-Images": str(total)},
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
    check_scope("blur:read", key_meta)
    settings = request.app.state.settings

    result = await validate_batch_files(
        request, files, settings.MAX_BATCH_SIZE, settings.MAX_FILE_SIZE
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, "blur_classify_batch", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    image_paths = store_blobs_and_get_paths(job_id, raw_bytes_list)

    from src.workers.tasks.blur_tasks import blur_classify_batch

    blur_classify_batch.delay(
        job_id, image_paths, blur_type.value if blur_type else None
    )

    return batch_accepted_response(request, job_id, len(files))


@router.post("/detect/mega", status_code=202)
async def detect_blur_mega(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (up to 500)"),
    key_meta: dict = Depends(verify_api_key),
):
    """Submit a mega-batch of images for async blur detection.

    Accepts up to 500 images per request. The server automatically splits
    them into sub-tasks and merges results into a single job.
    """
    check_scope("blur:read", key_meta)
    settings = request.app.state.settings

    result = await validate_batch_files(
        request, files, settings.MEGA_BATCH_MAX_SIZE, settings.MAX_FILE_SIZE
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, "blur_detect_mega", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    from src.workers.tasks.blur_tasks import blur_detect_batch

    dispatch_mega_batch(job_id, raw_bytes_list, blur_detect_batch)

    return batch_accepted_response(request, job_id, len(files))


@router.post("/classify/mega", status_code=202)
async def classify_blur_mega(
    request: Request,
    files: list[UploadFile] = File(..., description="Image files (up to 500)"),
    blur_type: BlurType | None = Query(
        default=None,
        description="Specific blur type to detect in mega-batch mode",
    ),
    key_meta: dict = Depends(verify_api_key),
):
    """Submit a mega-batch of images for async blur classification.

    Accepts up to 500 images per request. The server automatically splits
    them into sub-tasks and merges results into a single job.
    """
    check_scope("blur:read", key_meta)
    settings = request.app.state.settings

    result = await validate_batch_files(
        request, files, settings.MEGA_BATCH_MAX_SIZE, settings.MAX_FILE_SIZE
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, "blur_classify_mega", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    from src.workers.tasks.blur_tasks import blur_classify_batch

    dispatch_mega_batch(
        job_id, raw_bytes_list, blur_classify_batch,
        extra_kwargs={"blur_type": blur_type.value if blur_type else None},
    )

    return batch_accepted_response(request, job_id, len(files))
