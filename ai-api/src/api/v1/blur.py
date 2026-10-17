from __future__ import annotations

import asyncio
import json
import time
import uuid
from functools import partial
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.responses import StreamingResponse

from src.api.v1.batch_utils import (
    batch_accepted_response,
    create_batch_job,
    reject_if_at_job_cap,
    store_blobs_and_get_paths,
    validate_batch_files,
)
from src.api.v1.mega_batch import dispatch_mega_batch
from src.middleware.auth import check_scope, verify_api_key
from src.schemas.blur import (
    BlurClassificationResponse,
    BlurClassProbabilities,
    BlurDetectionResponse,
    BlurMetrics,
    BlurType,
    BlurTypeDetectionResponse,
)
from src.schemas.common import APIResponse
from src.utils.image_utils import (
    bounded_to_thread,
    decode_gray_for_blur,
    downscale_for_inference,
    get_image_dimensions,
    validate_and_decode,
    validate_stream_file,
)
from src.utils.blur_metrics import images, first_result, stream_duration, stream_outcomes, timeouts
from src.utils.logging import get_logger
from src.utils.timeout import (
    InferenceBusyError,
    InferenceTimeoutError,
    run_with_timeout,
    tracked_to_thread,
)

logger = get_logger(__name__)


async def _submit_blur_job(request, job_id, raw_bytes_list, task, *, mega=False, blur_type=None):
    """Stage every input before dispatch and terminally fail an unaccepted submission."""
    try:
        classify = task.name == "blur.classify_batch"
        if mega:
            kwargs = {"blur_type": blur_type} if classify else None
            await asyncio.to_thread(dispatch_mega_batch, job_id, raw_bytes_list, task,
                                    extra_kwargs=kwargs)
        else:
            keys = await asyncio.to_thread(store_blobs_and_get_paths, job_id, raw_bytes_list)
            if classify:
                await asyncio.to_thread(task.delay, job_id, keys, blur_type)
            else:
                await asyncio.to_thread(task.delay, job_id, keys)
        return None
    except Exception as exc:
        from src.db.repositories.job_repo import JobRepository
        from src.db.session import get_session_ctx
        from src.utils.blob_store import cleanup_batch

        logger.error("Blur async submission failed", job_id=job_id,
                     error_type=type(exc).__name__)
        async with get_session_ctx() as session:
            failed = await JobRepository(session).fail(uuid.UUID(job_id), "Submission failed")
        if failed:
            try:
                await asyncio.to_thread(cleanup_batch, job_id)
            except Exception as cleanup_error:
                logger.error("Blur submission cleanup failed", job_id=job_id,
                             error_type=type(cleanup_error).__name__)
        return JSONResponse(status_code=503, content=APIResponse(
            success=False, request_id=getattr(request.state, "request_id", ""),
            error={"code": "ASYNC_UNAVAILABLE", "message": "Async job submission failed"},
        ).model_dump(mode="json"))


def _inference_error_response(request: Request, error: Exception) -> JSONResponse:
    busy = isinstance(error, InferenceBusyError)
    return JSONResponse(
        status_code=503 if busy else 504,
        headers={"Retry-After": "1"} if busy else None,
        content=APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={
                "code": "BLUR_BUSY" if busy else "INFERENCE_TIMEOUT",
                "message": "Blur service is busy" if busy else "Blur inference timed out",
            },
        ).model_dump(mode="json"),
    )


class BlurAuthRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def authenticate_before_form(request: Request):
            key_meta = await verify_api_key(request, request.headers.get("X-API-Key"))
            check_scope("blur:read", key_meta)
            request.state.blur_key_meta = key_meta
            admit = getattr(request.state, "blur_admit", None)
            refusal = await admit(key_meta["key_id"]) if admit is not None else None
            if refusal == "key":
                from src.utils.blur_metrics import rejected_requests

                rejected_requests.labels("key_busy").inc()
                return JSONResponse(status_code=429, headers={"Retry-After": "1"}, content={
                    "success": False,
                    "request_id": getattr(request.state, "request_id", ""),
                    "error": {"code": "BLUR_KEY_BUSY",
                              "message": "This API key already has a blur request running"},
                })
            if refusal:
                from src.utils.blur_metrics import rejected_requests

                rejected_requests.labels("busy").inc()
                return JSONResponse(status_code=503, headers={"Retry-After": "1"}, content={
                    "success": False,
                    "request_id": getattr(request.state, "request_id", ""),
                    "error": {"code": "BLUR_BUSY", "message": "Blur service is busy"},
                })
            operation = request.url.path.rsplit("/", 1)[-1]
            if operation not in ("detect", "classify"):
                return await handler(request)
            images.labels(operation, "attempted").inc()
            try:
                response = await handler(request)
            except BaseException:
                images.labels(operation, "failed").inc()
                raise
            images.labels(operation, "succeeded" if response.status_code < 400 else "failed").inc()
            return response

        return authenticate_before_form


router = APIRouter(prefix="/blur", tags=["Blur Detection"], route_class=BlurAuthRoute)


def _downscale_stream_image(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    if min(h, w) * scale <= 0.5:
        return cv2.resize(
            image, (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Fast decode + detect for streaming batch (bypasses full PIL pipeline)
# ---------------------------------------------------------------------------

def _fast_decode_and_detect(
    data: bytes,
    filename: str,
    index: int,
    detector: Any,
    threshold: float | None,
    decode_dim: int,
    include_hf_ratio: bool = False,
    request_id: str = "",
) -> dict:
    """Decode an image with OpenCV and run blur detection in one call.

    Uses the same grayscale decode, scale, and detector score as single and worker.
    """
    try:
        gray = decode_gray_for_blur(data, decode_dim)
        if gray is None:
            return {"index": index, "filename": filename, "error": "Failed to decode image"}
        result = (detector.detect(gray, threshold_override=threshold)
                  if include_hf_ratio else detector.detect_fast(gray, threshold_override=threshold))
        return {"index": index, "filename": filename, **result}
    except Exception as e:
        logger.warning(
            "Stream image processing failed", error_type=type(e).__name__, request_id=request_id,
        )
        return {"index": index, "filename": filename, "error": "Failed to process image"}


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
    return _downscale_stream_image(image, max_dim)


@router.post("/detect", response_model=APIResponse)
async def detect_blur(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    threshold: float | None = Query(default=None, ge=1.0, le=10000.0),
    include_metrics: bool = Query(default=True),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect if an image is blurry."""
    check_scope("blur:read", key_meta)
    start = time.perf_counter()

    settings = request.app.state.settings
    if settings.MODEL_PROFILE == "blur":
        _, image = await validate_and_decode(
            file, max_file_size=settings.MAX_FILE_SIZE,
            max_pixels=settings.BLUR_MAX_IMAGE_PIXELS,
            decode_semaphore=request.app.state.blur_decode_semaphore,
            grayscale_max_dim=settings.BLUR_DETECT_DECODE_DIM,
        )
    else:
        _, image = await validate_and_decode(
            file, max_file_size=settings.MAX_FILE_SIZE,
            grayscale_max_dim=settings.BLUR_DETECT_DECODE_DIM,
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

    try:
        if settings.MODEL_PROFILE == "blur":
            result = await tracked_to_thread(
                run_with_timeout, detector.detect, args=(image,),
                kwargs={"threshold_override": threshold, "include_hf_ratio": include_metrics},
                timeout_seconds=settings.BLUR_INFERENCE_TIMEOUT_SECONDS,
            )
        else:
            result = await tracked_to_thread(
                detector.detect, image, threshold_override=threshold,
                include_hf_ratio=include_metrics,
            )
    except (InferenceBusyError, InferenceTimeoutError) as error:
        return _inference_error_response(request, error)
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

    total = len(files)
    images.labels("detect", "attempted").inc(total)
    start = time.perf_counter()
    deadline = time.monotonic() + settings.BLUR_STREAM_TIMEOUT_SECONDS
    request_id = getattr(request.state, "request_id", "")

    async def generate():
        successful = errors = 0
        complete = True
        first_seen = False
        sem = (
            request.app.state.blur_decode_semaphore
            if settings.MODEL_PROFILE == "blur"
            else asyncio.Semaphore(concurrency)
        )

        async def process(index: int, filename: str, data: bytes, error: str | None) -> dict:
            if error is not None:
                # Rejected before decode: no semaphore slot, no worker thread.
                return {"index": index, "filename": filename, "error": error}
            try:
                if settings.MODEL_PROFILE == "blur":
                    work = partial(
                        run_with_timeout, _fast_decode_and_detect,
                        args=(data, filename, index, detector, threshold,
                              settings.BLUR_DETECT_DECODE_DIM, include_hf_ratio, request_id),
                        timeout_seconds=settings.BLUR_INFERENCE_TIMEOUT_SECONDS,
                    )
                    return await bounded_to_thread(sem, work)
                return await bounded_to_thread(
                    sem, _fast_decode_and_detect, data, filename, index, detector,
                    threshold, settings.BLUR_DETECT_DECODE_DIM, include_hf_ratio, request_id,
                )
            except (InferenceBusyError, InferenceTimeoutError):
                raise
            except Exception as e:
                logger.error(
                    "Stream image processing failed", error_type=type(e).__name__,
                    request_id=request_id,
                )
                return {"index": index, "filename": filename, "error": "Failed to process image"}

        try:
            for start_idx in range(0, total, settings.INFERENCE_SUB_BATCH_SIZE):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Blur stream deadline exceeded")
                if await request.is_disconnected():
                    return
                items = []
                end_idx = min(start_idx + settings.INFERENCE_SUB_BATCH_SIZE, total)
                for i in range(start_idx, end_idx):
                    f = files[i]
                    data = await f.read(settings.MAX_FILE_SIZE + 1)
                    name = f.filename or f"image_{i}"
                    err = validate_stream_file(
                        data, name, settings.MAX_FILE_SIZE,
                        settings.BLUR_MAX_IMAGE_PIXELS if settings.MODEL_PROFILE == "blur" else 0,
                    )
                    items.append((i, name, data, err))
                tasks = [
                    asyncio.create_task(process(i, name, data, err))
                    for i, name, data, err in items
                ]
                try:
                    for coro in asyncio.as_completed(tasks):
                        result = await asyncio.wait_for(
                            coro, timeout=max(0, deadline - time.monotonic())
                        )
                        if not first_seen:
                            first_result.labels("detect").observe(time.perf_counter() - start)
                            first_seen = True
                        successful += "error" not in result
                        errors += "error" in result
                        images.labels(
                            "detect", "failed" if "error" in result else "succeeded"
                        ).inc()
                        yield json.dumps(result, separators=(",", ":")) + "\n"
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            complete = False
            if isinstance(e, TimeoutError):
                timeouts.labels("stream").inc()
            logger.error(
                "Stream blur detection failed", error_type=type(e).__name__, request_id=request_id,
            )
        finally:
            for f in files:
                await f.close()
            unaccounted = total - successful - errors
            if unaccounted:
                images.labels("detect", "unaccounted").inc(unaccounted)
            outcome = "completed" if complete and not unaccounted else "interrupted"
            stream_outcomes.labels("detect", outcome).inc()
            stream_duration.labels("detect").observe(time.perf_counter() - start)
            logger.info("Stream blur detection finished", request_id=request_id,
                        outcome=outcome, successful=successful, errors=errors,
                        unaccounted=unaccounted)

        # Final summary line
        elapsed = (time.perf_counter() - start) * 1000
        summary = {"_summary": True, "total": total, "successful": successful,
                   "errors": errors, "complete": complete, "processing_time_ms": round(elapsed, 2)}
        yield json.dumps(summary, separators=(",", ":")) + "\n"

    logger.info(
        "Stream blur detection started", total=total,
        request_id=getattr(request.state, "request_id", ""),
    )

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

    refused = await reject_if_at_job_cap(request, key_meta.get("key_id"))
    if refused is not None:
        return refused

    result = await validate_batch_files(
        request, files, settings.MAX_BATCH_SIZE, settings.MAX_FILE_SIZE,
        settings.BLUR_MAX_IMAGE_PIXELS if settings.MODEL_PROFILE == "blur" else 0,
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, "blur_detect_batch", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    from src.workers.tasks.blur_tasks import blur_detect_batch
    failure = await _submit_blur_job(request, job_id, raw_bytes_list, blur_detect_batch)
    if failure is not None:
        return failure

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
    if settings.MODEL_PROFILE == "blur":
        _, image = await validate_and_decode(
            file, max_file_size=settings.MAX_FILE_SIZE,
            max_pixels=settings.BLUR_MAX_IMAGE_PIXELS,
            decode_semaphore=request.app.state.blur_decode_semaphore,
        )
    else:
        _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)
    # Reported before the classify downscale so the response keeps meaning the
    # same thing it did before.
    reported_w, reported_h = get_image_dimensions(image)
    # validate_and_decode caps at MAX_INFERENCE_DIMENSION (1280), which is sized
    # for bib OCR, not for this model. Take it the rest of the way down to the
    # calibrated classify dimension so this endpoint agrees with
    # /blur/classify/stream and the blur_classify_batch worker. Measured on the
    # 1057-image val set, holding preprocess constant: at 640 this endpoint now
    # scores the same as the stream path (4/698 blurry called sharp); at 1280 it
    # scored 7. See BLUR_CLASSIFY_DECODE_DIM in config.py.
    image = downscale_for_inference(image, settings.BLUR_CLASSIFY_DECODE_DIM)

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
        try:
            result = await tracked_to_thread(classifier.detect_blur_type, image, blur_type.value)
        except (InferenceBusyError, InferenceTimeoutError) as error:
            return _inference_error_response(request, error)
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

        response_data = BlurTypeDetectionResponse(
            blur_type=result["blur_type"],
            detected=result["detected"],
            confidence=result["confidence"],
            blur_type_probability=result["blur_type_probability"],
            image_dimensions=(reported_w, reported_h),
            processing_time_ms=round(elapsed_ms, 2),
        )
    else:
        # Full classification mode (backward compatible)
        try:
            result = await tracked_to_thread(classifier.classify, image)
        except (InferenceBusyError, InferenceTimeoutError) as error:
            return _inference_error_response(request, error)
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

        response_data = BlurClassificationResponse(
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=BlurClassProbabilities(**result["probabilities"]),
            image_dimensions=(reported_w, reported_h),
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

    total = len(files)
    images.labels("classify", "attempted").inc(total)
    bt = blur_type.value if blur_type else None
    sub_batch = settings.INFERENCE_SUB_BATCH_SIZE
    decode_dim = settings.BLUR_CLASSIFY_DECODE_DIM
    min_conf = classifier.min_detection_confidence
    start = time.perf_counter()
    deadline = time.monotonic() + settings.BLUR_STREAM_TIMEOUT_SECONDS

    def _build_line(index: int, filename: str, image, result, error: str | None) -> dict:
        if error is not None:
            # More specific than the generic decode failure below — say which
            # limit the file broke.
            return {"index": index, "filename": filename, "error": error}
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
        successful = errors = 0
        complete = True
        first_seen = False
        phase = "stream"
        request_id = getattr(request.state, "request_id", "")
        snapshot = getattr(classifier, "timing_snapshot", dict)
        timing_before = snapshot()
        decode_seconds = 0.0
        sem = (
            request.app.state.blur_decode_semaphore
            if settings.MODEL_PROFILE == "blur"
            else asyncio.Semaphore(concurrency)
        )

        async def decode_one(data: bytes, error: str | None):
            if error is not None:
                # Rejected before decode; only valid arrays reach inference.
                return None
            try:
                return await bounded_to_thread(sem, _decode_bgr_for_classify, data, decode_dim)
            except (cv2.error, ValueError, OverflowError) as e:
                logger.warning(
                    "Stream image decode failed", error_type=type(e).__name__,
                    request_id=request_id,
                )
                return None

        # Sequential sub-batches (mirrors the Celery blur_classify_batch worker):
        # decode each sub-batch's images in parallel, then ONE batched inference.
        # classify_batch runs a true (N,3,224,224) ONNX call when the model has a
        # dynamic batch axis, and a per-image loop otherwise — identical results.
        try:
            for chunk_start in range(0, total, sub_batch):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Blur stream deadline exceeded")
                if await request.is_disconnected():
                    return
                sub = []
                for i in range(chunk_start, min(chunk_start + sub_batch, total)):
                    f = files[i]
                    data = await f.read(settings.MAX_FILE_SIZE + 1)
                    name = f.filename or f"image_{i}"
                    err = validate_stream_file(
                        data, name, settings.MAX_FILE_SIZE,
                        settings.BLUR_MAX_IMAGE_PIXELS if settings.MODEL_PROFILE == "blur" else 0,
                    )
                    sub.append((i, name, data, err))
                phase = "decode"
                decode_started = time.perf_counter()
                decoded = await asyncio.wait_for(
                    asyncio.gather(*[decode_one(d, e) for _, _, d, e in sub]),
                    timeout=max(0, deadline - time.monotonic()),
                )
                decode_seconds += time.perf_counter() - decode_started
                valid = [i for i, ((_, _, _, err), image) in enumerate(zip(sub, decoded))
                         if err is None and image is not None]
                results = [None] * len(sub)
                phase = "inference"
                try:
                    if valid:
                        if getattr(classifier, "_dynamic_batch", True):
                            inferred = await asyncio.wait_for(
                                tracked_to_thread(
                                    classifier.classify_batch, [decoded[i] for i in valid]
                                ),
                                timeout=max(0, deadline - time.monotonic()),
                            )
                        else:
                            inferred = []
                            for i in valid:
                                if await request.is_disconnected():
                                    return
                                inferred.append(await asyncio.wait_for(
                                    tracked_to_thread(classifier.classify, decoded[i]),
                                    timeout=max(0, deadline - time.monotonic()),
                                ))
                        if len(inferred) != len(valid):
                            raise ValueError("Classifier result count mismatch")
                        for i, result in zip(valid, inferred):
                            results[i] = result
                except (InferenceBusyError, InferenceTimeoutError, TimeoutError):
                    raise
                except Exception as e:
                    # Headers are already sent; preserve the stream's indexed rows.
                    logger.error(
                        "Stream classify sub-batch failed",
                        chunk_start=chunk_start,
                        size=len(sub),
                        error_type=type(e).__name__,
                        request_id=request_id,
                    )
                    for (idx, name, _, err), image in zip(sub, decoded):
                        # A rejected file keeps its own reason.
                        line = {
                            "index": idx, "filename": name,
                            "error": err or (
                                "Failed to decode image"
                                if image is None else "Classification failed"
                            ),
                        }
                        errors += 1
                        images.labels("classify", "failed").inc()
                        if not first_seen:
                            first_result.labels("classify").observe(time.perf_counter() - start)
                            first_seen = True
                        yield json.dumps(line, separators=(",", ":")) + "\n"
                    continue
                for (idx, name, _, err), image, result in zip(sub, decoded, results):
                    line = _build_line(idx, name, image, result, err)
                    if not first_seen:
                        first_result.labels("classify").observe(time.perf_counter() - start)
                        first_seen = True
                    successful += "error" not in line
                    errors += "error" in line
                    images.labels("classify", "failed" if "error" in line else "succeeded").inc()
                    yield json.dumps(line, separators=(",", ":")) + "\n"
                phase = "stream"
        except Exception as e:
            complete = False
            if isinstance(e, TimeoutError):
                timeouts.labels(phase).inc()
            logger.error(
                "Stream blur classification failed", error_type=type(e).__name__,
                request_id=request_id,
            )
        finally:
            for f in files:
                await f.close()
            unaccounted = total - successful - errors
            if unaccounted:
                images.labels("classify", "unaccounted").inc(unaccounted)
            outcome = "completed" if complete and not unaccounted else "interrupted"
            stream_outcomes.labels("classify", outcome).inc()
            stream_duration.labels("classify").observe(time.perf_counter() - start)
            per_photo = 1000 / max(total, 1)
            stages = snapshot()
            logger.info("Stream blur classification finished", request_id=request_id,
                        outcome=outcome, successful=successful, errors=errors,
                        unaccounted=unaccounted,
                        ms_per_photo=round((time.perf_counter() - start) * per_photo, 1),
                        decode_ms_per_photo=round(decode_seconds * per_photo, 1),
                        **{f"{stage}_ms_per_photo": round(
                            (seconds - timing_before.get(stage, 0.0)) * per_photo, 1)
                           for stage, seconds in stages.items()})

        # Final summary line
        elapsed = (time.perf_counter() - start) * 1000
        summary = {"_summary": True, "total": total, "successful": successful,
                   "errors": errors, "complete": complete, "processing_time_ms": round(elapsed, 2)}
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

    refused = await reject_if_at_job_cap(request, key_meta.get("key_id"))
    if refused is not None:
        return refused

    result = await validate_batch_files(
        request, files, settings.MAX_BATCH_SIZE, settings.MAX_FILE_SIZE,
        settings.BLUR_MAX_IMAGE_PIXELS if settings.MODEL_PROFILE == "blur" else 0,
    )
    if isinstance(result, JSONResponse):
        return result
    raw_bytes_list = result

    job_id = await create_batch_job(
        request, "blur_classify_batch", len(files), key_meta.get("key_id")
    )
    if isinstance(job_id, JSONResponse):
        return job_id

    from src.workers.tasks.blur_tasks import blur_classify_batch
    failure = await _submit_blur_job(request, job_id, raw_bytes_list,
                                     blur_classify_batch, blur_type=blur_type.value if blur_type else None)
    if failure is not None:
        return failure

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

    refused = await reject_if_at_job_cap(request, key_meta.get("key_id"))
    if refused is not None:
        return refused

    result = await validate_batch_files(
        request, files, settings.MEGA_BATCH_MAX_SIZE, settings.MAX_FILE_SIZE,
        settings.BLUR_MAX_IMAGE_PIXELS if settings.MODEL_PROFILE == "blur" else 0,
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

    failure = await _submit_blur_job(request, job_id, raw_bytes_list, blur_detect_batch,
                                     mega=True)
    if failure is not None:
        return failure

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

    refused = await reject_if_at_job_cap(request, key_meta.get("key_id"))
    if refused is not None:
        return refused

    result = await validate_batch_files(
        request, files, settings.MEGA_BATCH_MAX_SIZE, settings.MAX_FILE_SIZE,
        settings.BLUR_MAX_IMAGE_PIXELS if settings.MODEL_PROFILE == "blur" else 0,
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

    failure = await _submit_blur_job(request, job_id, raw_bytes_list, blur_classify_batch,
                                     mega=True, blur_type=blur_type.value if blur_type else None)
    if failure is not None:
        return failure

    return batch_accepted_response(request, job_id, len(files))
