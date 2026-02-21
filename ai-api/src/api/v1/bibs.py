from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Request, UploadFile

from src.middleware.auth import verify_api_key
from src.schemas.bibs import BibCandidate, BibDetection, BibRecognitionResponse
from src.schemas.common import APIResponse
from src.utils.image_utils import get_image_dimensions, validate_and_decode

router = APIRouter(prefix="/bibs", tags=["Bib Number Recognition"])


@router.post("/recognize", response_model=APIResponse)
async def recognize_bibs(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    key_meta: dict = Depends(verify_api_key),
) -> APIResponse:
    """Detect bib regions and recognize bib numbers via OCR.

    If the bib detection model (YOLO) is loaded, runs the full pipeline:
    detect bib regions -> crop -> OCR on each region.

    If only the OCR model is available, runs OCR directly on the full image
    (fallback mode).
    """
    start = time.perf_counter()
    settings = request.app.state.settings
    _, image = await validate_and_decode(file, max_file_size=settings.MAX_FILE_SIZE)

    registry = request.app.state.model_registry
    bib_detector = registry.get("bib_detector")
    bib_ocr = registry.get("bib_ocr")

    if bib_ocr is None:
        return APIResponse(
            success=False,
            request_id=getattr(request.state, "request_id", ""),
            error={"code": "MODEL_UNAVAILABLE", "message": "Bib OCR model not loaded"},
        )

    w, h = get_image_dimensions(image)
    bib_results = []

    if bib_detector is not None and bib_detector.model is not None:
        # Full pipeline: detect bib regions -> crop -> OCR
        detections = bib_detector.detect(image)
        for det in detections:
            bbox = det["bbox"]
            x1, y1 = int(bbox["x1"]), int(bbox["y1"])
            x2, y2 = int(bbox["x2"]), int(bbox["y2"])
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            ocr_result = bib_ocr.recognize(cropped)
            if ocr_result["bib_number"]:
                bib_results.append(
                    BibDetection(
                        bib_number=ocr_result["bib_number"],
                        confidence=ocr_result["confidence"],
                        bbox=bbox,
                        all_candidates=[
                            BibCandidate(**c) for c in ocr_result["all_candidates"]
                        ],
                    )
                )
    else:
        # Fallback: run OCR on the full image
        ocr_result = bib_ocr.recognize(image)
        if ocr_result["bib_number"]:
            bib_results.append(
                BibDetection(
                    bib_number=ocr_result["bib_number"],
                    confidence=ocr_result["confidence"],
                    bbox={"x1": 0.0, "y1": 0.0, "x2": float(w), "y2": float(h)},
                    all_candidates=[
                        BibCandidate(**c) for c in ocr_result["all_candidates"]
                    ],
                )
            )

    elapsed_ms = (time.perf_counter() - start) * 1000

    data = BibRecognitionResponse(
        bibs_detected=len(bib_results),
        detections=bib_results,
        image_dimensions=(w, h),
        processing_time_ms=round(elapsed_ms, 2),
    )
    return APIResponse(
        success=True,
        request_id=getattr(request.state, "request_id", ""),
        data=data.model_dump(),
    )
