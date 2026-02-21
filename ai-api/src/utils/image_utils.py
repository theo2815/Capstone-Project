from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image

from src.utils.exceptions import ImageValidationError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_DIMENSION = 4096
MIN_DIMENSION = 32


async def validate_and_decode(
    file: UploadFile,
    max_file_size: int = 10 * 1024 * 1024,
) -> tuple[bytes, np.ndarray]:
    """Validate an uploaded image and decode it to a numpy array.

    Returns:
        Tuple of (raw_bytes, bgr_numpy_array)

    Raises:
        ImageValidationError: If the image fails validation.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )

    contents = await file.read()
    if len(contents) > max_file_size:
        raise ImageValidationError(
            f"File exceeds {max_file_size // (1024 * 1024)}MB limit"
        )

    # Verify it's actually a valid image (magic bytes, not just Content-Type)
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise ImageValidationError("Invalid or corrupt image file")

    # Check dimensions
    img = Image.open(io.BytesIO(contents))
    w, h = img.size
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({w}x{h}) exceed {MAX_DIMENSION}px limit"
        )
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image too small ({w}x{h}). Minimum is {MIN_DIMENSION}px"
        )

    # Decode to numpy BGR array (strip EXIF for privacy)
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageValidationError("Failed to decode image")

    return contents, image


def get_image_dimensions(image: np.ndarray) -> tuple[int, int]:
    """Return (width, height) of a BGR numpy image."""
    h, w = image.shape[:2]
    return w, h
