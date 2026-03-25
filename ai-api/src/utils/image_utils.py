from __future__ import annotations

import asyncio
import io

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps

from src.utils.exceptions import ImageValidationError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_DIMENSION = 4096
MIN_DIMENSION = 32

# Prevent Pillow decompression bombs (matches MAX_DIMENSION)
Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION


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

    # Offload all CPU-bound image decoding to a thread so the async event
    # loop is not blocked (~100-500ms for large images).
    image = await asyncio.to_thread(_decode_image_bytes, contents)

    return contents, image


def _decode_image_bytes(contents: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array (CPU-bound, runs in thread).

    Validates magic bytes, applies EXIF rotation, checks dimensions, and
    downscales for inference.
    """
    # Verify it's actually a valid image (magic bytes, not just Content-Type)
    try:
        img = Image.open(io.BytesIO(contents))
        img.verify()
    except Exception:
        raise ImageValidationError("Invalid or corrupt image file")

    # Re-open after verify (verify() invalidates the image object), then
    # apply EXIF rotation and convert to numpy — single decode, no cv2.imdecode
    img = Image.open(io.BytesIO(contents))
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({w}x{h}) exceed {MAX_DIMENSION}px limit"
        )
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image too small ({w}x{h}). Minimum is {MIN_DIMENSION}px"
        )

    # Convert PIL -> numpy RGB -> BGR (replaces separate cv2.imdecode)
    rgb_array = np.array(img.convert("RGB"))
    image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Downscale large images before inference (models resize to 640px internally)
    image = downscale_for_inference(image)

    return image


def downscale_for_inference(image: np.ndarray) -> np.ndarray:
    """Downscale image if it exceeds MAX_INFERENCE_DIMENSION.

    Returns the original image if downscaling is disabled (0) or not needed.
    """
    from src.config import get_settings

    max_dim = get_settings().MAX_INFERENCE_DIMENSION
    if max_dim <= 0:
        return image

    h, w = image.shape[:2]
    if w <= max_dim and h <= max_dim:
        return image

    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def validate_batch_file(raw: bytes, filename: str, max_file_size: int) -> None:
    """Validate a single file in a batch upload (size and content type).

    Raises:
        ImageValidationError: If the file fails validation.
    """
    if len(raw) > max_file_size:
        raise ImageValidationError(
            f"File '{filename}' exceeds {max_file_size // (1024 * 1024)}MB limit"
        )
    try:
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except Exception:
        raise ImageValidationError(f"File '{filename}' is not a valid image")


def get_image_dimensions(image: np.ndarray) -> tuple[int, int]:
    """Return (width, height) of a BGR numpy image."""
    h, w = image.shape[:2]
    return w, h
