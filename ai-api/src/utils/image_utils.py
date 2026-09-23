from __future__ import annotations

import asyncio
import io
from typing import Callable, TypeVar

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image, ImageOps

from src.config import get_settings
from src.utils.exceptions import ImageValidationError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
# Calibrated upload bound — see MAX_IMAGE_DIMENSION in config.py.
MAX_DIMENSION = get_settings().MAX_IMAGE_DIMENSION
MIN_DIMENSION = 32
T = TypeVar("T")

# Prevent Pillow decompression bombs (matches MAX_DIMENSION)
Image.MAX_IMAGE_PIXELS = MAX_DIMENSION * MAX_DIMENSION


async def validate_and_decode(
    file: UploadFile,
    max_file_size: int = 0,
    max_pixels: int = 0,
    decode_semaphore: asyncio.Semaphore | None = None,
) -> tuple[bytes, np.ndarray]:
    """Validate an uploaded image and decode it to a numpy array.

    Args:
        file: The uploaded file.
        max_file_size: Override in bytes. 0 (default) uses MAX_FILE_SIZE.
        max_pixels: Optional decoded-pixel cap for the blur profile.
        decode_semaphore: Optional process-wide blur decode limit.

    Returns:
        Tuple of (raw_bytes, bgr_numpy_array)

    Raises:
        ImageValidationError: If the image fails validation.
    """
    if max_file_size <= 0:
        max_file_size = get_settings().MAX_FILE_SIZE

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            f"Unsupported file type: {file.content_type}. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )

    contents = await file.read(max_file_size + 1)
    if len(contents) > max_file_size:
        raise ImageValidationError(
            f"File exceeds {max_file_size // (1024 * 1024)}MB limit"
        )

    # Offload all CPU-bound image decoding to a thread so the async event
    # loop is not blocked (~100-500ms for large images).
    image = await bounded_to_thread(decode_semaphore, _decode_image_bytes, contents, max_pixels)

    return contents, image


async def bounded_to_thread(
    semaphore: asyncio.Semaphore | None, fn: Callable[..., T], *args
) -> T:
    """Keep a decode slot occupied until the native thread really exits."""
    if semaphore is None:
        return await asyncio.to_thread(fn, *args)
    await semaphore.acquire()
    try:
        task = asyncio.create_task(asyncio.to_thread(fn, *args))
    except BaseException:
        semaphore.release()
        raise

    def release(finished: asyncio.Task):
        semaphore.release()
        if not finished.cancelled():
            finished.exception()  # Consume errors if the request was cancelled.

    task.add_done_callback(release)
    return await asyncio.shield(task)


def _decode_image_bytes(contents: bytes, max_pixels: int = 0) -> np.ndarray:
    """Decode raw image bytes to a BGR numpy array (CPU-bound, runs in thread).

    Validates magic bytes, applies EXIF rotation, checks dimensions, and
    downscales for inference.
    """
    # Single open + EXIF transpose (no .verify() — the decode itself catches
    # corrupt files, and .verify() forces a full decode then invalidates the
    # object, requiring a wasteful second open).
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise ImageValidationError("Invalid or corrupt image file") from None

    if img.format not in {"JPEG", "PNG", "WEBP"}:
        raise ImageValidationError("Unsupported image format")

    w, h = img.size
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({w}x{h}) exceed {MAX_DIMENSION}px limit"
        )
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image too small ({w}x{h}). Minimum is {MIN_DIMENSION}px"
        )
    if max_pixels and w * h > max_pixels:
        raise ImageValidationError(f"Image exceeds {max_pixels} pixel limit")

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        raise ImageValidationError("Invalid or corrupt image file") from None

    # Skip .convert("RGB") when already RGB (always true for JPEG after EXIF
    # transpose). np.asarray gives a zero-copy view when possible.
    if img.mode != "RGB":
        img = img.convert("RGB")
    rgb_array = np.asarray(img)
    image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    # Downscale large images before inference (models resize to 640px internally)
    image = downscale_for_inference(image)

    return image


def downscale_for_inference(image: np.ndarray, max_dim: int = 0) -> np.ndarray:
    """Downscale image if it exceeds *max_dim* (or MAX_INFERENCE_DIMENSION).

    Args:
        image: BGR numpy array.
        max_dim: Override dimension. 0 (default) uses the global setting.

    Returns the original image if downscaling is disabled (0) or not needed.
    """
    if max_dim <= 0:
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


def validate_batch_file(
    raw: bytes, filename: str, max_file_size: int, content_type: str | None = None,
    max_pixels: int = 0,
) -> None:
    """Validate a single file in a batch upload (size, content type, magic bytes,
    dimensions).

    Enforces the same MAX_DIMENSION bound as validate_and_decode. The batch
    path used to skip it, so an over-cap photo was rejected by /faces/enroll
    but accepted by /faces/enroll/mega — the same image indexed or failed
    depending on the backend's INDEXING_MODE.

    Raises:
        ImageValidationError: If the file fails validation.
    """
    if len(raw) > max_file_size:
        raise ImageValidationError(
            f"File '{filename}' exceeds {max_file_size // (1024 * 1024)}MB limit"
        )
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            f"File '{filename}' has unsupported type: {content_type}. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )
    try:
        img = Image.open(io.BytesIO(raw))
        # .size is populated by open() from the header.
        w, h = img.size
        if img.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("unsupported image format")
    except Exception:
        raise ImageValidationError(f"File '{filename}' is not a valid image") from None

    # Outside the try: an ImageValidationError raised inside would be swallowed
    # by the bare `except` above and mis-reported as a corrupt file.
    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        raise ImageValidationError(
            f"File '{filename}' dimensions ({w}x{h}) exceed {MAX_DIMENSION}px limit"
        )
    if max_pixels and w * h > max_pixels:
        raise ImageValidationError(f"File '{filename}' exceeds {max_pixels} pixel limit")

    try:
        img.verify()
    except Exception:
        raise ImageValidationError(f"File '{filename}' is not a valid image") from None


def validate_stream_file(
    raw: bytes, filename: str, max_file_size: int, max_pixels: int = 0
) -> str | None:
    """Validate one file on the streaming path. Returns an error message, or None.

    A lighter sibling of validate_batch_file rather than a flagged variant of it.
    It returns instead of raising because the stream endpoints answer a bad file
    with a per-image NDJSON error line, not by failing the whole request.

    Deliberately narrower than validate_batch_file: it reads only the header,
    leaving full decode to OpenCV. It accepts octet-stream multipart parts when
    their actual format is JPEG, PNG, or WebP.

    What is left is the pair neither of those covers: the bytes admitted into
    memory, and the pixel count a decode would allocate. cv2.imdecode has no
    decompression-bomb guard of its own, so the header read below is the only
    thing standing between a malicious or malformed header and the allocation.
    """
    if len(raw) > max_file_size:
        return f"File '{filename}' exceeds {max_file_size // (1024 * 1024)}MB limit"

    try:
        image = Image.open(io.BytesIO(raw))
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            return f"File '{filename}' has unsupported image format"
        w, h = image.size
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        # Over Image.MAX_IMAGE_PIXELS (MAX_DIMENSION squared) — PIL refuses to
        # report a size, so report the cap rather than dimensions we never read.
        return f"File '{filename}' exceeds the {MAX_DIMENSION}px limit"
    except Exception:
        return "Failed to decode image"

    if w > MAX_DIMENSION or h > MAX_DIMENSION:
        return f"File '{filename}' dimensions ({w}x{h}) exceed {MAX_DIMENSION}px limit"
    if max_pixels and w * h > max_pixels:
        return f"File '{filename}' exceeds {max_pixels} pixel limit"
    return None


def get_image_dimensions(image: np.ndarray) -> tuple[int, int]:
    """Return (width, height) of a BGR numpy image."""
    h, w = image.shape[:2]
    return w, h
