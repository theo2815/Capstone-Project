from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import UploadFile
from PIL import Image

from src.utils.exceptions import ImageValidationError
from src.utils.image_utils import (
    MAX_DIMENSION,
    MIN_DIMENSION,
    get_image_dimensions,
    validate_and_decode,
    validate_batch_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_bytes(
    width: int = 100,
    height: int = 100,
    fmt: str = "JPEG",
    color: tuple[int, int, int] = (255, 0, 0),
) -> bytes:
    """Create a valid in-memory image and return its raw bytes."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_upload_file(
    data: bytes,
    content_type: str = "image/jpeg",
    filename: str = "test.jpg",
) -> UploadFile:
    """Wrap raw bytes in a FastAPI UploadFile backed by BytesIO."""
    from starlette.datastructures import Headers

    headers = Headers({"content-type": content_type})
    return UploadFile(file=io.BytesIO(data), filename=filename, headers=headers)


# ===========================================================================
# validate_and_decode
# ===========================================================================


class TestValidateAndDecode:
    """Tests for the async validate_and_decode helper."""

    # 1. Valid JPEG passes and returns (bytes, numpy array)
    async def test_valid_jpeg(self):
        raw = _make_image_bytes(fmt="JPEG")
        upload = _make_upload_file(raw, content_type="image/jpeg")

        contents, image = await validate_and_decode(upload)

        assert isinstance(contents, bytes)
        assert len(contents) == len(raw)
        assert isinstance(image, np.ndarray)

    # 2. Valid PNG passes
    async def test_valid_png(self):
        raw = _make_image_bytes(fmt="PNG")
        upload = _make_upload_file(raw, content_type="image/png", filename="test.png")

        contents, image = await validate_and_decode(upload)

        assert isinstance(contents, bytes)
        assert isinstance(image, np.ndarray)

    # 3. Invalid content type raises ImageValidationError
    async def test_invalid_content_type(self):
        raw = _make_image_bytes()
        upload = _make_upload_file(raw, content_type="text/plain")

        with pytest.raises(ImageValidationError, match="Unsupported file type"):
            await validate_and_decode(upload)

    # 4. File exceeding max_file_size raises ImageValidationError
    async def test_file_exceeds_max_size(self):
        raw = _make_image_bytes()
        upload = _make_upload_file(raw, content_type="image/jpeg")

        # Set a tiny limit so the real image exceeds it
        with pytest.raises(ImageValidationError, match="limit"):
            await validate_and_decode(upload, max_file_size=10)

    # 5. Corrupt image data raises ImageValidationError
    async def test_corrupt_image_data(self):
        corrupt = b"not-an-image-at-all"
        upload = _make_upload_file(corrupt, content_type="image/jpeg")

        with pytest.raises(ImageValidationError, match="Invalid or corrupt"):
            await validate_and_decode(upload)

    # 6. Image too large (dimensions > MAX_DIMENSION) raises ImageValidationError
    async def test_image_too_large(self):
        # A strip, not a square: at MAX_DIMENSION=12000 a square would be
        # 144 Mpx (~432 MB) to allocate. A strip trips the per-side check while
        # staying far under MAX_IMAGE_PIXELS, so it isolates that check exactly.
        raw = _make_image_bytes(width=MAX_DIMENSION + 1, height=MIN_DIMENSION, fmt="PNG")
        upload = _make_upload_file(raw, content_type="image/png", filename="big.png")

        with pytest.raises(ImageValidationError, match="exceed"):
            await validate_and_decode(upload)

    # 6b. Real DSLR dimensions are ACCEPTED — regression guard for the bug where
    # a 4096 cap rejected 3.75% of real photos on the single-image endpoints
    # (the backend forwards originals to /faces/enroll, so those photos ended
    # up IndexingStatus.FAILED and were never searchable by face or bib).
    async def test_dslr_dimensions_are_accepted(self):
        raw = _make_image_bytes(width=6000, height=MIN_DIMENSION, fmt="JPEG")
        upload = _make_upload_file(raw, content_type="image/jpeg", filename="dslr.jpg")

        _, image = await validate_and_decode(upload)

        # Accepted, and downscaled to the inference dimension rather than refused.
        from src.config import get_settings

        assert max(image.shape[:2]) == get_settings().MAX_INFERENCE_DIMENSION

    # 7. Image too small (dimensions < MIN_DIMENSION) raises ImageValidationError
    async def test_image_too_small(self):
        tiny = MIN_DIMENSION - 1
        raw = _make_image_bytes(width=tiny, height=tiny, fmt="PNG")
        upload = _make_upload_file(raw, content_type="image/png", filename="tiny.png")

        with pytest.raises(ImageValidationError, match="too small"):
            await validate_and_decode(upload)

    # 8. Returned numpy array is BGR format (3 channels)
    async def test_returned_array_is_bgr_3_channels(self):
        # Create a pure-red image (RGB = 255,0,0 -> BGR = 0,0,255)
        raw = _make_image_bytes(width=64, height=64, color=(255, 0, 0), fmt="PNG")
        upload = _make_upload_file(raw, content_type="image/png", filename="red.png")

        _, image = await validate_and_decode(upload)

        # 3 channels
        assert image.ndim == 3
        assert image.shape[2] == 3

        # For a pure-red source the BGR pixel should be (0, 0, 255)
        pixel = image[0, 0]
        assert pixel[0] == 0    # B
        assert pixel[1] == 0    # G
        assert pixel[2] == 255  # R

    # 9. EXIF rotation is handled (ImageOps.exif_transpose is called)
    async def test_exif_transpose_is_applied(self):
        raw = _make_image_bytes(fmt="JPEG")
        upload = _make_upload_file(raw, content_type="image/jpeg")

        with patch("src.utils.image_utils.ImageOps.exif_transpose", wraps=__import__("PIL").ImageOps.exif_transpose) as mock_transpose:
            await validate_and_decode(upload)
            mock_transpose.assert_called_once()

    # 10. Decompression bomb protection — MAX_IMAGE_PIXELS is set
    def test_decompression_bomb_protection(self):
        assert Image.MAX_IMAGE_PIXELS == MAX_DIMENSION * MAX_DIMENSION


# ===========================================================================
# validate_batch_file
# ===========================================================================


class TestValidateBatchFile:
    """Tests for the synchronous validate_batch_file helper."""

    # 11. Valid image passes without raising
    def test_valid_image_passes(self):
        raw = _make_image_bytes()
        # Should not raise
        validate_batch_file(raw, filename="ok.jpg", max_file_size=10 * 1024 * 1024)

    # 12. Oversized file raises ImageValidationError
    def test_oversized_file_raises(self):
        raw = _make_image_bytes()
        with pytest.raises(ImageValidationError, match="exceeds"):
            validate_batch_file(raw, filename="big.jpg", max_file_size=10)

    # 13. Corrupt data raises ImageValidationError
    def test_corrupt_data_raises(self):
        with pytest.raises(ImageValidationError, match="not a valid image"):
            validate_batch_file(b"garbage", filename="bad.jpg", max_file_size=10 * 1024 * 1024)

    # 13b. Over-cap dimensions raise. The batch path had NO dimension check at
    # all, which is what let /faces/enroll/mega accept photos /faces/enroll
    # refused.
    def test_oversized_dimensions_raise(self):
        raw = _make_image_bytes(width=MAX_DIMENSION + 1, height=MIN_DIMENSION, fmt="PNG")
        with pytest.raises(ImageValidationError, match="exceed"):
            validate_batch_file(raw, filename="big.png", max_file_size=50 * 1024 * 1024)

    # 13c. ...and the message is the dimension one, not "not a valid image" —
    # the check sits outside the try, so the bare except cannot swallow it.
    def test_oversized_dimensions_report_dimensions(self):
        raw = _make_image_bytes(width=MAX_DIMENSION + 1, height=MIN_DIMENSION, fmt="PNG")
        with pytest.raises(ImageValidationError) as exc:
            validate_batch_file(raw, filename="big.png", max_file_size=50 * 1024 * 1024)
        assert "not a valid image" not in str(exc.value)

    # 13d. Real DSLR dimensions pass the batch path
    def test_dslr_dimensions_are_accepted(self):
        raw = _make_image_bytes(width=6000, height=MIN_DIMENSION, fmt="JPEG")
        validate_batch_file(raw, filename="dslr.jpg", max_file_size=50 * 1024 * 1024)


# ===========================================================================
# get_image_dimensions
# ===========================================================================


class TestGetImageDimensions:
    """Tests for the get_image_dimensions utility."""

    # 14. Returns (width, height) correctly
    def test_returns_width_height(self):
        # numpy shape is (height, width, channels)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        w, h = get_image_dimensions(image)
        assert w == 640
        assert h == 480

    def test_non_square(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        w, h = get_image_dimensions(image)
        assert w == 200
        assert h == 100


# ===========================================================================
# Cross-path parity + calibrated defaults
# ===========================================================================


class TestImageLimitParity:
    """The single-image and batch paths must enforce the SAME dimension bound.

    They did not: validate_and_decode capped at 4096 while validate_batch_file
    had no dimension check at all. The same photo was therefore rejected by
    /faces/enroll and accepted by /faces/enroll/mega, so it indexed or failed
    depending on the backend's INDEXING_MODE. This fails if they drift again.
    """

    @pytest.mark.parametrize(
        "width, expect_rejected",
        [
            (MAX_DIMENSION, False),
            (MAX_DIMENSION + 1, True),
        ],
    )
    async def test_single_and_batch_agree(self, width, expect_rejected):
        raw = _make_image_bytes(width=width, height=MIN_DIMENSION, fmt="PNG")
        size_cap = 50 * 1024 * 1024

        def batch_rejects() -> bool:
            try:
                validate_batch_file(raw, filename="p.png", max_file_size=size_cap)
                return False
            except ImageValidationError:
                return True

        async def single_rejects() -> bool:
            upload = _make_upload_file(raw, content_type="image/png", filename="p.png")
            try:
                await validate_and_decode(upload, max_file_size=size_cap)
                return False
            except ImageValidationError:
                return True

        assert batch_rejects() is expect_rejected
        assert (await single_rejects()) is expect_rejected


class TestCalibratedUploadLimits:
    """Guards on the two upload bounds, mirroring TestWorkerDecodeDimensions."""

    def test_max_image_dimension_is_the_calibrated_12000(self):
        from src.config import Settings

        # Sized to pass any real camera: largest input measured in
        # Training-Images/ is 6000x4000; 12000 clears 102 MP medium format.
        assert Settings().MAX_IMAGE_DIMENSION == 12000

    def test_max_file_size_matches_the_backend_multipart_ceiling(self):
        from src.config import Settings

        # The Spring backend accepts 25 MB and forwards ORIGINAL bytes for
        # indexing, so anything it accepts must be accepted here.
        assert Settings().MAX_FILE_SIZE == 25 * 1024 * 1024

    def test_decompression_bomb_guard_tracks_the_cap(self):
        assert Image.MAX_IMAGE_PIXELS == MAX_DIMENSION * MAX_DIMENSION
