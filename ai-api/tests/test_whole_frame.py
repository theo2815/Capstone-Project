"""Whole-frame measurements: is anything in the frame sharp, and is detail missing sideways?"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.ml.blur import whole_frame
from src.ml.blur.detector import BlurDetector

RULE = {"soft_max_detail": 34.0, "motion_min_ratio": 2.6, "motion_max_detail": 52.0}


def _texture(height: int = 900, width: int = 600) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_sharp_texture_has_detail_and_no_direction():
    detail, ratio = whole_frame.measure(_texture())
    assert detail > 1000
    assert 0.7 < ratio < 1.4


def test_even_blur_removes_detail_without_direction():
    detail, ratio = whole_frame.measure(cv2.GaussianBlur(_texture(), (0, 0), 8))
    assert detail < 5
    assert ratio < 2


def test_sideways_smear_raises_the_ratio():
    smeared = cv2.blur(cv2.GaussianBlur(_texture(), (0, 0), 3), (121, 1))
    detail, ratio = whole_frame.measure(smeared)
    assert detail < 5
    assert ratio > 5


def test_gray_and_colour_inputs_measure_the_same():
    bgr = _texture()
    assert whole_frame.measure(bgr) == whole_frame.measure(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))


def test_small_images_are_measured_without_enlarging():
    gray = cv2.cvtColor(_texture(height=300, width=200), cv2.COLOR_BGR2GRAY)
    expected = float(cv2.Laplacian(gray, cv2.CV_64F).var()) * 640 / 300
    assert whole_frame.measure(gray)[0] == expected


def test_large_images_are_measured_on_a_640_area_averaged_copy():
    # 853 is odd on purpose: at an exact 2x shrink OpenCV's bilinear equals area-averaging.
    gray =cv2.cvtColor(_texture(height=1280, width=853), cv2.COLOR_BGR2GRAY)
    at_640 = cv2.resize(gray, (426, 640), interpolation=cv2.INTER_AREA)
    expected = BlurDetector().detect_fast(at_640)["laplacian_variance"]
    assert round(whole_frame.measure(gray)[0], 2) == expected


@pytest.mark.parametrize(("detail", "ratio", "expected"), [
    (200.0, 1.0, None),                 # a sharp region exists
    (10.0, 1.0, "defocused_blurred"),   # nothing sharp, no direction
    (10.0, 6.0, "motion_blurred"),      # nothing sharp, smeared sideways
    (40.0, 6.0, "motion_blurred"),      # a smear keeps a little more detail than even blur
    (200.0, 6.0, None),                 # strong direction but real detail: not whole-frame blur
    (40.0, 1.0, None),
])
def test_decide(detail, ratio, expected):
    assert whole_frame.decide(detail, ratio, **RULE) == expected


def test_square_view_keeps_the_whole_frame():
    upright = np.zeros((1280, 853, 3), dtype=np.uint8)
    upright[:40] = 255      # a band along the top edge
    upright[-40:] = 255     # and the bottom edge: a centre crop would lose both
    view = whole_frame.square_gray_view(upright, 640)
    assert view.shape == (640, 640)
    assert view.dtype == np.uint8
    assert view[:15].max() == 255
    assert view[-15:].max() == 255
    assert view[:, :100].max() == 0     # black bars left and right
    assert view[:, -100:].max() == 0


def test_square_view_of_the_exact_size_is_unchanged():
    exact = np.random.default_rng(1).integers(0, 256, (640, 640), dtype=np.uint8)
    assert np.array_equal(whole_frame.square_gray_view(exact, 640), exact)


def test_square_view_area_averages_when_shrinking():
    stripes = np.zeros((1500, 1500), dtype=np.uint8)
    stripes[:, ::3] = 255
    view = whole_frame.square_gray_view(stripes, 640)
    # Area-averaging leaves std ~28 (of 255); point-sampling would leave ~85.
    assert view.std() < 40
