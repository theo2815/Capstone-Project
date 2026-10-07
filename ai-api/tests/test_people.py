"""Per-person sharpness numbers, the feature vector and the logistic rule."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from src.ml.blur import people
from src.ml.blur.person_detector import Detection


def _scene(soft_left: bool = False, sharp_right: bool = True) -> np.ndarray:
    """A 600 x 400 frame with a left 'person' (x 40..160) and a right one (x 240..360)."""
    frame = np.full((600, 400, 3), 60, dtype=np.uint8)
    left = np.zeros_like(frame)
    left[100:500, 40:160] = 230
    left[100:500:8, 40:160] = 90  # a striped shirt: texture a real person has
    left[101:500:8, 40:160] = 90
    if soft_left:
        left = cv2.GaussianBlur(left, (0, 0), 10)
    frame = np.maximum(frame, left)
    right = np.zeros_like(frame)
    right[150:550, 240:360] = 200
    right[150:550:8, 240:360] = 80
    right[151:550:8, 240:360] = 80
    if not sharp_right:
        right = cv2.GaussianBlur(right, (0, 0), 10)
    return np.maximum(frame, right)


def _detection(x0, y0, x1, y1, conf, shape=(600, 400)):
    """A detection whose mask is its box (the people here are rectangles)."""
    mask = np.zeros(shape, dtype=bool)
    mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    return Detection(box=(float(x0), float(y0), float(x1), float(y1)), conf=conf, mask=mask)


LEFT = _detection(40, 100, 160, 500, 0.9)
RIGHT = _detection(240, 150, 360, 550, 0.8)


def test_measure_reads_geometry_in_frame_shares():
    (left, right) = people.measure(_scene(), [LEFT, RIGHT])
    assert left.box == LEFT.box and left.conf == 0.9
    assert left.area == pytest.approx(120 * 400 / (400 * 600))
    assert left.mask_area == pytest.approx(left.area)
    assert left.height == pytest.approx(400 / 600) and left.bottom == pytest.approx(500 / 600)
    assert left.cx == pytest.approx(100 / 400)
    assert left.touches_edge is False
    assert right.conf == 0.8


def test_a_box_at_the_frame_border_is_cut_off():
    (cut,) = people.measure(_scene(), [_detection(0, 100, 160, 500, 0.9)])
    assert cut.touches_edge is True


def test_a_crisp_person_is_sharper_than_a_soft_one():
    (crisp,) = people.measure(_scene(soft_left=False), [LEFT])
    (soft,) = people.measure(_scene(soft_left=True), [LEFT])
    assert crisp.es > 2 * soft.es
    assert crisp.es95 > 2 * soft.es95
    assert soft.contrast > 50  # the soft person is still there to measure


def test_sharpness_comes_from_the_person_s_own_pixels_not_its_box():
    """A soft person whose box reaches over a crisp one must not borrow the crisp edges."""
    frame = _scene(soft_left=True)
    wide_box_own_pixels = _detection(40, 100, 160, 500, 0.9)  # mask: the soft person only
    wide_box_own_pixels = Detection(
        box=(40.0, 100.0, 300.0, 500.0), conf=0.9, mask=wide_box_own_pixels.mask
    )
    soft, crisp = people.measure(frame, [wide_box_own_pixels, RIGHT])
    assert soft.area > soft.mask_area * 1.5  # the box is much bigger than the person
    assert crisp.es95 > 2 * soft.es95  # measured on own pixels: still soft


def test_people_without_pixels_or_too_small_are_skipped_and_gray_input_is_accepted():
    gray = cv2.cvtColor(_scene(), cv2.COLOR_BGR2GRAY)
    empty = Detection(
        box=(200.0, 200.0, 300.0, 300.0), conf=0.9, mask=np.zeros((600, 400), dtype=bool)
    )
    tiny = _detection(10, 10, 14, 300, 0.9)
    found = people.measure(gray, [LEFT, empty, tiny])
    assert len(found) == 1


def test_dict_round_trip_keeps_every_number():
    (person,) = people.measure(_scene(), [LEFT])
    again = people.person_from_dict(json.loads(json.dumps(people.person_to_dict(person))))
    assert again.box == person.box and again.mask_area == pytest.approx(person.mask_area, abs=1e-5)
    assert again.es == pytest.approx(person.es, abs=1e-5)
    assert again.es95 == pytest.approx(person.es95, abs=1e-5)


def test_features_have_the_declared_length_and_are_finite_without_people():
    empty = people.features([], people.BIG_AREA, people.SOFT_RATIO)
    assert empty.shape == (len(people.FEATURE_NAMES),) == (9,)
    assert not empty.any()
    found = people.measure(_scene(soft_left=True), [LEFT, RIGHT])
    full = people.features(found, people.BIG_AREA, people.SOFT_RATIO)
    assert full.shape == empty.shape and np.isfinite(full).all()


def test_features_see_a_soft_big_person_beside_a_crisp_one():
    names = list(people.FEATURE_NAMES)
    soft_scene = people.features(people.measure(_scene(soft_left=True), [LEFT, RIGHT]), 0.05, 0.6)
    crisp_scene = people.features(people.measure(_scene(soft_left=False), [LEFT, RIGHT]), 0.05, 0.6)
    assert soft_scene[names.index("n_big")] == 2 == crisp_scene[names.index("n_big")]
    assert (
        soft_scene[names.index("max_area_soft_big")] > 0
        and crisp_scene[names.index("max_area_soft_big")] == 0
    )
    assert soft_scene[names.index("min_es_big")] < crisp_scene[names.index("min_es_big")]
    assert soft_scene[names.index("es_ratio")] < crisp_scene[names.index("es_ratio")]
    # the soft person's feet (y 500) are above the crisp one's (y 550): it is not nearer the camera
    assert soft_scene[names.index("soft_bottom_gap")] == pytest.approx((500 - 550) / 600)
    assert soft_scene[names.index("soft_area_ratio")] == pytest.approx(1.0)


def _person(area, es95, conf, bottom, edge=False, es=0.5, mask_area=0.01):
    return people.Person(
        box=(0.0, 0.0, 1.0, 1.0),
        conf=conf,
        area=area,
        mask_area=mask_area,
        height=0.5,
        bottom=bottom,
        cx=0.5,
        touches_edge=edge,
        contrast=100.0,
        es=es,
        es95=es95,
    )


def test_features_pin_each_number_on_people_of_different_sizes():
    """A crisp runner, a big soft person cut by the frame, a smaller soft one, a tiny soft one."""
    runner = _person(0.06, 0.40, 0.9, bottom=0.70)
    cut_off = _person(0.30, 0.10, 0.5, bottom=0.95, edge=True, es=0.95, mask_area=0.20)
    soft = _person(0.10, 0.20, 0.6, bottom=0.80)
    tiny = _person(0.02, 0.05, 0.7, bottom=0.60)
    got = dict(
        zip(
            people.FEATURE_NAMES,
            people.features([runner, cut_off, soft, tiny], 0.05, 0.6),
            strict=True,
        )
    )
    assert got == pytest.approx(
        {
            "n_big": 3,
            "largest_es": 0.10,
            "largest_conf": 0.5,
            "min_es_big": 0.10,
            "es_ratio": 0.25,
            "max_area_soft_big": 0.30,
            "area_edge_soft": 0.30,
            "soft_bottom_gap": 0.25,
            "soft_area_ratio": 5.0,
        }
    )
    alone = dict(zip(people.FEATURE_NAMES, people.features([tiny], 0.05, 0.6), strict=True))
    assert alone["n_big"] == 0 and alone["min_es_big"] == 0.05  # nobody big: the sharpest stands in


def test_logistic_rule_applies_standardisation_and_the_sigmoid(tmp_path):
    spec = {
        "feature_names": ["a", "b"],
        "mean": [1.0, 3.0],
        "scale": [2.0, 4.0],
        "coef": [1.0, -1.0],
        "intercept": 0.5,
        "min_conf": 0.2,
        "big_area": 0.05,
        "soft_ratio": 0.6,
    }
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(spec))
    rule = people.LogisticRule.load(path)
    z = (3.0 - 1.0) / 2.0 * 1.0 + (2.0 - 3.0) / 4.0 * -1.0 + 0.5
    assert rule.probability(np.array([3.0, 2.0])) == pytest.approx(1 / (1 + math.exp(-z)))
    assert rule.min_conf == 0.2 and rule.big_area == 0.05 and rule.soft_ratio == 0.6
    assert rule.feature_names == ["a", "b"]


def test_logistic_rule_with_zero_weights_answers_the_intercept_only():
    k = len(people.FEATURE_NAMES)
    rule = people.LogisticRule(
        {
            "feature_names": list(people.FEATURE_NAMES),
            "mean": [0.0] * k,
            "scale": [1.0] * k,
            "coef": [0.0] * k,
            "intercept": math.log(0.3 / 0.7),
            "min_conf": 0.2,
            "big_area": 0.05,
            "soft_ratio": 0.6,
        }
    )
    assert rule.probability(np.arange(k, dtype=float)) == pytest.approx(0.3)
