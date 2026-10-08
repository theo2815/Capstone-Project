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
    right[150:560, 240:360] = 200
    right[150:560:8, 240:360] = 80
    right[151:560:8, 240:360] = 80
    if not sharp_right:
        right = cv2.GaussianBlur(right, (0, 0), 10)
    return np.maximum(frame, right)


def _detection(x0, y0, x1, y1, conf, shape=(600, 400)):
    """A detection whose mask is its box (the people here are rectangles)."""
    mask = np.zeros(shape, dtype=bool)
    mask[int(y0) : int(y1), int(x0) : int(x1)] = True
    return Detection(box=(float(x0), float(y0), float(x1), float(y1)), conf=conf, mask=mask)


LEFT = _detection(40, 100, 160, 500, 0.9)
RIGHT = _detection(240, 150, 360, 560, 0.8)


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
    assert empty.shape == (len(people.FEATURE_NAMES),) == (13,)
    assert not empty.any()
    found = people.measure(_scene(soft_left=True), [LEFT, RIGHT])
    full = people.features(found, people.BIG_AREA, people.SOFT_RATIO)
    assert full.shape == empty.shape and np.isfinite(full).all()


def test_features_see_a_soft_big_person_beside_the_crisp_subject():
    names = list(people.FEATURE_NAMES)
    soft_scene = people.features(people.measure(_scene(soft_left=True), [LEFT, RIGHT]), 0.05, 0.7)
    crisp_scene = people.features(people.measure(_scene(soft_left=False), [LEFT, RIGHT]), 0.05, 0.7)
    assert soft_scene[names.index("n_big")] == 2 == crisp_scene[names.index("n_big")]
    # the subject is the right person (slightly bigger, confidence 0.8), crisp in both scenes
    assert soft_scene[names.index("es_ratio")] == pytest.approx(1.0)
    assert crisp_scene[names.index("es_ratio")] > 0.8
    assert soft_scene[names.index("max_area_soft_other")] == pytest.approx(120 * 400 / (400 * 600))
    assert crisp_scene[names.index("max_area_soft_other")] == 0
    ratio = names.index("biggest_other_es_ratio")
    assert soft_scene[ratio] < 0.7 < crisp_scene[ratio]
    # the soft person's feet (y 500) are above the subject's (y 560): it is not nearer the camera
    assert soft_scene[names.index("soft_bottom_gap")] == pytest.approx((500 - 560) / 600)
    assert soft_scene[names.index("soft_area_at_feet")] == 0
    assert soft_scene[names.index("soft_area_ratio_log")] == pytest.approx(
        math.log1p((120 * 400) / (120 * 410))
    )


def _person(box, conf, area, bottom, es95, edge=False):
    return people.Person(
        box=box,
        conf=conf,
        area=area,
        mask_area=area * 0.7,
        height=0.5,
        bottom=bottom,
        cx=0.5,
        touches_edge=edge,
        contrast=100.0,
        es=es95 * 1.2,
        es95=es95,
    )


def test_features_pin_each_number_on_a_subject_an_intruder_a_bystander_a_duplicate_and_a_tiny_one():
    runner = _person((400.0, 100.0, 600.0, 700.0), 0.9, 0.12, bottom=0.70, es95=0.40)
    cut_off = _person((0.0, 300.0, 350.0, 1000.0), 0.4, 0.30, bottom=1.00, es95=0.10, edge=True)
    behind = _person((650.0, 250.0, 800.0, 650.0), 0.8, 0.06, bottom=0.65, es95=0.20)
    duplicate = _person((410.0, 120.0, 590.0, 500.0), 0.3, 0.07, bottom=0.50, es95=0.05)
    tiny = _person((900.0, 400.0, 930.0, 460.0), 0.7, 0.002, bottom=0.46, es95=0.60)
    got = dict(
        zip(
            people.FEATURE_NAMES,
            people.features([runner, cut_off, behind, duplicate, tiny], 0.05, 0.6),
            strict=True,
        )
    )
    assert got == pytest.approx(
        {
            "n_big": 4,
            "subject_es": 0.40,  # the runner: largest among the confident people
            "subject_conf": 0.9,
            "min_es_big_confident": 0.20,  # the runner behind
            "es_ratio": 1.0,
            "max_area_soft_other": 0.30,  # the cut-off body
            "area_edge_soft_other": 0.30,
            "soft_bottom_gap": 0.30,  # its feet 0.30 of the frame below the runner's
            "soft_area_ratio_log": math.log1p(0.30 / 0.12),
            "soft_area_at_feet": 0.30,  # the body behind stands higher and does not count
            "lowconf_big_area": 0.30,  # the duplicate is not an "other"
            "subject_at_bottom_edge": 0.0,
            "biggest_other_es_ratio": 0.25,
        }
    )
    alone = dict(zip(people.FEATURE_NAMES, people.features([cut_off], 0.05, 0.6), strict=True))
    assert alone["subject_conf"] == 0.4  # nobody confident: the largest person stands in
    assert alone["subject_at_bottom_edge"] == 1.0
    assert alone["min_es_big_confident"] == 0.10 and alone["biggest_other_es_ratio"] == 1.0
    pair = dict(
        zip(
            people.FEATURE_NAMES,
            people.features([runner, duplicate], 0.05, 0.6),
            strict=True,
        )
    )
    assert pair["n_big"] == 2 and pair["max_area_soft_other"] == 0 and pair["lowconf_big_area"] == 0


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
