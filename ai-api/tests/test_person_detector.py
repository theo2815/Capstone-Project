"""The person detector's geometry: letterbox, raw-output decoding, box merging, mask decoding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.ml.blur import person_detector as pd


def test_letterbox_fits_an_upright_frame_with_gray_bars_at_the_sides():
    image = np.full((1280, 853, 3), 200, dtype=np.uint8)
    padded, scale, (left, top) = pd.letterbox(image)
    assert padded.shape == (640, 640, 3)
    assert scale == pytest.approx(0.5)
    assert top == 0 and 106 <= left <= 107  # (640 - 426) / 2
    assert padded[0, 0].tolist() == [114, 114, 114]  # the bar
    assert padded[320, 320].tolist() == [200, 200, 200]  # the picture


def test_letterbox_always_returns_the_square():
    for shape in ((1280, 853), (853, 1280), (640, 640), (1500, 1000), (90, 60)):
        padded, _, _ = pd.letterbox(np.zeros((*shape, 3), dtype=np.uint8))
        assert padded.shape == (640, 640, 3)


def test_tone_normalize_gives_the_same_picture_for_the_gray_copy():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (120, 90, 3), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brighter = np.clip(255 * (gray / 255.0) ** 0.6, 0, 255).astype(
        np.uint8
    )  # gamma, as the gate's copy
    a, b = pd.tone_normalize(image), pd.tone_normalize(cv2.merge([brighter] * 3))
    assert (
        a.shape == (120, 90, 3)
        and np.array_equal(a[..., 0], a[..., 1])
        and np.array_equal(a[..., 1], a[..., 2])
    )
    assert np.array_equal(a, pd.tone_normalize(cv2.merge([gray] * 3)))  # the gray copy is a no-op
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 6.0  # the gamma is largely cancelled


def test_input_tensor_is_rgb_float_in_unit_range():
    padded = np.zeros((640, 640, 3), dtype=np.uint8)
    padded[..., 0] = 255  # blue in BGR
    x = pd.to_input(padded)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert x[0, 2].max() == 1.0 and x[0, 0].max() == 0.0  # blue ends up in the last (B) plane


def _raw_output(rows):
    """rows: (cx, cy, w, h, person_score[, weight]) in letterbox pixels -> (1, 116, 8400)."""
    out = np.zeros((1, 116, 8400), dtype=np.float32)
    for i, row in enumerate(rows):
        cx, cy, w, h, score = row[:5]
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4, i] = score
        out[0, 84:, i] = row[5] if len(row) > 5 else 0.0
    return out


def test_boxes_are_kept_in_letterbox_pixels_with_their_weights():
    out = _raw_output([(300, 300, 50, 200, 0.9, 1.5)])
    boxes, scores, weights = pd.people_from_output(out, min_conf=0.2)
    assert boxes.tolist() == [[275, 200, 325, 400]]
    assert scores.tolist() == pytest.approx([0.9])
    assert weights.shape == (1, 32) and weights[0, 0] == pytest.approx(1.5)


def test_boxes_are_scaled_back_to_the_image_and_clipped():
    # upright 1280 x 853 image: scale 0.5, left pad 107
    boxes = np.array([[157.0, 200.0, 257.0, 400.0], [5.0, -10.0, 150.0, 90.0]])
    image_boxes = pd.boxes_to_image(boxes, 0.5, (107, 0), (1280, 853, 3))
    assert image_boxes[0].tolist() == pytest.approx([100, 400, 300, 800])
    assert image_boxes[1][0] == 0 and image_boxes[1][1] == 0  # clipped at the corner


def test_low_scores_and_other_classes_are_ignored():
    out = _raw_output([(300, 300, 50, 50, 0.1)])  # a person below the cut
    out[0, 4 + 2, 1] = 0.99  # a confident car
    out[0, :4, 1] = (100, 100, 20, 20)
    boxes, _, _ = pd.people_from_output(out, min_conf=0.2)
    assert len(boxes) == 0


def test_overlapping_detections_of_one_person_merge_into_the_strongest():
    out = _raw_output(
        [(300, 300, 100, 200, 0.8), (305, 302, 100, 200, 0.6), (500, 100, 40, 80, 0.5)]
    )
    _, scores, _ = pd.people_from_output(out, min_conf=0.2)
    assert sorted(round(float(s), 1) for s in scores) == [0.5, 0.8]


def test_masks_are_cut_to_the_box_and_scaled_to_the_image():
    """One prototype channel lit everywhere: the mask is exactly the box, in image pixels."""
    protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
    protos[0, 0] = 10.0  # sigmoid(10 * weight) ~ 1 everywhere
    weights = np.zeros((1, 32), dtype=np.float32)
    weights[0, 0] = 1.0
    box = np.array([[157.0, 200.0, 257.0, 400.0]])  # letterbox pixels, upright frame
    (mask,) = pd.masks_from_output(protos, weights, box, 0.5, (107, 0), (1280, 853, 3))
    assert mask.shape == (1280, 853) and mask.dtype == bool
    ys, xs = np.nonzero(mask)
    assert 395 <= ys.min() <= 405 and 795 <= ys.max() <= 805  # the box is y 400..800 in the image
    assert 95 <= xs.min() <= 105 and 295 <= xs.max() <= 305  # and x 100..300
    assert not mask[200, 400] and not mask[1000, 100]  # nothing outside the box


def test_a_negative_weight_gives_an_empty_mask():
    protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
    protos[0, 0] = 10.0
    weights = np.full((1, 32), -1.0, dtype=np.float32)
    (mask,) = pd.masks_from_output(
        protos, weights, np.array([[100.0, 100.0, 300.0, 500.0]]), 1.0, (0, 0), (640, 640, 3)
    )
    assert not mask.any()


def test_detector_runs_the_session_once_and_returns_detections():
    class Session:
        calls = 0

        def get_inputs(self):
            return [SimpleNamespace(name="images")]

        def run(self, _names, feed):
            Session.calls += 1
            assert feed["images"].shape == (1, 3, 640, 640)
            protos = np.zeros((1, 32, 160, 160), dtype=np.float32)
            protos[0, 0] = 10.0
            return [_raw_output([(320, 320, 100, 300, 0.7, 1.0)]), protos]

    detector = pd.PersonDetector(Session(), min_conf=0.2)
    people = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    assert Session.calls == 1 and len(people) == 1
    d = people[0]
    assert d.box == pytest.approx((270, 170, 370, 470)) and d.conf == pytest.approx(0.7)
    assert d.mask.shape == (640, 640) and d.mask[320, 320] and not d.mask[50, 50]


def test_matches_ultralytics_on_a_real_picture():
    """Our letterbox, NMS and mask decoding must give Ultralytics' boxes and masks."""
    pytest.importorskip("ultralytics")
    import onnxruntime as ort

    onnx = Path("models/person_detector/yolov8n_seg_person.onnx")
    if not onnx.is_file():
        pytest.skip("export the detector first (scripts/export_person_detector.py)")
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))  # ships with Ultralytics: four people
    reference = YOLO(str(onnx), task="segment").predict(
        pd.tone_normalize(image), imgsz=640, conf=0.2, classes=[0], retina_masks=True, verbose=False
    )[0]
    ours = pd.PersonDetector(ort.InferenceSession(str(onnx)), min_conf=0.2).detect(image)
    assert len(reference.boxes) >= 3  # the comparison below must have people to compare
    assert len(ours) == len(reference.boxes)
    ours = sorted(ours, key=lambda d: d.box)
    order = np.argsort([b[0] for b in reference.boxes.xyxy.tolist()])
    for d, i in zip(ours, order, strict=True):
        assert d.box == pytest.approx(reference.boxes.xyxy[i].tolist(), abs=1.5)
        theirs = reference.masks.data[i].numpy() > 0.5
        overlap = np.logical_and(d.mask, theirs).sum() / max(np.logical_or(d.mask, theirs).sum(), 1)
        assert overlap > 0.9
