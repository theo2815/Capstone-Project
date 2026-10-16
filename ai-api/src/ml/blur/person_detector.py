"""Find the people in a photo, each with its own pixels: YOLOv8n-seg (COCO) through ONNX Runtime.

Stage 2 of the blur classifier decides "subject missed focus" from how sharp each person is,
measured on the person's own pixels (its mask), not on a box that also holds background. The
detector reads only the `person` class. Letterboxing, score filtering, box merging and mask
decoding are done here with OpenCV and NumPy: serving needs neither torch nor Ultralytics.
The ONNX file is written by scripts/export_person_detector.py and recorded in
models/manifest.json.

The detector is fed a tone-normalised picture: the photo's gray values after a gamma curve
that brings the mean to 128,
copied to three channels, so which people it finds does not move with the light or the colour
(owner limit: verdict flips from tone alone <= 2%).

Raw outputs of the exported model, for a 640 x 640 input:
    output0  (1, 116, 8400)   per candidate box: cx, cy, w, h, 80 class scores, 32 mask weights
    output1  (1, 32, 160, 160) 32 mask prototypes at a quarter of the input size
A person's mask is sigmoid(weights . prototypes), cut to its box, scaled back to the photo.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

INPUT_SIZE = 640  # the detector was exported with a fixed 640 x 640 input
PERSON_CLASS = 0  # COCO class index of "person"
NUM_CLASSES = 80
MASK_WEIGHTS = 32
PAD_VALUE = 114  # Ultralytics' letterbox gray
NMS_IOU = 0.7  # Ultralytics' default overlap at which two boxes are one person
MASK_THRESHOLD = 0.5
# Most people kept per photo, largest boxes first: each kept person costs a full-photo
# mask, so a crowd or a crafted photo must not decide the memory. The labelled photos have
# at most 13 people (2026-10-16), and stage 2 reads only big people and the subject.
MAX_PEOPLE = 64


@dataclass(frozen=True)
class Detection:
    """One person: its box (x0, y0, x1, y1 in image pixels), score, and its pixels."""

    box: tuple[float, float, float, float]
    conf: float
    mask: np.ndarray  # bool, the image's height x width


def tone_normalize(image: np.ndarray) -> np.ndarray:
    """BGR or gray -> gray brought to a mean of 128 by a gamma curve, in three channels.

    A gamma curve is monotonic, so a brighter or darker copy of the photo lands on almost the
    same picture; the gray copy lands on exactly the same one.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = min(254.0, max(1.0, float(gray.mean())))
    gamma = np.log(128 / 255) / np.log(mean / 255)
    lut = np.clip(255 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)
    normalised = cv2.LUT(gray, lut)
    return cv2.merge([normalised, normalised, normalised])


def letterbox(
    image: np.ndarray, size: int = INPUT_SIZE
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Fit the image inside a size x size gray square; return it, the scale, the (left, top) pad."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    return padded, scale, (left, top)


def to_input(padded_bgr: np.ndarray) -> np.ndarray:
    """Letterboxed BGR -> (1, 3, size, size) RGB float tensor in [0, 1]."""
    rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def people_from_output(
    output: np.ndarray, min_conf: float, iou: float = NMS_IOU
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw output0 -> (boxes in letterbox pixels as x0 y0 x1 y1, scores, mask weights), merged."""
    pred = output[0]
    scores = pred[4 + PERSON_CLASS]
    keep = scores >= min_conf
    empty = (
        np.zeros((0, 4), np.float32),
        np.zeros(0, np.float32),
        np.zeros((0, MASK_WEIGHTS), np.float32),
    )
    if not keep.any():
        return empty
    cx, cy, bw, bh = (pred[i, keep] for i in range(4))
    scores = scores[keep]
    weights = pred[4 + NUM_CLASSES : 4 + NUM_CLASSES + MASK_WEIGHTS, keep].T
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    kept = cv2.dnn.NMSBoxes(
        np.stack([boxes[:, 0], boxes[:, 1], bw, bh], axis=1).tolist(),
        scores.tolist(),
        min_conf,
        iou,
    )
    if len(kept) == 0:
        return empty
    idx = np.asarray(kept).flatten()
    return boxes[idx], scores[idx], weights[idx]


def boxes_to_image(
    boxes: np.ndarray, scale: float, pad: tuple[int, int], image_shape: tuple[int, ...]
) -> np.ndarray:
    """Letterbox pixels -> image pixels, clipped to the image."""
    h, w = image_shape[:2]
    out = boxes.astype(np.float64).copy()
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - pad[0]) / scale, 0, w)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - pad[1]) / scale, 0, h)
    return out


def masks_from_output(
    protos: np.ndarray,
    weights: np.ndarray,
    boxes: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    image_shape: tuple[int, ...],
) -> list[np.ndarray]:
    """Mask prototypes + each person's weights -> one bool mask per person at the image's size.

    boxes are in letterbox pixels. Each mask is cut to its box (so a person's mask never
    leaks into a neighbour), the gray padding is removed, and the result is scaled to the photo.
    """
    h, w = image_shape[:2]
    content_w, content_h = round(w * scale), round(h * scale)
    p = protos[0]  # (32, 160, 160)
    channels, mh, mw = p.shape
    flat = 1.0 / (1.0 + np.exp(-(weights @ p.reshape(channels, -1))))  # (n, 160*160)
    ratio = mw / INPUT_SIZE
    masks = []
    for soft, (x0, y0, x1, y1) in zip(flat.reshape(-1, mh, mw), boxes, strict=True):
        cut = np.zeros_like(soft)
        mx0, my0 = max(0, int(x0 * ratio)), max(0, int(y0 * ratio))
        mx1, my1 = min(mw, int(np.ceil(x1 * ratio))), min(mh, int(np.ceil(y1 * ratio)))
        cut[my0:my1, mx0:mx1] = soft[my0:my1, mx0:mx1]
        full = cv2.resize(cut, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        content = full[pad[1] : pad[1] + content_h, pad[0] : pad[0] + content_w]
        mask = cv2.resize(content, (w, h), interpolation=cv2.INTER_LINEAR) > MASK_THRESHOLD
        ix0, iy0 = int(max(0, (x0 - pad[0]) / scale)), int(max(0, (y0 - pad[1]) / scale))
        ix1, iy1 = (
            int(np.ceil(min(w, (x1 - pad[0]) / scale))),
            int(np.ceil(min(h, (y1 - pad[1]) / scale))),
        )
        boxed = np.zeros((h, w), dtype=bool)
        boxed[iy0:iy1, ix0:ix1] = mask[iy0:iy1, ix0:ix1]
        masks.append(boxed)
    return masks


class PersonDetector:
    """One ONNX session; `detect` returns the people in a BGR image, each with its pixels."""

    def __init__(self, session, min_conf: float) -> None:
        self.session = session
        self.min_conf = min_conf
        self._input_name = session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> list[Detection]:
        padded, scale, pad = letterbox(tone_normalize(image))
        output, protos = self.session.run(None, {self._input_name: to_input(padded)})[:2]
        boxes, scores, weights = people_from_output(output, self.min_conf)
        if len(boxes) == 0:
            return []
        image_boxes = boxes_to_image(boxes, scale, pad, image.shape)
        if len(boxes) > MAX_PEOPLE:
            area = (image_boxes[:, 2] - image_boxes[:, 0]) * (image_boxes[:, 3] - image_boxes[:, 1])
            keep = np.sort(np.argsort(-area, kind="stable")[:MAX_PEOPLE])  # keep detector order
            boxes, scores, weights, image_boxes = (
                boxes[keep], scores[keep], weights[keep], image_boxes[keep]
            )
        masks = masks_from_output(protos, weights, boxes, scale, pad, image.shape)
        return [
            Detection(box=tuple(float(v) for v in b), conf=float(s), mask=m)
            for b, s, m in zip(image_boxes, scores, masks, strict=True)
        ]
