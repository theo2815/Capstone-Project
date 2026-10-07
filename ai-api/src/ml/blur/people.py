"""Stage 2 of the blur classifier: how sharp is each person in the frame?

"Subject missed focus" in the owner's labelled photos means a blurred person: either a
blurred person in front cuts into the frame while the runner behind is sharp (about
three quarters of the cases), or the runner is soft while something else is sharp. A
whole-frame model learned the look of the shoot instead (vault journal
2026-10-06-blur-stage2-failure-advice). So stage 2 measures the people: the detector
(person_detector.py) finds each person and its pixels, this module measures each person's
edge sharpness on those pixels at the decoded upload's full detail, turns the people into a
few numbers, and a logistic rule fitted by scripts/fit_blur_people_rule.py turns those into
P(missed focus).

Sharpness is measured on the person's own pixels, never on its box: a box also holds the
background, and in the "runner soft, background sharp" photos the background's edges would
make the runner read sharp.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.ml.blur.person_detector import Detection

EDGE_MARGIN = 2  # a box this close to the frame border counts as cut off
MIN_PIXELS = 64  # fewer own pixels than this: nothing to measure, the person is skipped


@dataclass(frozen=True)
class Person:
    """One detected person and its sharpness numbers (shares are fractions of the frame)."""

    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the measured image
    conf: float  # detector confidence
    area: float  # box area / frame area
    mask_area: float  # the person's own pixels / frame area
    height: float  # box height / frame height
    bottom: float  # y1 / frame height (1.0 = feet at the bottom edge)
    cx: float  # box centre x / frame width
    touches_edge: bool  # cut off by the frame border
    contrast: float  # 98th - 2nd percentile of gray on the own pixels
    es: float  # edge sharpness: 99.5th-percentile slope / contrast
    es95: float  # the same with the 95th percentile: what the rule reads


def _edge_sharpness(mag: np.ndarray, gray: np.ndarray) -> tuple[float, float, float]:
    """(es, es95, contrast) of one set of pixels."""
    contrast = float(np.percentile(gray, 98) - np.percentile(gray, 2))
    scale = max(contrast, 8.0)
    return (
        float(np.percentile(mag, 99.5) / scale),
        float(np.percentile(mag, 95) / scale),
        contrast,
    )


def measure(image: np.ndarray, detections: list[Detection]) -> list[Person]:
    """Measure every detected person on a BGR (or gray) image; the gradient is computed once."""
    gray = (image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).astype(
        np.float32
    )
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8)
    mag = np.sqrt(gx * gx + gy * gy)
    people = []
    for d in detections:
        x0, y0, x1, y1 = d.box
        ix0, iy0 = int(max(0, x0)), int(max(0, y0))
        ix1, iy1 = int(min(w, x1)), int(min(h, y1))
        if ix1 - ix0 < 8 or iy1 - iy0 < 8:
            continue
        own = d.mask if d.mask.shape == (h, w) else np.zeros((h, w), dtype=bool)
        if own.sum() < MIN_PIXELS:
            continue
        es, es95, contrast = _edge_sharpness(mag[own], gray[own])
        people.append(
            Person(
                box=(float(x0), float(y0), float(x1), float(y1)),
                conf=float(d.conf),
                area=(ix1 - ix0) * (iy1 - iy0) / (w * h),
                mask_area=float(own.mean()),
                height=(iy1 - iy0) / h,
                bottom=iy1 / h,
                cx=(ix0 + ix1) / 2 / w,
                touches_edge=bool(
                    x0 <= EDGE_MARGIN
                    or y0 <= EDGE_MARGIN
                    or x1 >= w - EDGE_MARGIN
                    or y1 >= h - EDGE_MARGIN
                ),
                contrast=contrast,
                es=es,
                es95=es95,
            )
        )
    return people


def person_to_dict(person: Person) -> dict:
    """For the measurement cache written by scripts/measure_blur_people.py."""
    return {
        "box": [round(v, 1) for v in person.box],
        "conf": round(person.conf, 4),
        "area": round(person.area, 5),
        "mask_area": round(person.mask_area, 5),
        "height": round(person.height, 4),
        "bottom": round(person.bottom, 4),
        "cx": round(person.cx, 4),
        "touches_edge": person.touches_edge,
        "contrast": round(person.contrast, 2),
        "es": round(person.es, 5),
        "es95": round(person.es95, 5),
    }


def person_from_dict(d: dict) -> Person:
    return Person(
        box=tuple(d["box"]),
        conf=d["conf"],
        area=d["area"],
        mask_area=d["mask_area"],
        height=d["height"],
        bottom=d["bottom"],
        cx=d["cx"],
        touches_edge=d["touches_edge"],
        contrast=d["contrast"],
        es=d["es"],
        es95=d["es95"],
    )


# --- the numbers the rule reads (chosen on the five folds in the planning session, 2026-10-06) ---

MIN_CONF = 0.2  # detector confidence below which a person is ignored
BIG_AREA = 0.05  # a person whose box covers at least this share of the frame is "big"
SOFT_RATIO = 0.6  # a big person this much softer than the sharpest person is "soft"
FEATURE_NAMES = (
    "n_big",  # how many big people there are
    "largest_es",  # sharpness of the largest person
    "largest_conf",  # the detector's confidence in the largest person (a blurred body scores lower)
    "min_es_big",  # sharpness of the softest big person
    "es_ratio",  # largest person's sharpness over the sharpest person's
    "max_area_soft_big",  # size of the biggest soft big person
    "area_edge_soft",  # total size of soft people cut off by the frame border
    "soft_bottom_gap",  # the biggest soft person's feet below the sharpest person's (nearer)
    "soft_area_ratio",  # the biggest soft person's size over the sharpest person's
)


def features(people: list[Person], big_area: float, soft_ratio: float) -> np.ndarray:
    """The feature vector of one photo, in FEATURE_NAMES order (all zeros when nobody was found)."""
    if not people:
        return np.zeros(len(FEATURE_NAMES))
    largest = max(people, key=lambda p: p.area)
    sharpest = max(people, key=lambda p: p.es95)
    big = [p for p in people if p.area >= big_area]
    soft_cut = soft_ratio * sharpest.es95
    soft_big = [p for p in big if p.es95 < soft_cut]
    top_soft = max(soft_big, key=lambda p: p.area) if soft_big else None
    return np.array(
        [
            len(big),
            largest.es95,
            largest.conf,
            min(p.es95 for p in big) if big else sharpest.es95,
            largest.es95 / max(sharpest.es95, 1e-6),
            max((p.area for p in soft_big), default=0.0),
            sum(p.area for p in people if p.touches_edge and p.es95 < soft_cut),
            (top_soft.bottom - sharpest.bottom) if top_soft else 0.0,
            (top_soft.area / max(sharpest.area, 1e-6)) if top_soft else 0.0,
        ],
        dtype=float,
    )


class LogisticRule:
    """P(missed focus) = sigmoid(coef . standardised features + intercept), loaded from JSON."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.feature_names: list[str] = list(spec["feature_names"])
        self.mean = np.asarray(spec["mean"], dtype=np.float64)
        self.scale = np.asarray(spec["scale"], dtype=np.float64)
        self.coef = np.asarray(spec["coef"], dtype=np.float64)
        self.intercept = float(spec["intercept"])
        self.min_conf = float(spec["min_conf"])
        self.big_area = float(spec["big_area"])
        self.soft_ratio = float(spec["soft_ratio"])

    @classmethod
    def load(cls, path: str | Path) -> LogisticRule:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def probability(self, x: np.ndarray) -> float:
        z = float(
            np.dot((np.asarray(x, dtype=np.float64) - self.mean) / self.scale, self.coef)
            + self.intercept
        )
        return float(1.0 / (1.0 + np.exp(-z)))
