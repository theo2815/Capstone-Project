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


# --- the numbers the rule reads (chosen on the five folds and the spent exam, 2026-10-08) ---

MIN_CONF = 0.2  # detector confidence below which a person is ignored
BIG_AREA = 0.05  # a person whose box covers at least this share of the frame is "big"
SOFT_RATIO = 0.7  # a big person this much softer than the reference is "soft"
SUBJECT_CONF = 0.5  # the subject is the largest person the detector is at least this sure about
DUPLICATE_OVERLAP = 0.7  # a box this much inside (or around) the subject's box is a duplicate
FEET_GAP = 0.02  # feet within this share of the frame height of the subject's are "at the feet"
BOTTOM_EDGE = 0.98  # a subject whose box reaches this far down is cut off by the bottom edge
FEATURE_NAMES = (
    "n_big",  # how many big people there are
    "subject_es",  # sharpness of the subject (the largest confident person)
    "subject_conf",  # the detector's confidence in the subject
    "min_es_big_confident",  # sharpness of the softest big confident person
    "es_ratio",  # the subject's sharpness over the reference (the sharpest big confident person)
    "max_area_soft_other",  # size of the biggest soft big "other" (not the subject or a duplicate)
    "area_edge_soft_other",  # total size of soft big others cut off by the frame border
    "soft_bottom_gap",  # the biggest soft other's feet below the subject's (nearer the camera)
    "soft_area_ratio_log",  # log(1 + the biggest soft other's size over the subject's)
    "soft_area_at_feet",  # total size of soft big others whose feet are at the subject's or lower
    "lowconf_big_area",  # total size of big others the detector is less than SUBJECT_CONF sure of
    "subject_at_bottom_edge",  # 1 when the subject's box reaches the bottom edge (feet cut off)
    "biggest_other_es_ratio",  # the biggest big other's sharpness over the reference (1 if none)
)


def _overlap_of_smaller(a: tuple, b: tuple) -> float:
    """How much of the smaller of two boxes lies inside the other (0 = apart, 1 = contained)."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def features(people: list[Person], big_area: float, soft_ratio: float) -> np.ndarray:
    """The feature vector of one photo, in FEATURE_NAMES order (all zeros when nobody was found).

    The subject is the largest person the detector is confident about, so a loose second box
    around a costume, an inflatable arch or a ghost in a dark background cannot take its place;
    boxes that mostly overlap the subject's box are its duplicates and are ignored; the rest
    are "others". Softness is measured against the sharpest big confident person.
    """
    if not people:
        return np.zeros(len(FEATURE_NAMES))
    confident = [p for p in people if p.conf >= SUBJECT_CONF]
    subject = max(confident or people, key=lambda p: p.area)
    big_confident = [p for p in confident if p.area >= big_area] or [subject]
    reference = max(p.es95 for p in big_confident)
    others = [
        p
        for p in people
        if p is not subject and _overlap_of_smaller(p.box, subject.box) < DUPLICATE_OVERLAP
    ]
    big_others = [p for p in others if p.area >= big_area]
    soft_cut = soft_ratio * reference
    soft_others = [p for p in big_others if p.es95 < soft_cut]
    top_soft = max(soft_others, key=lambda p: p.area) if soft_others else None
    biggest_other = max(big_others, key=lambda p: p.area) if big_others else None
    return np.array(
        [
            sum(p.area >= big_area for p in people),
            subject.es95,
            subject.conf,
            min(p.es95 for p in big_confident),
            subject.es95 / max(reference, 1e-6),
            max((p.area for p in soft_others), default=0.0),
            sum(p.area for p in soft_others if p.touches_edge),
            (top_soft.bottom - subject.bottom) if top_soft else 0.0,
            np.log1p(top_soft.area / max(subject.area, 1e-6)) if top_soft else 0.0,
            sum(p.area for p in soft_others if p.bottom >= subject.bottom - FEET_GAP),
            sum(p.area for p in big_others if p.conf < SUBJECT_CONF),
            1.0 if subject.bottom >= BOTTOM_EDGE else 0.0,
            (biggest_other.es95 / max(reference, 1e-6)) if biggest_other else 1.0,
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
