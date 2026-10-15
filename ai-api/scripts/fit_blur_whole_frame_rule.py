"""Learn the whole-frame rule's three thresholds from the labelled folds.

Every photo goes through the real upload path (fit 1280, JPEG q90: the emulation
scripts/blur_gate.py uses) and is measured by src/ml/blur/whole_frame.py. The rule is
fitted once per fold with that fold left out, to show how it does on events it never
saw, and then once on everything for the thresholds to ship.

Usage (from ai-api/):  python scripts/fit_blur_whole_frame_rule.py
Copy the printed values into src/config.py (BLUR_SOFT_MAX_DETAIL,
BLUR_MOTION_MIN_RATIO, BLUR_MOTION_MAX_DETAIL). Measurements are cached in
<folds>/whole_frame_measures.json; delete that file after changing the folds or how a
photo is measured (whole_frame.measure, blur_gate.resize_bytes).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = ROOT / "Training-Images" / "blur_v2"
# 0 = some region is sharp (sharp or missed focus), 1 = whole-frame soft, 2 = motion
GROUP = {"sharp": 0, "defocused_object_portrait": 0, "defocused_blurred": 1, "motion_blurred": 2}
KIND = {None: 0, "defocused_blurred": 1, "motion_blurred": 2}
MAX_SHARP_SENT_TO_BLUR = 0.002   # the 2026-10-05 investigation measured 0 of 4,073
MAX_WHOLE_FRAME_SLIPPED = 0.02


def best_threshold(values: np.ndarray, is_low: np.ndarray) -> float:
    """The cut t that best satisfies `value <= t  <=>  is_low`, by balanced accuracy."""
    order = np.argsort(values, kind="stable")
    ordered, low = values[order], is_low[order].astype(np.int64)
    n_low, n_high = int(low.sum()), int(len(low) - low.sum())
    if not n_low or not n_high:
        raise ValueError("both groups are needed to place a threshold")
    low_at_or_below = np.cumsum(low)
    high_above = n_high - (np.arange(1, len(ordered) + 1) - low_at_or_below)
    score = low_at_or_below / n_low + high_above / n_high
    between_distinct = np.append(ordered[1:] != ordered[:-1], False)
    i = int(np.argmax(np.where(between_distinct, score, -1.0)))
    return float((ordered[i] + ordered[i + 1]) / 2)


def fit_rule(detail: np.ndarray, ratio: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """Thresholds for whole_frame.decide from measured photos (group: see GROUP)."""
    motion_min_ratio = best_threshold(ratio, group != 2)
    low = ratio <= motion_min_ratio
    soft_max_detail = best_threshold(detail[low], group[low] != 0)
    high = ~low
    if (group[high] == 0).any() and (group[high] != 0).any():
        motion_max_detail = best_threshold(detail[high], group[high] != 0)
    else:  # no sharp-region photo above the ratio cut: keep every smeared photo
        motion_max_detail = float(detail[high].max())
    return {"soft_max_detail": soft_max_detail, "motion_min_ratio": motion_min_ratio,
            "motion_max_detail": motion_max_detail}


def collect(folds_dir: Path) -> list[dict]:
    """One row per photo in fold_*/<event>/<class>/: fold, event, label and path."""
    rows = []
    for fold in sorted(folds_dir.glob("fold_*")):
        for event in sorted(p for p in fold.iterdir() if p.is_dir()):
            for label_dir in sorted(p for p in event.iterdir() if p.is_dir()):
                if label_dir.name not in GROUP:
                    raise ValueError(f"unknown class folder: {label_dir}")
                rows += [{"fold": fold.name, "event": event.name, "label": label_dir.name,
                          "path": str(path)} for path in sorted(label_dir.iterdir())]
    return rows


def measure_photo(path: str) -> list[float]:
    """Upload emulation -> server decode -> [fine_detail, direction_ratio]."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import cv2

    from scripts.blur_gate import resize_bytes
    from src.ml.blur import whole_frame

    upload = np.frombuffer(resize_bytes(Path(path), 1280, 90), dtype=np.uint8)
    return list(whole_frame.measure(cv2.imdecode(upload, cv2.IMREAD_COLOR)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=FOLDS_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.ml.blur import whole_frame

    rows = collect(args.folds)
    cache = args.folds / "whole_frame_measures.json"
    if cache.is_file():
        measures = json.loads(cache.read_text(encoding="utf-8"))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            values = list(pool.map(measure_photo, [row["path"] for row in rows], chunksize=32))
        measures = {Path(row["path"]).relative_to(args.folds).as_posix(): value
                    for row, value in zip(rows, values, strict=True)}
        cache.write_text(json.dumps(measures), encoding="utf-8")
    keys = [Path(row["path"]).relative_to(args.folds).as_posix() for row in rows]
    detail = np.array([measures[key][0] for key in keys])
    ratio = np.array([measures[key][1] for key in keys])
    group = np.array([GROUP[row["label"]] for row in rows])
    fold = np.array([row["fold"] for row in rows])
    sharp = np.array([row["label"] == "sharp" for row in rows])

    predicted = np.zeros(len(rows), dtype=int)
    for name in sorted(set(fold)):
        held_out = fold == name
        rule = fit_rule(detail[~held_out], ratio[~held_out], group[~held_out])
        for i in np.nonzero(held_out)[0]:
            predicted[i] = KIND[whole_frame.decide(detail[i], ratio[i], **rule)]
        print(f"{name} left out: " + ", ".join(f"{k}={v:.2f}" for k, v in rule.items()))
    sharp_sent = int((predicted[sharp] != 0).sum())
    whole_frame_blur = group != 0
    slipped = int((predicted[whole_frame_blur] == 0).sum())
    print(f"\nOn folds the rule never saw ({len(rows)} photos):")
    print(f"  sharp photos sent to a blur group:  {sharp_sent}/{int(sharp.sum())}")
    print(f"  whole-frame blur that slipped past: {slipped}/{int(whole_frame_blur.sum())}")
    missed_focus = (group == 0) & ~sharp
    print(f"  missed-focus photos caught early:   "
          f"{int((predicted[missed_focus] != 0).sum())}/{int(missed_focus.sum())}")
    final = fit_rule(detail, ratio, group)
    print("\nThresholds from all folds (put these in src/config.py):")
    print(f"  BLUR_SOFT_MAX_DETAIL   = {final['soft_max_detail']:.1f}")
    print(f"  BLUR_MOTION_MIN_RATIO  = {final['motion_min_ratio']:.2f}")
    print(f"  BLUR_MOTION_MAX_DETAIL = {final['motion_max_detail']:.1f}")
    passed = (sharp_sent <= MAX_SHARP_SENT_TO_BLUR * sharp.sum()
              and slipped <= MAX_WHOLE_FRAME_SLIPPED * whole_frame_blur.sum())
    print("\nPASS" if passed else "\nFAIL: the rule is outside its limits; do not ship it")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
