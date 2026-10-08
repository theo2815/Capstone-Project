"""Lay the labelled event photos out as five event folds.

Reads Training-Images/quality_evidence/labels_2026-10.csv (one row per unique photo:
sha256, event, final_label) and hard-links every photo into

    Training-Images/blur_v2/fold_<a-e>/<event>/<class>/<sha16>.jpg

Whole events stay inside one fold, so a model can always be scored on events it
never saw. Hard links cost no disk and leave the originals untouched. The layout is
the one scripts/blur_gate.py reads (event folders that hold class folders).

Usage (from ai-api/):  python scripts/build_blur_folds.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent.parent / "Training-Images"
LABELS = "quality_evidence/labels_2026-10.csv"
SOURCES = ("Sharp event photos", "Event subject missed focus", "Event defocus", "Event motion",
           "holdout_events_2026-09", "final_exam_2026-10")
CLASSES = ("sharp", "defocused_object_portrait", "defocused_blurred", "motion_blurred")
# Hand-balanced on 2026-10-05: every fold holds sharp and missed-focus photos, and the
# two motion sources (alin-move, palawanpay) sit in different folds.
FOLDS = {
    "bohol-intl-marathon-2026-08": "a",
    "decathlon-half-marathon-2026-09": "b",
    "trilogy-run-asia-leg3-2026-09": "b",
    "alin-move-half-marathon-cebu-2026-09-27": "b",
    "reyna-bulakna-run-2026-08": "c",
    "iloilo-city-marathon-2026-08": "c",
    "st-ignatius-run-2026-08": "c",
    "nustar-run-2026-08": "c",
    "palawanpay-trilogy-cebu-2026-09-27": "c",
    "davao-marathon-2026-08": "d",
    "milo-marathon-2026-09": "d",
    "discovery-marathon-2026-09": "d",
    "rd-50th-anniversary-run-2026-06": "d",
    "yakult-run-2026-8": "d",
    "uprun-2026-05": "d",
    "pain-and-pleasure-run-2026-08": "e",
    "trilogy-run-asia-leg3-2026-08": "e",
    "ironman-70-3-lapu-lapu-cebu-2026-08": "e",
    "womens-run-ph-2026-07": "e",
    "winzir-color-run-2026-07": "e",
    "cebu-marathon-2026-07": "e",
    "iloilo-run-2026-09": "e",
    # The spent final exam of 2026-10-08 (owner decision D1 of the round-three plan).
    "dipolog-marathon-1-2026-10": "e",
    "gaisano-grand-run-2026-10": "d",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(training_dir: Path = TRAINING_DIR) -> Counter:
    """Link every labelled photo into its fold; return photo counts per (fold, class)."""
    out = training_dir / "blur_v2"
    if any(out.glob("fold_*")):
        raise FileExistsError(f"Refusing to replace existing folds under {out}")
    with (training_dir / LABELS).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    unknown = sorted({row["event"] for row in rows} - set(FOLDS))
    if unknown:
        raise ValueError(f"events without a fold: {', '.join(unknown)}")
    bad = sorted({row["final_label"] for row in rows} - set(CLASSES))
    if bad:
        raise ValueError(f"unknown labels: {', '.join(bad)}")
    by_hash: dict[str, Path] = {}
    for source in SOURCES:
        for path in sorted((training_dir / source).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
                by_hash[_sha256(path)] = path
    missing = [row["file"] for row in rows if row["sha256"] not in by_hash]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} labelled photo(s) have no local copy; first: {missing[0]}")
    counts: Counter = Counter()
    for row in rows:
        fold = FOLDS[row["event"]]
        target = (out / f"fold_{fold}" / row["event"] / row["final_label"]
                  / f"{row['sha256'][:16]}.jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(by_hash[row["sha256"]], target)
        counts[(fold, row["final_label"])] += 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-images", type=Path, default=TRAINING_DIR)
    args = parser.parse_args()
    totals = build(args.training_images)
    print(f"{'fold':6s}" + "".join(f"{name[:26]:>28s}" for name in CLASSES))
    for fold in sorted({key[0] for key in totals}):
        print(f"{fold:6s}" + "".join(f"{totals[(fold, name)]:28d}" for name in CLASSES))
    print(f"total  {sum(totals.values())} photos")
