"""Extract bib-only labels from the combined face+bib dataset.

Copies images and filters label files from face_bib_detection/ to bib_detection/,
keeping only bib annotations (class 1 -> class 0) and skipping images with no bibs.

This lets you reuse the 1,863 existing bib annotations from the combined dataset
as a starting point for bib-only training.  You can then add more bib-specific
images to Training-Images/bib_detection/ to improve detection quality.

Usage:
    python scripts/extract_bib_labels.py              # extract + split 80/20
    python scripts/extract_bib_labels.py --preview     # show counts without copying
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Source: combined face+bib dataset
SRC_DIR = PROJECT_ROOT / "Training-Images" / "face_bib_detection"
SRC_TRAIN_IMAGES = SRC_DIR / "images" / "train"
SRC_TRAIN_LABELS = SRC_DIR / "labels" / "train"
SRC_VAL_IMAGES = SRC_DIR / "images" / "val"
SRC_VAL_LABELS = SRC_DIR / "labels" / "val"

# Destination: bib-only dataset
DST_DIR = PROJECT_ROOT / "Training-Images" / "bib_detection"
DST_TRAIN_IMAGES = DST_DIR / "images" / "train"
DST_TRAIN_LABELS = DST_DIR / "labels" / "train"
DST_VAL_IMAGES = DST_DIR / "images" / "val"
DST_VAL_LABELS = DST_DIR / "labels" / "val"

COMBINED_BIB_CLASS_ID = 1  # bib class ID in face+bib dataset
BIB_ONLY_CLASS_ID = 0      # bib class ID in bib-only dataset

RANDOM_SEED = 42
VAL_RATIO = 0.2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _extract_bib_lines(label_path: Path) -> list[str]:
    """Read a combined label file and return only bib lines (remapped to class 0)."""
    bib_lines = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == str(COMBINED_BIB_CLASS_ID):
            # Remap class 1 (bib in combined) -> class 0 (bib in bib-only)
            bib_lines.append(f"{BIB_ONLY_CLASS_ID} {' '.join(parts[1:])}")
    return bib_lines


def extract(preview_only: bool = False) -> None:
    """Extract bib-only labels and images from the combined dataset."""
    # Check source exists
    for src in [SRC_TRAIN_IMAGES, SRC_TRAIN_LABELS]:
        if not src.exists():
            print(f"ERROR: Source not found: {src}")
            print("Run auto_annotate_face_bib.py first or place the combined dataset.")
            return

    # Gather all source label files (from both train and val of combined dataset)
    all_pairs: list[tuple[Path, Path]] = []

    for img_dir, lbl_dir in [
        (SRC_TRAIN_IMAGES, SRC_TRAIN_LABELS),
        (SRC_VAL_IMAGES, SRC_VAL_LABELS),
    ]:
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for label_path in sorted(lbl_dir.glob("*.txt")):
            bib_lines = _extract_bib_lines(label_path)
            if not bib_lines:
                continue  # skip images with no bib annotations
            # Find matching image
            img_path = None
            for ext in IMAGE_EXTENSIONS:
                candidate = img_dir / f"{label_path.stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
                # Try uppercase
                candidate = img_dir / f"{label_path.stem}{ext.upper()}"
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path:
                all_pairs.append((img_path, label_path))

    total_bibs = 0
    for _, lbl in all_pairs:
        total_bibs += len(_extract_bib_lines(lbl))

    print("=" * 60)
    print("Extract Bib-Only Labels from Combined Dataset")
    print("=" * 60)
    print(f"  Source:          {SRC_DIR}")
    print(f"  Destination:     {DST_DIR}")
    print(f"  Images with bibs: {len(all_pairs)}")
    print(f"  Total bib annotations: {total_bibs}")
    print(f"  Class remap:     {COMBINED_BIB_CLASS_ID} (combined) -> {BIB_ONLY_CLASS_ID} (bib-only)")

    if preview_only:
        print("\n[PREVIEW MODE] No files copied.")
        return

    if not all_pairs:
        print("\nERROR: No bib annotations found in source dataset.")
        return

    # Check if destination already has data
    existing_train = list(DST_TRAIN_IMAGES.glob("*.*")) if DST_TRAIN_IMAGES.exists() else []
    if existing_train:
        print(f"\n  NOTE: Destination already has {len(existing_train)} training images.")
        print("  Extracted images will be ADDED (not overwritten).")

    # Create destination directories
    for d in [DST_TRAIN_IMAGES, DST_TRAIN_LABELS, DST_VAL_IMAGES, DST_VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)

    # Split into train/val
    rng = random.Random(RANDOM_SEED)
    pairs = list(all_pairs)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # Copy files
    for split_name, split_pairs, img_dst, lbl_dst in [
        ("train", train_pairs, DST_TRAIN_IMAGES, DST_TRAIN_LABELS),
        ("val", val_pairs, DST_VAL_IMAGES, DST_VAL_LABELS),
    ]:
        for img_path, label_path in split_pairs:
            # Copy image
            shutil.copy2(str(img_path), str(img_dst / img_path.name))
            # Write bib-only label
            bib_lines = _extract_bib_lines(label_path)
            (lbl_dst / f"{label_path.stem}.txt").write_text("\n".join(bib_lines) + "\n")
        print(f"  {split_name}: {len(split_pairs)} images copied")

    print(f"\nBib-only dataset ready at: {DST_DIR}")
    print("Next step: python scripts/train_bib_detector.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract bib-only labels from combined face+bib dataset"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show counts without copying any files",
    )
    args = parser.parse_args()
    extract(preview_only=args.preview)
