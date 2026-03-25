"""Prepare the blur classification dataset in ImageFolder format for YOLOv8-cls.

Reads raw images from Training-Images/ subdirectories, applies augmentation
to balance under-represented classes, and creates an 80/20 train/val split.

The `sharp` class is sourced from two directories (Sharp Object in portrait/
and Sharp_images/) which are merged during preparation.

Usage:
    python scripts/prepare_blur_dataset.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "Training-Images"
DATASET_DIR = TRAINING_DIR / "dataset"

# Mapping: class_name -> source directory name(s)
# Classes with multiple source directories use a list.
CLASS_DIRS: dict[str, str | list[str]] = {
    "sharp": ["Sharp Object in portrait", "Sharp_images"],
    "defocused_object_portrait": "Defocused object in portrait",
    "defocused_blurred": "defocused_blurred",
    "motion_blurred": "motion_blurred",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

# Augmentation target: how many total images per class (original + augmented)
# For classes below this count, augment to reach it.
AUGMENTATION_TARGET = 300

RANDOM_SEED = 42
VAL_RATIO = 0.2


def list_images(directory: Path) -> list[Path]:
    """List all image files in a directory."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def augment_image(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a random combination of augmentations to an image.

    Uses OpenCV only (no extra dependencies).
    """
    h, w = img.shape[:2]
    result = img.copy()

    # 1. Random horizontal flip (50%)
    if rng.random() < 0.5:
        result = cv2.flip(result, 1)

    # 2. Random rotation (-15 to +15 degrees)
    angle = rng.uniform(-15, 15)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # 3. Random crop and resize back (85-100% of original)
    scale = rng.uniform(0.85, 1.0)
    crop_h, crop_w = int(h * scale), int(w * scale)
    y_off = rng.randint(0, h - crop_h)
    x_off = rng.randint(0, w - crop_w)
    result = result[y_off : y_off + crop_h, x_off : x_off + crop_w]
    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    # 4. Color jitter: brightness and contrast
    alpha = rng.uniform(0.8, 1.2)  # contrast
    beta = rng.uniform(-20, 20)  # brightness
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    # 5. Gaussian noise (30%)
    if rng.random() < 0.3:
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(
            0, 10, result.shape
        ).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return result


def _unique_name(img_path: Path, seen: set[str]) -> str:
    """Return a unique filename, appending a suffix on collision."""
    name = img_path.name
    if name not in seen:
        seen.add(name)
        return name
    stem, suffix = img_path.stem, img_path.suffix
    i = 1
    while True:
        candidate = f"{stem}_{i}{suffix}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        i += 1


def prepare_dataset() -> None:
    """Build the ImageFolder dataset with augmentation and train/val split."""
    rng = random.Random(RANDOM_SEED)

    # Clean previous dataset
    if DATASET_DIR.exists():
        print(f"Removing existing dataset at {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    print("=" * 60)
    print("Preparing blur classification dataset")
    print("=" * 60)

    for split in ("train", "val"):
        for cls_name in CLASS_DIRS:
            (DATASET_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

    total_stats: dict[str, dict[str, int]] = {}

    for cls_name, dir_names in CLASS_DIRS.items():
        # Support single string or list of directories per class
        if isinstance(dir_names, str):
            dir_names = [dir_names]

        images: list[Path] = []
        for dir_name in dir_names:
            src_dir = TRAINING_DIR / dir_name
            found = list_images(src_dir)
            if not found:
                print(f"  WARNING: No images found in {src_dir}")
            images.extend(found)

        n_originals = len(images)

        if n_originals == 0:
            print(f"  WARNING: No images found for class '{cls_name}'")
            continue

        # Shuffle deterministically
        shuffled = images.copy()
        rng.shuffle(shuffled)

        # Split into train/val
        n_val = max(1, int(n_originals * VAL_RATIO))
        val_images = shuffled[:n_val]
        train_images = shuffled[n_val:]

        source_dirs = ", ".join(dir_names)
        print(f"\n{cls_name}:")
        print(f"  Source: {source_dirs}")
        print(f"  Original images: {n_originals}")
        print(f"  Train: {len(train_images)}, Val: {n_val}")

        # Copy validation images (no augmentation)
        val_count = 0
        seen_names: set[str] = set()
        for img_path in val_images:
            name = _unique_name(img_path, seen_names)
            dest = DATASET_DIR / "val" / cls_name / name
            shutil.copy2(img_path, dest)
            val_count += 1

        # Copy train images
        train_count = 0
        for img_path in train_images:
            name = _unique_name(img_path, seen_names)
            dest = DATASET_DIR / "train" / cls_name / name
            shutil.copy2(img_path, dest)
            train_count += 1

        # Augment if this class needs more images (train only — never augment val)
        if n_originals < AUGMENTATION_TARGET:
            n_augmented_needed = AUGMENTATION_TARGET - n_originals

            print(f"  Augmenting: {n_augmented_needed} train (val uses originals only)")

            # Generate train augmentations only
            for i in range(n_augmented_needed):
                src_path = train_images[i % len(train_images)]
                img = cv2.imread(str(src_path))
                if img is None:
                    continue
                aug = augment_image(img, rng)
                stem = src_path.stem
                dest = DATASET_DIR / "train" / cls_name / f"{stem}_aug{i:04d}.jpg"
                cv2.imwrite(str(dest), aug)
                train_count += 1

        total_stats[cls_name] = {"train": train_count, "val": val_count}

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"{'Class':<25} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 51)
    grand_train = grand_val = 0
    for cls_name, counts in total_stats.items():
        t, v = counts["train"], counts["val"]
        grand_train += t
        grand_val += v
        print(f"{cls_name:<25} {t:>8} {v:>8} {t + v:>8}")
    print("-" * 51)
    print(f"{'TOTAL':<25} {grand_train:>8} {grand_val:>8} {grand_train + grand_val:>8}")
    print(f"\nDataset written to: {DATASET_DIR}")


if __name__ == "__main__":
    prepare_dataset()
