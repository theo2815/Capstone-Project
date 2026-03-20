"""Train a YOLOv8n face+bib detection model.

Requires: auto-annotated dataset from auto_annotate_face_bib.py
(images + YOLO-format labels split into train/val).

Usage:
    python scripts/train_face_bib_detector.py
"""

from __future__ import annotations

import re
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Training-Images" / "face_bib_detection"
CLASSES_YAML = DATASET_DIR / "classes.yaml"


def _ensure_absolute_path_in_yaml() -> None:
    """Patch classes.yaml so ``path:`` is the absolute dataset directory.

    Ultralytics resolves relative ``path:`` values from the *working directory*,
    not from the YAML file location.  This silently fixes it if needed.
    """
    text = CLASSES_YAML.read_text(encoding="utf-8")
    abs_path_line = f"path: {DATASET_DIR}"
    if re.search(r"^path:\s*\.", text, re.MULTILINE):
        text = re.sub(r"^path:\s*\.", abs_path_line, text, count=1, flags=re.MULTILINE)
        CLASSES_YAML.write_text(text, encoding="utf-8")


def train() -> None:
    """Train YOLOv8n on face+bib detection dataset."""
    # Validate dataset structure
    for required in [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "labels" / "train",
        DATASET_DIR / "labels" / "val",
        CLASSES_YAML,
    ]:
        if not required.exists():
            print(f"ERROR: Missing {required}")
            print("Run auto_annotate_face_bib.py first.")
            return

    # Count files
    n_train_images = len(list((DATASET_DIR / "images" / "train").glob("*.*")))
    n_val_images = len(list((DATASET_DIR / "images" / "val").glob("*.*")))
    n_train_labels = len(list((DATASET_DIR / "labels" / "train").glob("*.txt")))
    n_val_labels = len(list((DATASET_DIR / "labels" / "val").glob("*.txt")))

    print("=" * 60)
    print("Training YOLOv8n Face+Bib Detector")
    print("=" * 60)
    print(f"  Dataset:       {DATASET_DIR}")
    print(f"  Classes YAML:  {CLASSES_YAML}")
    print(f"  Train images:  {n_train_images}")
    print(f"  Train labels:  {n_train_labels}")
    print(f"  Val images:    {n_val_images}")
    print(f"  Val labels:    {n_val_labels}")
    print("=" * 60)

    if n_train_labels == 0:
        print("ERROR: No training labels found. Run auto_annotate_face_bib.py first.")
        return

    # Ensure classes.yaml uses absolute path so it works from any working directory
    _ensure_absolute_path_in_yaml()

    # Load pretrained YOLOv8n (COCO weights)
    model = YOLO("yolov8n.pt")

    # Train on face+bib detection dataset
    model.train(
        data=str(CLASSES_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Output
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="face_bib_det",
        exist_ok=True,
        verbose=True,
    )

    # Validate
    best_path = (
        PROJECT_ROOT / "runs" / "detect" / "face_bib_det" / "weights" / "best.pt"
    )
    if best_path.exists():
        print(f"\nBest model saved to: {best_path}")
        print("\nRunning validation on best model...")
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=str(CLASSES_YAML))
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
    else:
        print("\nWARNING: best.pt not found after training")

    print("\nNext step: python scripts/export_face_bib_detector.py")


if __name__ == "__main__":
    train()
