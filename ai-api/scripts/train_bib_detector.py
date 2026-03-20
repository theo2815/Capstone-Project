"""Train a YOLOv8 bib-only detection model.

Expects a dataset at Training-Images/bib_detection/ in YOLO format:
    bib_detection/
        classes.yaml          # single class: bib
        images/train/         # training images
        images/val/           # validation images
        labels/train/         # YOLO-format .txt labels
        labels/val/

Usage:
    python scripts/train_bib_detector.py
    python scripts/train_bib_detector.py --model yolov8s.pt --epochs 150 --imgsz 1280
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Training-Images" / "bib_detection"
CLASSES_YAML = DATASET_DIR / "classes.yaml"


def _ensure_absolute_path_in_yaml() -> None:
    """Patch classes.yaml so ``path:`` is the absolute dataset directory."""
    text = CLASSES_YAML.read_text(encoding="utf-8")
    abs_path_line = f"path: {DATASET_DIR}"
    if re.search(r"^path:\s*\.", text, re.MULTILINE):
        text = re.sub(r"^path:\s*\.", abs_path_line, text, count=1, flags=re.MULTILINE)
        CLASSES_YAML.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 bib detector")
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="Pretrained model to fine-tune (default: yolov8n.pt). "
             "Use yolov8s.pt or yolov8m.pt for higher accuracy.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Train YOLOv8 on bib-only detection dataset."""
    for required in [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "labels" / "train",
        DATASET_DIR / "labels" / "val",
        CLASSES_YAML,
    ]:
        if not required.exists():
            print(f"ERROR: Missing {required}")
            print("Place your bib dataset in Training-Images/bib_detection/ first.")
            return

    n_train = len(list((DATASET_DIR / "images" / "train").glob("*.*")))
    n_val = len(list((DATASET_DIR / "images" / "val").glob("*.*")))
    n_train_lbl = len(list((DATASET_DIR / "labels" / "train").glob("*.txt")))
    n_val_lbl = len(list((DATASET_DIR / "labels" / "val").glob("*.txt")))

    print("=" * 60)
    print("Training YOLOv8 Bib-Only Detector")
    print("=" * 60)
    print(f"  Base model:    {args.model}")
    print(f"  Dataset:       {DATASET_DIR}")
    print(f"  Train images:  {n_train}  (labels: {n_train_lbl})")
    print(f"  Val images:    {n_val}  (labels: {n_val_lbl})")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Image size:    {args.imgsz}")
    print(f"  Batch size:    {args.batch}")
    print("=" * 60)

    if n_train_lbl == 0:
        print("ERROR: No training labels found.")
        return

    _ensure_absolute_path_in_yaml()

    model = YOLO(args.model)

    model.train(
        data=str(CLASSES_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        # Augmentation — aggressive for small datasets
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.15,
        scale=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        # Output
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="bib_det",
        exist_ok=True,
        verbose=True,
    )

    best_path = PROJECT_ROOT / "runs" / "detect" / "bib_det" / "weights" / "best.pt"
    if not best_path.exists():
        # Handle Ultralytics nesting: runs/detect/detect/bib_det/...
        alt = PROJECT_ROOT / "runs" / "detect" / "detect" / "bib_det" / "weights" / "best.pt"
        if alt.exists():
            best_path = alt

    if best_path.exists():
        print(f"\nBest model saved to: {best_path}")
        print("\nRunning validation on best model...")
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=str(CLASSES_YAML))
        print(f"  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
    else:
        print("\nWARNING: best.pt not found after training")

    print("\nNext step: python scripts/export_bib_detector.py")


if __name__ == "__main__":
    train(parse_args())
