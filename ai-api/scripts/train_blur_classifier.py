"""Train a YOLOv8n-cls blur classifier on the prepared dataset.

Requires: dataset prepared by prepare_blur_dataset.py

Usage:
    python scripts/train_blur_classifier.py
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Training-Images" / "dataset"


def train() -> None:
    """Fine-tune YOLOv8n-cls on the blur classification dataset."""
    if not DATASET_DIR.exists():
        print("ERROR: Dataset not found. Run prepare_blur_dataset.py first.")
        return

    print("=" * 60)
    print("Training YOLOv8n-cls Blur Classifier")
    print(f"  Dataset: {DATASET_DIR}")
    print("=" * 60)

    # Load pretrained YOLOv8n-cls (ImageNet weights)
    model = YOLO("yolov8n-cls.pt")

    # Fine-tune on our 4-class blur dataset
    model.train(
        data=str(DATASET_DIR),
        epochs=100,
        imgsz=224,
        batch=16,
        patience=20,
        dropout=0.3,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.001,
        label_smoothing=0.1,
        # Augmentation (built-in)
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        erasing=0.1,
        # Output
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name="blur_cls",
        exist_ok=True,
        verbose=True,
    )

    # Validate
    best_path = PROJECT_ROOT / "runs" / "classify" / "blur_cls" / "weights" / "best.pt"
    if best_path.exists():
        print(f"\nBest model saved to: {best_path}")
        print("\nRunning validation on best model...")
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=str(DATASET_DIR))
        print(f"  Top-1 accuracy: {metrics.top1:.4f}")
        print(f"  Top-5 accuracy: {metrics.top5:.4f}")
    else:
        print("\nWARNING: best.pt not found after training")

    print("\nNext step: python scripts/export_blur_classifier.py")


if __name__ == "__main__":
    train()
