"""Train a YOLOv8n-cls blur classifier on the prepared dataset.

Requires: dataset prepared by prepare_blur_dataset.py

Usage:
    python scripts/train_blur_classifier.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Training-Images" / "dataset_clean"


def train(run_name: str, data_dir: Path = DATASET_DIR,
          project_dir: Path = PROJECT_ROOT / "runs" / "classify") -> None:
    """Fine-tune YOLOv8n-cls on the blur classification dataset."""
    # Ultralytics nests a relative project under its own runs/classify; resolve it so
    # best.pt lands where this function looks for it.
    project_dir = project_dir.resolve()
    if not (data_dir / "split_manifest.json").is_file():
        print("ERROR: Dataset not found. Run prepare_blur_dataset.py first.")
        return

    print("=" * 60)
    print("Training YOLOv8n-cls Blur Classifier")
    print(f"  Dataset: {data_dir}")
    print("=" * 60)

    # Load pretrained YOLOv8n-cls (ImageNet weights)
    model = YOLO("yolov8n-cls.pt")

    # Fine-tune on our 4-class blur dataset
    model.train(
        data=str(data_dir),
        epochs=100,
        imgsz=224,
        batch=16,
        patience=20,
        dropout=0.3,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.001,
        label_smoothing=0.1,
        seed=42,
        # Augmentation (built-in)
        hsv_h=0.015,
        # Wide tone/colour jitter: the 2026-08-14 model keyed on brightness and
        # colour (2026-09-29 investigation).
        hsv_s=0.7,
        hsv_v=0.6,
        degrees=15.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        erasing=0.1,
        # Output
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    # Validate
    best_path = project_dir / run_name / "weights" / "best.pt"
    if best_path.exists():
        print(f"\nBest model saved to: {best_path}")
        print("\nRunning validation on best model...")
        best_model = YOLO(str(best_path))
        metrics = best_model.val(data=str(data_dir))
        print(f"  Top-1 accuracy: {metrics.top1:.4f}")
        print(f"  Top-5 accuracy: {metrics.top5:.4f}")
    else:
        print("\nWARNING: best.pt not found after training")

    print("\nNext step: python scripts/export_blur_classifier.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Run name, e.g. blur_cls_event_2026_10")
    parser.add_argument("--data", type=Path, default=DATASET_DIR,
                        help="Prepared dataset with train/, val/ and split_manifest.json")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "classify",
                        help="Folder the run is written to")
    args = parser.parse_args()
    train(args.name, args.data, args.project)
