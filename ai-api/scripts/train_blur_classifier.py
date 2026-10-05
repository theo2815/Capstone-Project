"""Train the stage-2 blur model: sharp versus subject missed focus, on the whole frame.

Expects <data>/train/<class>/ and <data>/val/<class>/ holding the 640x640 gray pictures
written by scripts/package_blur_stage2.py (classes: defocused_object_portrait, sharp).

Label-safe on purpose. Ultralytics' picture-classification defaults crop the frame,
erase patches and randomly sharpen or blur pictures; each of those can remove or change
the very thing the label is about (vault journal 2026-10-05-blur-new-model-investigation,
finding A3.7). TRAIN_ARGS turns them all off and keeps only a left-right mirror and
brightness/contrast changes.

The run has a fixed length and its LAST weights are used, so the validation folder never
picks the model. It may therefore be the held-out fold of a cross-validation round.

Usage:  python scripts/train_blur_classifier.py --data <folder> --name <run-name>
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_ARGS = {
    "imgsz": 640,
    "epochs": 40,
    "batch": 32,
    "patience": 0,          # never stop early on the validation folder
    "lr0": 0.001,
    "lrf": 0.01,
    "weight_decay": 0.001,
    "dropout": 0.3,
    "seed": 42,
    "scale": 0.0,           # no random crop: the whole frame, always
    "erasing": 0.0,         # no erased patches
    "auto_augment": None,   # no RandAugment (it includes a sharpness change)
    "fliplr": 0.5,          # a mirrored photo keeps its label
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.4,           # brightness/contrast only; the pictures are gray
}


def train(run_name: str, data_dir: Path,
          project_dir: Path = PROJECT_ROOT / "runs" / "classify",
          model_name: str = "yolov8n-cls.pt") -> Path:
    """Fine-tune the pretrained classifier; return the path of the last weights."""
    if not (data_dir / "train").is_dir() or not (data_dir / "val").is_dir():
        raise FileNotFoundError(f"{data_dir} needs train/ and val/ class folders")
    from ultralytics import YOLO

    # Ultralytics nests a relative project under its own runs/classify; resolve it so
    # the weights land where this function says.
    project_dir = project_dir.resolve()
    YOLO(model_name).train(data=str(data_dir), project=str(project_dir), name=run_name,
                           exist_ok=True, verbose=True, **TRAIN_ARGS)
    return project_dir / run_name / "weights" / "last.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Folder with train/ and val/ class folders")
    parser.add_argument("--name", required=True,
                        help="Run name, e.g. blur2_cv_a; never reuse an old run's name")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "classify",
                        help="Where runs are written (on Kaggle: /kaggle/working/runs)")
    parser.add_argument("--model", default="yolov8n-cls.pt",
                        help="Pretrained Ultralytics classifier to start from")
    args = parser.parse_args()
    print(f"last weights: {train(args.name, args.data, args.project, args.model)}")
