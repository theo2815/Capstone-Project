"""Export the trained blur classifier to ONNX format.

Requires: trained model at runs/classify/blur_cls/weights/best.pt

Usage:
    python scripts/export_blur_classifier.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_PT = PROJECT_ROOT / "runs" / "classify" / "blur_cls" / "weights" / "best.pt"
MODEL_DIR = PROJECT_ROOT / "models" / "blur_classifier"
ONNX_DEST = MODEL_DIR / "blur_classifier.onnx"
CLASS_NAMES_DEST = MODEL_DIR / "class_names.json"


def export() -> None:
    """Export best.pt to ONNX and copy to models/ directory."""
    if not BEST_PT.exists():
        print(f"ERROR: {BEST_PT} not found. Run train_blur_classifier.py first.")
        return

    print("=" * 60)
    print("Exporting blur classifier to ONNX")
    print(f"  Source: {BEST_PT}")
    print("=" * 60)

    model = YOLO(str(BEST_PT))

    # Export to ONNX
    onnx_path = model.export(format="onnx", imgsz=224, simplify=True)
    print(f"  ONNX exported: {onnx_path}")

    # Create destination directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Copy ONNX to models directory
    shutil.copy2(onnx_path, ONNX_DEST)
    print(f"  Copied to: {ONNX_DEST}")

    # Save class names mapping
    class_names = model.names  # dict: {0: 'class0', 1: 'class1', ...}
    class_names_list = [class_names[i] for i in sorted(class_names.keys())]
    with open(CLASS_NAMES_DEST, "w") as f:
        json.dump(class_names_list, f, indent=2)
    print(f"  Class names: {class_names_list}")
    print(f"  Saved to: {CLASS_NAMES_DEST}")

    # Verify
    print("\nVerification:")
    print(f"  python -c \"import onnxruntime as ort; s = ort.InferenceSession('{ONNX_DEST}'); print(s.get_inputs()[0].shape)\"")


if __name__ == "__main__":
    export()
