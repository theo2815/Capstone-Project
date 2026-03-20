"""Export the trained face+bib detector to ONNX format.

Requires: trained model at runs/detect/face_bib_det/weights/best.pt

Usage:
    python scripts/export_face_bib_detector.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Ultralytics nests output under runs/detect/{project}/  -- the training script
# uses project="runs/detect", name="face_bib_det", so the weights end up two
# levels deep.  Check the nested path first, fall back to the flat one.
_NESTED = PROJECT_ROOT / "runs" / "detect" / "detect" / "face_bib_det" / "weights" / "best.pt"
_FLAT = PROJECT_ROOT / "runs" / "detect" / "face_bib_det" / "weights" / "best.pt"
BEST_PT = _NESTED if _NESTED.exists() else _FLAT
MODEL_DIR = PROJECT_ROOT / "models" / "bib_detection"
ONNX_DEST = MODEL_DIR / "yolov8n_bib.onnx"


def export() -> None:
    """Export best.pt to ONNX and copy to models/ directory."""
    if not BEST_PT.exists():
        print(f"ERROR: {BEST_PT} not found. Run train_face_bib_detector.py first.")
        return

    print("=" * 60)
    print("Exporting face+bib detector to ONNX")
    print(f"  Source: {BEST_PT}")
    print("=" * 60)

    model = YOLO(str(BEST_PT))

    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    print(f"  ONNX exported: {onnx_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(onnx_path, ONNX_DEST)
    print(f"  Copied to: {ONNX_DEST}")

    class_names = model.names
    print(f"  Classes: {class_names}")

    # Validate the exported model loads correctly
    print("\nValidating exported model...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(ONNX_DEST), providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        print(f"  Input name:  {input_info.name}")
        print(f"  Input shape: {input_info.shape}")
        print(f"  Input type:  {input_info.type}")
        print("  Validation passed.")
    except Exception as e:
        print(f"  Validation warning: {e}")
        print("  The ONNX file was exported but could not be validated.")

    print(f"\nModel ready for production at: {ONNX_DEST}")


if __name__ == "__main__":
    export()
