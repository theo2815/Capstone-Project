"""Export the trained bib-only detector to ONNX format.

This exports the bib-only model trained by train_bib_detector.py.  The combined
face+bib detector achieved bib mAP50 = 0.561, so this dedicated bib model is the
production path for bib detection going forward.

Both this and export_face_bib_detector.py write to the same production path
(models/bib_detection/yolov8n_bib.onnx) because BibDetector filters by class
name "bib" and works with either model type.  This script will prompt before
overwriting an existing model.

Requires: trained model from train_bib_detector.py

Usage:
    python scripts/export_bib_detector.py           # prompts if production model exists
    python scripts/export_bib_detector.py --force   # overwrites without prompting
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Check both possible nesting patterns
_NESTED = PROJECT_ROOT / "runs" / "detect" / "detect" / "bib_det" / "weights" / "best.pt"
_FLAT = PROJECT_ROOT / "runs" / "detect" / "bib_det" / "weights" / "best.pt"
BEST_PT = _NESTED if _NESTED.exists() else _FLAT

MODEL_DIR = PROJECT_ROOT / "models" / "bib_detection"
ONNX_DEST = MODEL_DIR / "yolov8n_bib.onnx"


def export(force: bool = False) -> None:
    """Export best.pt to ONNX and copy to models/ directory."""
    if not BEST_PT.exists():
        print(f"ERROR: {BEST_PT} not found. Run train_bib_detector.py first.")
        return

    # Guard: confirm before overwriting an existing model.
    if ONNX_DEST.exists() and not force:
        print("=" * 60)
        print("NOTE: A model already exists at the production path:")
        print(f"  {ONNX_DEST}")
        print()
        print("This will be replaced with the new bib-only model.")
        print("To proceed, re-run with --force:")
        print("  python scripts/export_bib_detector.py --force")
        print("=" * 60)
        return

    print("=" * 60)
    print("Exporting bib-only detector to ONNX")
    print(f"  Source: {BEST_PT}")
    print("=" * 60)

    model = YOLO(str(BEST_PT))
    print(f"  Classes: {model.names}")

    onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
    print(f"  ONNX exported: {onnx_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, ONNX_DEST)
    print(f"  Copied to: {ONNX_DEST}")

    # Validate
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

    print(f"\nBib-only model deployed to: {ONNX_DEST}")
    print("BibDetector will use this bib-only model (filters by class name 'bib').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export bib-only detector to ONNX")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing production model without prompting",
    )
    args = parser.parse_args()
    export(force=args.force)
