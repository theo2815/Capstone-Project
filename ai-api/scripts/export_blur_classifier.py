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

    # Export to ONNX with a DYNAMIC batch axis so the API/worker can run true
    # batched inference (BlurClassifier.classify_batch stacks N images into one
    # session.run). simplify=False on purpose: the onnx-simplifier can fold the
    # dynamic batch dim back to a static 1, and ONNX Runtime already applies full
    # graph optimization at load — so skipping simplify keeps the axis dynamic at
    # negligible runtime cost.
    onnx_path = model.export(format="onnx", imgsz=224, dynamic=True, simplify=False)
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

    # Verify the dynamic batch axis survived export. BlurClassifier flips
    # _dynamic_batch=True only when input dim 0 is None/-1/N/"batch"; if it stayed
    # a literal 1, classify_batch silently falls back to a per-image loop (no
    # batched speedup) — so fail loudly here instead of discovering it in prod.
    print("\nVerification:")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(ONNX_DEST), providers=["CPUExecutionProvider"])
        in_shape = sess.get_inputs()[0].shape
        batch_dim = in_shape[0]
        is_dynamic = batch_dim is None or (
            isinstance(batch_dim, (int, str)) and str(batch_dim) in ("-1", "N", "batch")
        )
        print(f"  Input shape: {in_shape}  (dynamic batch: {is_dynamic})")
        if not is_dynamic:
            print("  WARNING: batch axis is STATIC — classify_batch will run a per-image")
            print("           loop (no batched speedup). Re-run and confirm Ultralytics")
            print("           produced a dynamic axis (try a newer ultralytics, or check")
            print("           that simplify did not fold it).")
    except Exception as e:
        print(f"  Could not verify ONNX input shape: {e}")


if __name__ == "__main__":
    export()
