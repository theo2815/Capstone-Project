"""Export a trained stage-2 blur model (sharp vs subject missed focus) to ONNX.

Requires: a trained run at runs/classify/<run>/weights/last.pt. The LAST weights are
used on purpose: runs have a fixed length, so the validation folder never picks the
model (see scripts/train_blur_classifier.py).

Usage:
    python scripts/export_blur_classifier.py --run <run> --candidate
        a cross-validation fold: writes last.onnx + class_names.json beside the weights
    python scripts/export_blur_classifier.py --run <run> --version <YYYY-MM-DD-name>
        the model to ship: also copies into models/blur_classifier/ and updates the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "blur_classifier"
ONNX_DEST = MODEL_DIR / "blur_classifier.onnx"
CLASS_NAMES_DEST = MODEL_DIR / "class_names.json"
MANIFEST = PROJECT_ROOT / "models" / "manifest.json"
IMGSZ = 640


def update_manifest(manifest: Path, onnx: Path, labels: Path, version: str) -> None:
    """Record the exported artifact's identity; the blur profile refuses mismatches."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entry = data["models"]["blur_classifier"]
    for key, path in (("sha256", onnx), ("labels_sha256", labels)):
        with path.open("rb") as f:
            entry[key] = hashlib.file_digest(f, "sha256").hexdigest()
    entry["artifact_version"] = version
    entry["name"] = "YOLOv8n-cls blur classifier, stage 2 (sharp vs subject missed focus)"
    entry["classes"] = json.loads(labels.read_text(encoding="utf-8"))
    entry["export"] = {"imgsz": IMGSZ, "dynamic": True, "simplify": False}
    entry["notes"] = ("Stage 2 of the two-stage blur detector: sees the whole frame as a "
                      f"{IMGSZ}x{IMGSZ} padded gray square. Whole-frame soft and motion are "
                      "decided before it by src/ml/blur/whole_frame.py. Quality evidence: "
                      "docs/ai-system-overview.md.")
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def export(run_name: str, version: str | None, candidate: bool = False) -> None:
    """Export last.pt to ONNX; unless `candidate`, install it under models/ too."""
    from ultralytics import YOLO

    last_pt = PROJECT_ROOT / "runs" / "classify" / run_name / "weights" / "last.pt"
    if not last_pt.exists():
        print(f"ERROR: {last_pt} not found. Run train_blur_classifier.py first.")
        return

    print("=" * 60)
    print("Exporting blur classifier to ONNX")
    print(f"  Source: {last_pt}")
    print("=" * 60)

    model = YOLO(str(last_pt))

    # Export to ONNX with a DYNAMIC batch axis so the API/worker can run true
    # batched inference (BlurClassifier.classify_batch stacks N images into one
    # session.run). simplify=False on purpose: the onnx-simplifier can fold the
    # dynamic batch dim back to a static 1, and ONNX Runtime already applies full
    # graph optimization at load — so skipping simplify keeps the axis dynamic at
    # negligible runtime cost.
    onnx_path = model.export(format="onnx", imgsz=IMGSZ, dynamic=True, simplify=False)
    print(f"  ONNX exported: {onnx_path}")
    class_names_list = [model.names[i] for i in sorted(model.names)]
    if candidate:  # a cross-validation fold: leave models/ and the manifest alone
        labels = Path(onnx_path).with_name("class_names.json")
        labels.write_text(json.dumps(class_names_list, indent=2), encoding="utf-8")
        print(f"  Class names: {class_names_list} -> {labels}")
        return

    # Create destination directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Copy ONNX to models directory
    shutil.copy2(onnx_path, ONNX_DEST)
    print(f"  Copied to: {ONNX_DEST}")

    # Save class names mapping
    with open(CLASS_NAMES_DEST, "w") as f:
        json.dump(class_names_list, f, indent=2)
    print(f"  Class names: {class_names_list}")
    print(f"  Saved to: {CLASS_NAMES_DEST}")
    update_manifest(MANIFEST, ONNX_DEST, CLASS_NAMES_DEST, version)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run name under runs/classify")
    parser.add_argument("--version", help="Artifact version, e.g. 2026-10-15-two-stage")
    parser.add_argument("--candidate", action="store_true",
                        help="Only write last.onnx + class_names.json beside the weights")
    args = parser.parse_args()
    if not args.candidate and not args.version:
        parser.error("--version is required unless --candidate is given")
    export(args.run, args.version, args.candidate)
