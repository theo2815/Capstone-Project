"""Export the COCO person detector that stage 2 of the blur classifier runs on.

Ultralytics' pretrained `yolov8n-seg.pt` (COCO, 80 classes, 7 MB) is downloaded once, exported to
ONNX with a fixed 640 x 640 input, copied to models/person_detector/yolov8n_seg_person.onnx and
recorded in models/manifest.json with its hash. Ultralytics and torch are tools of this script
only: serving loads the ONNX file with ONNX Runtime (src/ml/blur/person_detector.py).

Licence: the weights and Ultralytics are AGPL-3.0, like the project's other YOLO models
(owner decision 2026-10-06, vault ai-api/decisions.md).

Usage (from ai-api/):  python scripts/export_person_detector.py [--weights yolov8n.pt]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "person_detector"
ONNX_DEST = MODEL_DIR / "yolov8n_seg_person.onnx"
MANIFEST = PROJECT_ROOT / "models" / "manifest.json"
IMGSZ = 640
EXPECTED_OUTPUTS = (
    [1, 116, 8400],
    [1, 32, 160, 160],
)  # boxes + classes + mask weights; mask prototypes


def sha256_of(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def update_manifest(manifest: Path, onnx: Path, weights_name: str) -> None:
    """Record the detector's identity; BlurClassifier refuses a file with another hash."""
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["models"]["person_detector"] = {
        "name": "YOLOv8n-seg COCO person detector (stage 2 of the blur classifier)",
        "file": "person_detector/yolov8n_seg_person.onnx",
        "sha256": sha256_of(onnx),
        "source": f"Ultralytics {weights_name} (COCO) -> scripts/export_person_detector.py",
        "export": {"imgsz": IMGSZ, "dynamic": False, "simplify": False},
        "input_shape": [1, 3, IMGSZ, IMGSZ],
        "output_shapes": list(EXPECTED_OUTPUTS),
        "notes": (
            "Only the person class (index 0) is read; each person's mask gives its own "
            "pixels for the sharpness measure. Letterbox, score filter, NMS and mask decoding "
            "live in src/ml/blur/person_detector.py, so serving needs no torch or Ultralytics. "
            "Licence AGPL-3.0 (owner decision 2026-10-06)."
        ),
    }
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def verify(onnx: Path) -> None:
    """Fail loudly here rather than at service start."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    in_shape = session.get_inputs()[0].shape
    out_shapes = [list(o.shape) for o in session.get_outputs()]
    print(f"  Input shape: {in_shape}  Output shapes: {out_shapes}")
    if list(in_shape) != [1, 3, IMGSZ, IMGSZ] or out_shapes != [list(s) for s in EXPECTED_OUTPUTS]:
        raise SystemExit(
            "ERROR: unexpected detector shapes; stage 2 expects a static "
            f"[1, 3, {IMGSZ}, {IMGSZ}] input and the outputs {list(EXPECTED_OUTPUTS)}"
        )


def export(weights: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)  # downloads the pretrained file on first use
    onnx_path = Path(model.export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=False))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, ONNX_DEST)
    print(f"  Copied to: {ONNX_DEST}")
    verify(ONNX_DEST)
    update_manifest(MANIFEST, ONNX_DEST, Path(weights).name)
    print(f"  sha256: {sha256_of(ONNX_DEST)}")
    return ONNX_DEST


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="yolov8n-seg.pt",
        help="Ultralytics segmentation weights to export (COCO person = class 0)",
    )
    args = parser.parse_args()
    print("=" * 60)
    print("Exporting the person detector to ONNX")
    print("=" * 60)
    export(args.weights)
