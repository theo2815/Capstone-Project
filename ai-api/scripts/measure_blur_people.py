"""Measure the people in every sharp and missed-focus photo of the fold tree, once.

For each photo: upload emulation (fit 1280, JPEG q90, scripts/blur_gate.resize_bytes) ->
server decode -> person detector (boxes + masks) -> per-person sharpness numbers
(src/ml/blur/people.py).
One JSON line per photo goes to Training-Images/blur_v2/people_measures.jsonl:

    {"fold": "a", "event": "...", "label": "sharp", "name": "<sha16>.jpg",
     "W": 853, "H": 1280, "people": [{"box": [...], "conf": ..., "es": ..., ...}, ...]}

The fit script (scripts/fit_blur_people_rule.py) reads this cache, so changing the rule's
numbers never means re-running the detector. Resumable: photos already in the file are
skipped. Boxes are kept from detector confidence 0.2 upward; the rule may use a higher cut.

Usage (from ai-api/):  python scripts/measure_blur_people.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = ROOT / "Training-Images" / "blur_v2"
OUT = FOLDS_DIR / "people_measures.jsonl"
DETECTOR = ROOT / "models" / "person_detector" / "yolov8n_seg_person.onnx"
STAGE2_CLASSES = ("defocused_object_portrait", "sharp")
CACHE_MIN_CONF = 0.2
_detector = None


def _init() -> None:
    """One ONNX session per worker, one quiet thread: eight workers share the cores evenly."""
    global _detector
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import cv2
    import onnxruntime as ort

    from src.ml.blur.person_detector import PersonDetector

    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(str(DETECTOR), options, providers=["CPUExecutionProvider"])
    _detector = PersonDetector(session, min_conf=CACHE_MIN_CONF)


def measure_one(job: tuple[str, str, str, str]) -> dict:
    """(fold, event, label, path) -> the cache line for that photo."""
    import cv2
    import numpy as np

    from scripts.blur_gate import resize_bytes
    from src.ml.blur import people

    fold, event, label, path = job
    upload = np.frombuffer(resize_bytes(Path(path), 1280, 90), dtype=np.uint8)
    image = cv2.imdecode(upload, cv2.IMREAD_COLOR)
    found = people.measure(image, _detector.detect(image))
    return {
        "fold": fold,
        "event": event,
        "label": label,
        "name": Path(path).name,
        "W": image.shape[1],
        "H": image.shape[0],
        "people": [people.person_to_dict(p) for p in found],
    }


def jobs_for(folds_dir: Path) -> list[tuple[str, str, str, str]]:
    jobs = []
    for fold in sorted(folds_dir.glob("fold_*")):
        for event in sorted(p for p in fold.iterdir() if p.is_dir()):
            for label in STAGE2_CLASSES:
                for photo in sorted((event / label).glob("*")) if (event / label).is_dir() else []:
                    if photo.is_file():
                        jobs.append((fold.name[-1], event.name, label, str(photo)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=FOLDS_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not DETECTOR.is_file():
        raise SystemExit(f"{DETECTOR} is missing: run scripts/export_person_detector.py first")
    jobs = jobs_for(args.folds)
    done = set()
    if args.out.exists():
        done = {
            (row["fold"], row["event"], row["label"], row["name"])
            for row in map(json.loads, args.out.read_text(encoding="utf-8").splitlines())
        }
    todo = [j for j in jobs if (j[0], j[1], j[2], Path(j[3]).name) not in done]
    print(f"{len(jobs)} photos, {len(done)} already measured, {len(todo)} to do", flush=True)
    started = time.time()
    with (
        ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool,
        args.out.open("a", encoding="utf-8") as out,
    ):
        for n, row in enumerate(pool.map(measure_one, todo, chunksize=8), 1):
            out.write(json.dumps(row) + "\n")
            if n % 500 == 0:
                out.flush()
                print(f"  {n}/{len(todo)}  {time.time() - started:.0f} s", flush=True)
    print(f"{len(todo)} photos measured in {time.time() - started:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
