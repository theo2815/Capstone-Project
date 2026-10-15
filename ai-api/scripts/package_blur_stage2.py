"""Write the stage-2 training pictures: exactly what the model sees when serving.

For every sharp and missed-focus photo in the folds: upload emulation (fit 1280, JPEG
q90) -> server decode -> whole_frame.square_gray_view -> PNG. PNG is lossless, so
training sees the same pixels serving builds. Whole-frame soft and motion photos are
stage 1's job and are left out.

    Training-Images/blur_v2/kaggle/blur_stage2/fold_<a-e>/<class>/<sha16>.png
    Training-Images/blur_v2/kaggle/blur_stage2/train_blur_classifier.py   (for the notebook)

Usage (from ai-api/):  python scripts/package_blur_stage2.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDS_DIR = ROOT / "Training-Images" / "blur_v2"
STAGE2_CLASSES = ("defocused_object_portrait", "sharp")
VIEW_SIZE = 640


def write_view(job: tuple[str, str, int]) -> None:
    """One photo -> one square gray PNG, through the real upload and decode path."""
    source, target, size = job
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import cv2
    import numpy as np

    from scripts.blur_gate import resize_bytes
    from src.ml.blur import whole_frame

    upload = np.frombuffer(resize_bytes(Path(source), 1280, 90), dtype=np.uint8)
    view = whole_frame.square_gray_view(cv2.imdecode(upload, cv2.IMREAD_COLOR), size)
    if not cv2.imwrite(target, view):
        raise OSError(f"could not write {target}")


def package(folds_dir: Path = FOLDS_DIR, size: int = VIEW_SIZE, workers: int = 8) -> int:
    """Write every stage-2 picture; return how many were written."""
    out = folds_dir / "kaggle" / "blur_stage2"
    if out.exists():
        raise FileExistsError(f"Refusing to replace existing package: {out}")
    jobs = []
    for fold in sorted(folds_dir.glob("fold_*")):
        for label in STAGE2_CLASSES:
            for photo in sorted(fold.glob(f"*/{label}/*")):
                target = out / fold.name / label / f"{photo.stem}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((str(photo), str(target), size))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(write_view, jobs, chunksize=16))
    else:
        for job in jobs:
            write_view(job)
    shutil.copy2(Path(__file__).with_name("train_blur_classifier.py"), out)
    return len(jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=Path, default=FOLDS_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(f"{package(args.folds, workers=args.workers)} pictures written under "
          f"{args.folds / 'kaggle' / 'blur_stage2'}")
