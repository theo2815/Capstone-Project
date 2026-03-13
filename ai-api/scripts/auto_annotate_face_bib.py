"""Auto-annotate face+bib detection dataset using pre-trained models.

Uses InsightFace (RetinaFace) for face detection and PaddleOCR for bib
text region detection. Generates YOLO-format .txt label files, then
splits into 80/20 train/val.

Optimizations:
- Images are resized to max 800px before PaddleOCR (7s vs 8min per image)
- Text orientation detection disabled for speed
- Coordinates scaled back to original image dimensions

Usage:
    python scripts/auto_annotate_face_bib.py                # annotate all images
    python scripts/auto_annotate_face_bib.py --preview 5    # preview first 5 images
    python scripts/auto_annotate_face_bib.py --visualize 10 # save 10 annotated samples
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_BIB_CHAR_RE = re.compile(r"[A-Za-z0-9\-_]")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# PaddleOCR environment setup (must be before import)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

DATASET_DIR = PROJECT_ROOT / "Training-Images" / "face_bib_detection"
ALL_IMAGES_DIR = DATASET_DIR / "images" / "train"

# Classes (must match classes.yaml)
FACE_CLASS_ID = 0
BIB_CLASS_ID = 1

# Face detection settings
FACE_CONFIDENCE_THRESHOLD = 0.5

# Bib detection settings
BIB_MIN_DIGITS = 2          # Require at least 2 digits to be a bib number
BIB_BOX_EXPAND_RATIO = 0.6  # Expand text box to approximate full bib card
BIB_MIN_AREA_RATIO = 0.003  # Min bib area as fraction of image
BIB_MAX_AREA_RATIO = 0.15   # Max bib area as fraction of image
BIB_MERGE_IOU_THRESHOLD = 0.3

# OCR resize: resize images before PaddleOCR for speed
OCR_MAX_DIM = 800

RANDOM_SEED = 42
VAL_RATIO = 0.2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_face_detector():
    """Load InsightFace FaceAnalysis for face detection."""
    from insightface.app import FaceAnalysis

    print("Loading InsightFace (RetinaFace)...")
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(PROJECT_ROOT / "models"),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("  Face detector loaded.")
    return app


def load_text_detector():
    """Load PaddleOCR for text/bib region detection."""
    from paddleocr import PaddleOCR

    print("Loading PaddleOCR...")
    ocr = PaddleOCR(
        use_textline_orientation=False,  # Disabled for speed
        lang="en",
    )
    print("  Text detector loaded.")
    return ocr


def _resize_for_ocr(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Resize image for faster OCR processing. Returns (resized_img, scale)."""
    h, w = image.shape[:2]
    scale = OCR_MAX_DIM / max(h, w)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h)), scale
    return image, 1.0


def detect_faces(face_app, image: np.ndarray) -> list[list[float]]:
    """Detect faces, return list of [x1, y1, x2, y2, confidence]."""
    faces = face_app.get(image)
    results = []
    h, w = image.shape[:2]
    for face in faces:
        conf = float(face.det_score)
        if conf >= FACE_CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                results.append([x1, y1, x2, y2, conf])
    return results


def detect_bibs_paddleocr(ocr, image: np.ndarray) -> list[list[float]]:
    """Detect bib regions using PaddleOCR text detection on resized image.

    Strategy: resize image -> find text boxes with digits -> expand ->
    scale coordinates back to original -> merge overlapping boxes.

    Returns list of [x1, y1, x2, y2, confidence] in original image coords.
    """
    orig_h, orig_w = image.shape[:2]
    orig_area = orig_h * orig_w

    # Resize for speed
    img_small, scale = _resize_for_ocr(image)

    try:
        results = list(ocr.predict(img_small))
    except Exception:
        try:
            raw = ocr.ocr(img_small, cls=True)
            if not raw or not raw[0]:
                return []
            dt_polys = []
            rec_texts = []
            rec_scores = []
            for line in raw[0]:
                poly, (text, score) = line
                dt_polys.append(poly)
                rec_texts.append(text)
                rec_scores.append(score)
            results = [{"dt_polys": dt_polys, "rec_texts": rec_texts, "rec_scores": rec_scores}]
        except Exception:
            return []

    if not results:
        return []

    result = results[0]

    if hasattr(result, "get"):
        dt_polys = result.get("dt_polys", [])
        rec_texts = result.get("rec_texts", [])
        rec_scores = result.get("rec_scores", [])
    elif hasattr(result, "dt_polys"):
        dt_polys = result.dt_polys if result.dt_polys is not None else []
        rec_texts = result.rec_texts if result.rec_texts is not None else []
        rec_scores = result.rec_scores if result.rec_scores is not None else []
    else:
        return []

    if not dt_polys:
        return []

    # Find text boxes containing bib-like text (digits, optionally with letters/hyphens)
    digit_boxes = []
    for poly, text, score in zip(dt_polys, rec_texts, rec_scores):
        cleaned = "".join(_BIB_CHAR_RE.findall(str(text))).strip("-_")
        digit_count = sum(c.isdigit() for c in cleaned)
        if digit_count < BIB_MIN_DIGITS:
            continue

        poly = np.array(poly)
        # Scale coordinates back to original image size
        x1 = float(poly[:, 0].min()) / scale
        y1 = float(poly[:, 1].min()) / scale
        x2 = float(poly[:, 0].max()) / scale
        y2 = float(poly[:, 1].max()) / scale

        digit_boxes.append([x1, y1, x2, y2, float(score), cleaned])

    if not digit_boxes:
        return []

    # Expand boxes to approximate full bib card area
    expanded_boxes = []
    for x1, y1, x2, y2, score, _text in digit_boxes:
        bw = x2 - x1
        bh = y2 - y1

        ex1 = max(0, x1 - bw * BIB_BOX_EXPAND_RATIO)
        ey1 = max(0, y1 - bh * BIB_BOX_EXPAND_RATIO)
        ex2 = min(orig_w, x2 + bw * BIB_BOX_EXPAND_RATIO)
        ey2 = min(orig_h, y2 + bh * BIB_BOX_EXPAND_RATIO)

        area = (ex2 - ex1) * (ey2 - ey1)
        area_ratio = area / orig_area

        if area_ratio < BIB_MIN_AREA_RATIO or area_ratio > BIB_MAX_AREA_RATIO:
            continue

        expanded_boxes.append([ex1, ey1, ex2, ey2, score])

    merged = _merge_boxes(expanded_boxes, BIB_MERGE_IOU_THRESHOLD)
    return merged


def _compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2, ...]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def _merge_boxes(boxes: list, iou_threshold: float) -> list:
    """Merge overlapping boxes into single bounding boxes."""
    if len(boxes) <= 1:
        return boxes

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        group = [boxes[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if _compute_iou(boxes[i], boxes[j]) > iou_threshold:
                group.append(boxes[j])
                used[j] = True

        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        conf = max(b[4] for b in group)

        merged.append([x1, y1, x2, y2, conf])

    return merged


def to_yolo_format(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int, class_id: int,
) -> str:
    """Convert [x1, y1, x2, y2] xyxy to YOLO format string."""
    x_center = max(0, min(1, ((x1 + x2) / 2) / img_w))
    y_center = max(0, min(1, ((y1 + y2) / 2) / img_h))
    width = max(0, min(1, (x2 - x1) / img_w))
    height = max(0, min(1, (y2 - y1) / img_h))
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def visualize_annotations(
    image: np.ndarray,
    faces: list[list[float]],
    bibs: list[list[float]],
    save_path: Path,
) -> None:
    """Draw bounding boxes on image and save for visual verification."""
    vis = image.copy()
    for x1, y1, x2, y2, conf in faces:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(vis, f"face {conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for bib in bibs:
        x1, y1, x2, y2, conf = bib[:5]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(vis, f"bib {conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(save_path), vis)


def split_train_val(dataset_dir: Path, val_ratio: float, seed: int) -> tuple[int, int]:
    """Split images + labels into train/val sets (80/20)."""
    rng = random.Random(seed)

    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val = dataset_dir / "labels" / "val"

    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        p for p in images_train.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    annotated = [
        img for img in all_images
        if (labels_train / f"{img.stem}.txt").exists()
        and (labels_train / f"{img.stem}.txt").stat().st_size > 0
    ]

    rng.shuffle(annotated)
    n_val = max(1, int(len(annotated) * val_ratio))
    val_set = annotated[:n_val]

    moved = 0
    for img_path in val_set:
        label_path = labels_train / f"{img_path.stem}.txt"
        shutil.move(str(img_path), str(images_val / img_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(labels_val / label_path.name))
        moved += 1

    n_train = len(annotated) - moved
    return n_train, moved


def auto_annotate(preview: int = 0, visualize: int = 0) -> dict:
    """Main auto-annotation pipeline."""
    print("=" * 60)
    print("Face + Bib Auto-Annotation Pipeline (Optimized)")
    print("=" * 60)
    print(f"  Dataset dir:  {DATASET_DIR}")
    print(f"  Images dir:   {ALL_IMAGES_DIR}")
    print(f"  OCR resize:   max {OCR_MAX_DIM}px")
    print(f"  Bib min digits: {BIB_MIN_DIGITS}")

    # Clean up previous label files from partial runs
    labels_dir = DATASET_DIR / "labels" / "train"
    existing_labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    if existing_labels:
        print(f"\n  Cleaning {len(existing_labels)} existing label files from previous run...")
        for lbl in existing_labels:
            lbl.unlink()

    # Load models
    face_app = load_face_detector()
    ocr = load_text_detector()

    # List images
    image_files = sorted(
        p for p in ALL_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    total = len(image_files)
    print(f"\n  Found {total} images")

    if preview > 0:
        image_files = image_files[:preview]
        print(f"  [PREVIEW MODE] Processing first {preview} only\n")
    else:
        print(f"  Processing all {total} images...\n")

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Visualization directory
    vis_dir = None
    if visualize > 0:
        vis_dir = DATASET_DIR / "annotation_preview"
        if vis_dir.exists():
            shutil.rmtree(vis_dir)
        vis_dir.mkdir(parents=True)

    # Stats
    stats = {
        "total": len(image_files),
        "faces_detected": 0,
        "bibs_detected": 0,
        "images_with_faces": 0,
        "images_with_bibs": 0,
        "images_with_both": 0,
        "images_with_neither": 0,
        "images_face_only": 0,
        "images_bib_only": 0,
        "total_annotations": 0,
        "skipped": 0,
    }

    start = time.time()

    for i, img_path in enumerate(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i+1}/{len(image_files)}] SKIP (unreadable): {img_path.name}")
            stats["skipped"] += 1
            continue

        h, w = img.shape[:2]

        # Detect faces (on original image - InsightFace handles resizing internally)
        faces = detect_faces(face_app, img)

        # Detect bibs (with internal resize for speed)
        bibs = detect_bibs_paddleocr(ocr, img)

        # Generate YOLO labels
        lines = []
        for face in faces:
            lines.append(to_yolo_format(face[0], face[1], face[2], face[3], w, h, FACE_CLASS_ID))
        for bib in bibs:
            lines.append(to_yolo_format(bib[0], bib[1], bib[2], bib[3], w, h, BIB_CLASS_ID))

        # Update stats
        n_faces = len(faces)
        n_bibs = len(bibs)
        stats["faces_detected"] += n_faces
        stats["bibs_detected"] += n_bibs
        has_face = n_faces > 0
        has_bib = n_bibs > 0

        if has_face:
            stats["images_with_faces"] += 1
        if has_bib:
            stats["images_with_bibs"] += 1
        if has_face and has_bib:
            stats["images_with_both"] += 1
        elif has_face:
            stats["images_face_only"] += 1
        elif has_bib:
            stats["images_bib_only"] += 1
        else:
            stats["images_with_neither"] += 1
        stats["total_annotations"] += len(lines)

        # Write label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        # Save visualization
        if vis_dir and i < visualize:
            visualize_annotations(img, faces, bibs, vis_dir / f"{img_path.stem}_annotated.jpg")

        # Progress every 25 images (or first 5, or last)
        if (i + 1) % 25 == 0 or (i + 1) == len(image_files) or (i + 1) <= 3:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(image_files) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(image_files)}] {img_path.name}: "
                f"{n_faces} face(s), {n_bibs} bib(s)  "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    elapsed = time.time() - start

    # Print summary
    print("\n" + "=" * 60)
    print("Auto-Annotation Summary")
    print("=" * 60)
    n = max(1, stats["total"])
    print(f"  Images processed:       {stats['total']}")
    print(f"  Skipped (unreadable):   {stats['skipped']}")
    print(f"  Total faces detected:   {stats['faces_detected']}")
    print(f"  Total bibs detected:    {stats['bibs_detected']}")
    print(f"  Images with faces:      {stats['images_with_faces']} ({stats['images_with_faces']/n*100:.1f}%)")
    print(f"  Images with bibs:       {stats['images_with_bibs']} ({stats['images_with_bibs']/n*100:.1f}%)")
    print(f"  Images with BOTH:       {stats['images_with_both']} ({stats['images_with_both']/n*100:.1f}%)")
    print(f"  Images face only:       {stats['images_face_only']} ({stats['images_face_only']/n*100:.1f}%)")
    print(f"  Images bib only:        {stats['images_bib_only']} ({stats['images_bib_only']/n*100:.1f}%)")
    print(f"  Images with NEITHER:    {stats['images_with_neither']} ({stats['images_with_neither']/n*100:.1f}%)")
    print(f"  Total annotations:      {stats['total_annotations']}")
    print(f"  Time elapsed:           {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Avg per image:          {elapsed/max(1,stats['total']-stats['skipped']):.1f}s")

    if vis_dir:
        print(f"\n  Annotated previews saved to: {vis_dir}")

    if preview > 0:
        print(f"\n[PREVIEW MODE] Only processed first {preview} images.")
        print("Run without --preview to process all images, then split train/val.")
    else:
        # Split into train/val
        print("\n" + "-" * 40)
        print("Splitting into train/val...")
        n_train, n_val = split_train_val(DATASET_DIR, VAL_RATIO, RANDOM_SEED)
        print(f"  Train: {n_train} images, Val: {n_val} images")
        print("\nReady for training: python scripts/train_face_bib_detector.py")

    print("\nDone!")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-annotate face+bib detection dataset"
    )
    parser.add_argument(
        "--preview", type=int, default=0,
        help="Only process first N images for preview (no train/val split)",
    )
    parser.add_argument(
        "--visualize", type=int, default=0,
        help="Save N annotated preview images for visual verification",
    )
    args = parser.parse_args()
    auto_annotate(preview=args.preview, visualize=args.visualize)
