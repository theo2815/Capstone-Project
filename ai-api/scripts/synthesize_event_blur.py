"""Paired synthetic blur from sharp event photos (training data only, never holdout).

The 2026-08-14 model learned shoot appearance (brightness, colour, scene) instead of
blur because every class came from different shoots. Here each sharp event frame
yields one defocused and one motion-blurred copy of the SAME frame, so scene, light
and colour are identical across classes and only blur separates them. Copies keep
the source's `<event-id>__` prefix, so prepare_blur_dataset.shoot_group() places
them in the same split as their source.

Usage (from ai-api/):  python scripts/synthesize_event_blur.py [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from scripts.prepare_blur_dataset import EVENT_PREFIX, IMAGE_EXTENSIONS
except ImportError:  # run as a plain script from ai-api/scripts
    from prepare_blur_dataset import EVENT_PREFIX, IMAGE_EXTENSIONS

TRAINING_DIR = Path(__file__).resolve().parent.parent / "Training-Images"
SOURCE_DIR = "Sharp event photos"
DEFOCUS_DIR = "Synthetic event defocus"
MOTION_DIR = "Synthetic event motion"
PROVENANCE = "synthetic-event-provenance.json"
WORK_LONG_EDGE = 2560          # 2x the desktop's 1280 upload, so kernels stay exact after downscale
DISPLAY_EDGE = 2048            # copies look blurry even at a 2048-px view, so surely at 100% (D6)
DEFOCUS_RADIUS = (2.0, 9.0)    # px at a 2048-px long edge: clearly soft, never borderline
MOTION_LENGTH = (5.0, 24.0)    # px streak at a 2048-px long edge; short streaks cover subtle motion

_LEVELS = np.arange(256) / 255.0
_TO_LINEAR = np.where(_LEVELS <= 0.04045, _LEVELS / 12.92,
                      ((_LEVELS + 0.055) / 1.055) ** 2.4).astype(np.float32)


def disk_kernel(radius: float) -> np.ndarray:
    """Anti-aliased pillbox: the point spread of an out-of-focus lens."""
    r = max(radius, 0.5)
    half = int(np.ceil(r))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    kernel = np.clip(r + 0.5 - np.hypot(xx, yy), 0.0, 1.0).astype(np.float32)
    return kernel / kernel.sum()


def line_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Straight streak `length` px long at `angle_deg`: camera shake or subject motion."""
    size = int(np.ceil(length)) | 1
    kernel = np.zeros((size, size), np.float32)
    c = size // 2
    dx = np.cos(np.radians(angle_deg)) * (length - 1) / 2
    dy = np.sin(np.radians(angle_deg)) * (length - 1) / 2
    cv2.line(kernel, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)),
             1.0, 1, cv2.LINE_AA)
    return kernel / kernel.sum()


def noise_sigma(bgr: np.ndarray) -> float:
    """Per-channel sensor-noise estimate (Immerkaer 1996), averaged over channels."""
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = bgr.shape[:2]
    total = sum(float(np.abs(cv2.filter2D(bgr[:, :, c].astype(np.float32), -1, mask))
                      [1:-1, 1:-1].sum()) for c in range(3))
    return float(np.sqrt(np.pi / 2) * total / (3 * 6 * (w - 2) * (h - 2)))


def apply_blur(bgr: np.ndarray, kernel: np.ndarray, sigma: float,
               rng: np.random.Generator) -> np.ndarray:
    """Blur in linear light (how a lens mixes light), then restore the removed noise
    so that 'clean' never becomes a shortcut for 'blurry'."""
    linear = np.clip(cv2.filter2D(_TO_LINEAR[bgr], -1, kernel, borderType=cv2.BORDER_REFLECT),
                     0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.power(linear, 1 / 2.4) - 0.055) * 255
    return np.clip(srgb + rng.normal(0.0, sigma, bgr.shape), 0, 255).astype(np.uint8)


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def synthesize(root: Path = TRAINING_DIR, seed: int = 42) -> list[dict]:
    """Write one defocus and one motion copy per sharp event frame; return provenance."""
    rng = random.Random(seed)
    noise = np.random.default_rng(seed)
    sources = sorted(p for p in (root / SOURCE_DIR).iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    for src in sources:   # check every name before deleting the previous copies
        if not EVENT_PREFIX.match(src.name):
            raise ValueError(f"{src.name}: event photos must be named "
                             "'<event-id>__<original name>'")
    for folder in (DEFOCUS_DIR, MOTION_DIR):     # generated only; rebuilt every run
        (root / folder).mkdir(parents=True, exist_ok=True)
        for old in (root / folder).glob("*.jpg"):
            old.unlink()
    records = []
    for src in sources:
        event = EVENT_PREFIX.match(src.name)
        with Image.open(src) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        scale = min(1.0, WORK_LONG_EDGE / max(rgb.size))
        if scale < 1:
            rgb = rgb.resize((round(rgb.width * scale), round(rgb.height * scale)), Image.LANCZOS)
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        sigma = noise_sigma(bgr)
        digest = _sha256(src)
        k = max(rgb.size) / DISPLAY_EDGE   # working pixels per 2048-px display pixel
        radius = rng.uniform(*DEFOCUS_RADIUS)
        length, angle = rng.uniform(*MOTION_LENGTH), rng.uniform(0.0, 180.0)
        for folder, kind, kernel, params in (
            (DEFOCUS_DIR, "defocus", disk_kernel(radius * k), {"radius_px_at_2048": radius}),
            (MOTION_DIR, "motion", line_kernel(length * k, angle),
             {"length_px_at_2048": length, "angle_deg": angle}),
        ):
            out = root / folder / f"{event.group(1)}__{kind}-{digest[:16]}.jpg"
            blurred = cv2.cvtColor(apply_blur(bgr, kernel, sigma, noise), cv2.COLOR_BGR2RGB)
            Image.fromarray(blurred).save(out, quality=95)
            records.append({"source": src.name, "source_sha256": digest, "kind": kind,
                            "output": out.relative_to(root).as_posix(),
                            "output_sha256": _sha256(out), "noise_sigma": sigma, **params})
    (root / PROVENANCE).write_text(json.dumps(records, indent=1), encoding="utf-8")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=TRAINING_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Wrote {len(synthesize(args.root, args.seed))} synthetic images")
