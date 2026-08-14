"""Blur stream gate — validate /blur/classify/stream against GROUND-TRUTH LABELS.

This replaces the earlier parity test, which compared the stream path against
POST /blur/classify and treated that as the reference. Measurement on the
labelled val set showed the reference is the *less* accurate of the two, so
parity was scoring the stream path as wrong for being right:

    640  cap (stream)  -> 97.45% accuracy, 1.4% of blurry photos missed
    1280 cap (single)  -> 96.59% accuracy, 2.7% of blurry photos missed

Cause: BlurClassifier._preprocess resizes the centre-crop to 224 with
INTER_LINEAR, which does not anti-alias. From 1280px that is a ~5.7x downsample,
and the resulting aliasing manufactures high-frequency edges that make blurry
photos look sharp. The stream path's 640 step uses INTER_AREA, so it gets one
proper anti-aliasing pass for free. See the 2026-08-14 journal entry.

So the gate now asks the question that matters — "does the production path
classify real photos correctly?" — instead of "does it agree with another path?"

Layout: pass a directory whose immediate subdirectories are class names, e.g.

    Training-Images/dataset/val/
        sharp/ defocused_blurred/ defocused_object_portrait/ motion_blurred/

Two criteria, both must hold:
  * overall accuracy            >= --min-accuracy
  * blurry-called-sharp rate    <= --max-miss-rate   (the dangerous direction:
                                                      a blurry photo escaping
                                                      culling and reaching runners)

Usage:
    python scripts/blur_gate.py Training-Images/dataset/val \
        [--url http://localhost:8000] [--api-key sk_...] \
        [--cand-dim 1280] [--cand-quality 90] [--batch 200] [--limit N] \
        [--min-accuracy 0.97] [--max-miss-rate 0.02]

Requires a running ai-api with the blur classifier loaded.

Caveat: resizing here uses Pillow LANCZOS while the desktop uses Sharp
(libvips Lanczos3). Resolution and JPEG quality dominate the class decision, not
the exact kernel. Run against the same ai-api build the desktop targets.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("This script needs httpx:  pip install httpx  (it's already an ai-api test dep).")

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("This script needs Pillow:  pip install pillow  (it's an ai-api runtime dep).")

# HEIC/HEIF support if pillow-heif is installed (optional).
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

CLASSES = ["sharp", "defocused_blurred", "defocused_object_portrait", "motion_blurred"]
BLUR_CLASSES = {c for c in CLASSES if c != "sharp"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".heic", ".heif"}


def resize_bytes(path: Path, max_dim: int, quality: int) -> bytes:
    """Resize fit-inside max_dim (no enlargement), EXIF-rotated, JPEG quality.

    Mirrors the desktop's prepareImageForUpload intent (rotate -> fit inside ->
    JPEG q). Raises on undecodable files (caller treats as a skip).
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def classify_stream(client: httpx.Client, base: str, headers: dict,
                    items: list[tuple[int, bytes]], batch: int) -> dict[int, str | None]:
    """Stream items (global_index, bytes) through /blur/classify/stream in chunks.

    Returns {global_index: predicted_class or None}. Results map back by the
    filename we upload (= the global index as a string), so chunking is safe.
    """
    out: dict[int, str | None] = {}
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        files = [("files", (str(gi), data, "image/jpeg")) for gi, data in chunk]
        try:
            with client.stream("POST", f"{base}/api/v1/blur/classify/stream",
                               headers=headers, files=files) as resp:
                if resp.status_code != 200:
                    resp.read()
                    print(f"  ! chunk @{start}: {resp.status_code} {resp.text[:120]}", file=sys.stderr)
                    for gi, _ in chunk:
                        out[gi] = None
                    continue
                seen = 0
                for line in resp.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("_summary"):
                        seen = obj.get("total", seen)
                        continue
                    gi = int(obj["filename"])
                    out[gi] = None if "error" in obj else obj.get("predicted_class")
                if seen != len(chunk):
                    # A truncated stream means the server died mid-batch; surface
                    # it rather than silently scoring a short run.
                    print(f"  ! chunk @{start}: stream ended after {seen}/{len(chunk)}",
                          file=sys.stderr)
        except httpx.HTTPError as e:
            print(f"  ! chunk @{start}: request failed ({e})", file=sys.stderr)
            for gi, _ in chunk:
                out[gi] = None
    return out


def collect_labelled(root: Path, limit: int) -> list[tuple[Path, str]]:
    """Find (image_path, label) pairs from immediate class subdirectories."""
    pairs: list[tuple[Path, str]] = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if cls_dir.name not in CLASSES:
            print(f"  (skipping unrecognised class dir: {cls_dir.name})", file=sys.stderr)
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                pairs.append((p, cls_dir.name))
    if limit:
        # Stride-sample so a --limit stays balanced across classes instead of
        # taking every image from whichever class sorts first.
        step = max(1, len(pairs) // limit)
        pairs = pairs[::step][:limit]
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Blur stream gate: /blur/classify/stream vs ground-truth labels.")
    ap.add_argument("labelled_dir", type=Path,
                    help="Directory whose subdirectories are class names.")
    ap.add_argument("--url", default="http://localhost:8000", help="ai-api base URL.")
    ap.add_argument("--api-key", default="", help="X-API-Key (omit in DEBUG no-key mode).")
    ap.add_argument("--cand-dim", type=int, default=1280, help="Client resize long edge.")
    ap.add_argument("--cand-quality", type=int, default=90, help="Client JPEG quality.")
    ap.add_argument("--batch", type=int, default=200, help="Images per stream request.")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of images (0 = all).")
    ap.add_argument("--min-accuracy", type=float, default=0.97, help="Gate: min overall accuracy.")
    ap.add_argument("--max-miss-rate", type=float, default=0.02,
                    help="Gate: max fraction of blurry photos predicted sharp.")
    args = ap.parse_args()

    if not args.labelled_dir.is_dir():
        print(f"Not a directory: {args.labelled_dir}", file=sys.stderr)
        return 2

    pairs = collect_labelled(args.labelled_dir, args.limit)
    if not pairs:
        print(f"No labelled images found under {args.labelled_dir}", file=sys.stderr)
        return 2

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=300.0, pool=10.0)

    print(f"Testing {len(pairs)} labelled images against {args.url}")
    print(f"  candidate: fit {args.cand_dim}px q{args.cand_quality} -> /blur/classify/stream\n")

    items: list[tuple[int, bytes]] = []
    labels: dict[int, str] = {}
    resize_failures = 0
    for i, (path, label) in enumerate(pairs):
        try:
            items.append((i, resize_bytes(path, args.cand_dim, args.cand_quality)))
            labels[i] = label
        except Exception as e:
            resize_failures += 1
            print(f"  ! {path.name}: resize failed ({e})", file=sys.stderr)

    with httpx.Client(timeout=timeout) as client:
        preds = classify_stream(client, args.url, headers, items, args.batch)

    scored = [i for i in labels if preds.get(i)]
    n = len(scored)
    if not n:
        print("No images were successfully classified.", file=sys.stderr)
        return 2

    correct = sum(1 for i in scored if preds[i] == labels[i])
    n_blur = sum(1 for i in scored if labels[i] in BLUR_CLASSES)
    n_sharp = sum(1 for i in scored if labels[i] == "sharp")
    missed = [i for i in scored if labels[i] in BLUR_CLASSES and preds[i] == "sharp"]
    overcull = [i for i in scored if labels[i] == "sharp" and preds[i] in BLUR_CLASSES]
    confusion = Counter((labels[i], preds[i]) for i in scored if labels[i] != preds[i])
    api_failures = len(labels) - n

    accuracy = correct / n
    miss_rate = (len(missed) / n_blur) if n_blur else 0.0

    print("\n" + "=" * 68)
    print("BLUR STREAM GATE — vs GROUND-TRUTH LABELS")
    print("=" * 68)
    print(f"images scored:                   {n}  (sharp={n_sharp}, blurry={n_blur})")
    print(f"accuracy:                        {correct}/{n} = {accuracy:.4f}")
    print(f"blurry called sharp (dangerous): {len(missed)}/{n_blur} = {miss_rate:.4f}")
    print(f"sharp called blurry (over-cull): {len(overcull)}/{max(n_sharp, 1)}")
    if resize_failures:
        print(f"resize failures (skipped):       {resize_failures}")
    if api_failures:
        print(f"API failures (unscored):         {api_failures}")

    if confusion:
        print("\nmisclassifications (label -> predicted : count):")
        for (lab, pred), c in confusion.most_common():
            mark = "  <-- DANGEROUS" if lab in BLUR_CLASSES and pred == "sharp" else ""
            print(f"  {lab:28s} -> {pred:28s} : {c}{mark}")

    if missed:
        print("\nblurry photos the stream called sharp:")
        for i in missed[:25]:
            print(f"  - {pairs[i][0].name}  ({labels[i]})")
        if len(missed) > 25:
            print(f"  … and {len(missed) - 25} more")

    passed = accuracy >= args.min_accuracy and miss_rate <= args.max_miss_rate
    print("\n" + ("PASS" if passed else "FAIL")
          + f"  (gate: accuracy >= {args.min_accuracy}, miss rate <= {args.max_miss_rate})")
    print("=" * 68)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
