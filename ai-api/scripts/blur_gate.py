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

Layout: pass a directory whose immediate subdirectories are class names, or
event folders that each contain class folders (per-event results are reported):

    holdout/<event-id>/{sharp,defocused_blurred,defocused_object_portrait,motion_blurred}/

Quality criteria, plus complete input coverage, must hold:
  * binary (sharp vs any blur) accuracy >= --min-accuracy
  * blurry-called-sharp rate    <= --max-miss-rate   (the dangerous direction:
                                                      a blurry photo escaping
                                                      culling and reaching runners)
  * sharp-called-blurry rate    <= --max-overcull-rate

Usage:
    python scripts/blur_gate.py Training-Images/dataset/val \
        [--url http://localhost:8000] [--api-key-file untracked-key.txt] \
        [--cand-dim 1280] [--cand-quality 90] [--batch 200] [--limit N] \
        [--min-accuracy 0.97] [--max-miss-rate 0.02] [--max-overcull-rate 0]

Set AI_API_KEY in the environment instead of --api-key-file when preferred.
--limit requires --diagnostic-subset. Diagnostic runs exit 2, never release PASS.

Requires a running ai-api with the blur classifier loaded.

Caveat: resizing here uses Pillow LANCZOS while the desktop uses Sharp
(libvips Lanczos3). Resolution and JPEG quality dominate the class decision, not
the exact kernel. Run against the same ai-api build the desktop targets.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("This script needs httpx:  pip install httpx  (it's already an ai-api test dep).")

try:
    from PIL import Image, ImageOps
    from PIL import __version__ as pillow_version
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


def _valid_probability(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval: how far a rate measured on n images can be trusted."""
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


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


PHOTOMETRIC_VARIANTS = (("brighter", 0.6), ("darker", 1.6), ("grayscale", None))


def photometric_variants(jpeg: bytes, quality: int) -> list[bytes]:
    """Same pixels with a different tone or no colour: a blur decision must not move.

    The 2026-08-14 model flagged sharp night photos 9% -> 56% as they were only
    brightened, and grayscale cut its flags 32% -> 9% (2026-09-29 investigation).
    """
    out = []
    with Image.open(io.BytesIO(jpeg)) as im:
        rgb = im.convert("RGB")
        for _, gamma in PHOTOMETRIC_VARIANTS:
            if gamma is None:
                variant = ImageOps.grayscale(rgb).convert("RGB")
            else:
                variant = rgb.point([round(255 * (v / 255) ** gamma) for v in range(256)] * 3)
            buf = io.BytesIO()
            variant.save(buf, format="JPEG", quality=quality)
            out.append(buf.getvalue())
    return out


def classify_stream(client: httpx.Client, base: str, headers: dict,
                    items: list[tuple[int, bytes]], batch: int,
                    issues: list[str] | None = None,
                    probabilities_out: dict[int, dict] | None = None) -> dict[int, str | None]:
    """Stream items (global_index, bytes) through /blur/classify/stream in chunks.

    Returns {global_index: predicted_class or None}. Results map back by the
    filename we upload (= the global index as a string), so chunking is safe.
    """
    if issues is None:
        issues = []
    out: dict[int, str | None] = {i: None for i, _ in items}
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        files = [("files", (str(gi), data, "image/jpeg")) for gi, data in chunk]
        try:
            with client.stream("POST", f"{base}/api/v1/blur/classify/stream",
                               headers=headers, files=files) as resp:
                if resp.status_code != 200:
                    issues.append(f"chunk @{start}: HTTP {resp.status_code}")
                    continue
                seen: set[int] = set()
                successes = errors = summaries = 0
                summary = None
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except (ValueError, TypeError):
                        issues.append(f"chunk @{start}: malformed NDJSON")
                        continue
                    if not isinstance(obj, dict):
                        issues.append(f"chunk @{start}: non-object NDJSON row")
                        continue
                    if "_summary" in obj:
                        summaries += 1
                        summary = obj
                        if obj["_summary"] is not True or summaries > 1:
                            issues.append(f"chunk @{start}: invalid or duplicate summary")
                        continue
                    if summaries:
                        issues.append(f"chunk @{start}: result after summary")
                    index = obj.get("index")
                    if type(index) is not int or not 0 <= index < len(chunk):
                        issues.append(f"chunk @{start}: unknown result index")
                        continue
                    if index in seen:
                        issues.append(f"chunk @{start}: duplicate result index {index}")
                        continue
                    seen.add(index)
                    gi = chunk[index][0]
                    if obj.get("filename") != str(gi):
                        issues.append(f"chunk @{start}: wrong filename for index {index}")
                        continue
                    if "error" in obj:
                        errors += 1
                        issues.append(f"chunk @{start}: file {index} error")
                        continue
                    predicted = obj.get("predicted_class")
                    confidence = obj.get("confidence")
                    probabilities = obj.get("probabilities")
                    if (predicted not in CLASSES or not _valid_probability(confidence)
                            or not isinstance(probabilities, dict)
                            or set(probabilities) != set(CLASSES)
                            or not all(_valid_probability(v) for v in probabilities.values())):
                        issues.append(f"chunk @{start}: invalid prediction for index {index}")
                        continue
                    successes += 1
                    out[gi] = predicted
                    if probabilities_out is not None:
                        probabilities_out[gi] = probabilities
                if len(seen) != len(chunk):
                    issues.append(f"chunk @{start}: {len(seen)}/{len(chunk)} result rows")
                if summaries != 1 or summary is None:
                    issues.append(f"chunk @{start}: missing summary")
                elif (type(summary.get("total")) is not int
                      or summary["total"] != len(chunk)
                      or type(summary.get("successful")) is not int
                      or summary["successful"] != successes
                      or type(summary.get("errors")) is not int
                      or summary["errors"] != errors
                      or summary.get("complete") is not True
                      or ("processing_time_ms" in summary
                          and (type(summary["processing_time_ms"]) not in (int, float)
                               or not math.isfinite(summary["processing_time_ms"])
                               or summary["processing_time_ms"] < 0))):
                    issues.append(f"chunk @{start}: inconsistent summary")
        except httpx.HTTPError as e:
            issues.append(f"chunk @{start}: transport error ({type(e).__name__})")
            for gi, _ in chunk:
                out[gi] = None
    return out


SWEEP_THRESHOLDS = (0.30, 0.45, 0.65, 0.80, 0.90, 0.95, 0.99)


def threshold_sweep(labels: dict[int, str], probabilities: dict[int, dict]) -> dict:
    """Over-cull/miss per flag threshold, mirroring the desktop rules (calibration only).

    max_class: best single blur class >= t (desktop today).
    total:     1 - P(sharp) >= t (candidate rule, plan decision D7).
    """
    sweep = {}
    for t in SWEEP_THRESHOLDS:
        row = {}
        for rule in ("max_class", "total"):
            over = miss = n_sharp = n_blur = 0
            for i, label in labels.items():
                p = probabilities.get(i)
                if p is None:
                    continue
                score = (max(p[c] for c in BLUR_CLASSES) if rule == "max_class"
                         else 1 - p["sharp"])
                if label == "sharp":
                    n_sharp += 1
                    over += score >= t
                else:
                    n_blur += 1
                    miss += score < t
            row[rule] = {"sharp_overcull": over / n_sharp if n_sharp else None,
                         "blur_miss": miss / n_blur if n_blur else None}
        sweep[f"{t:.2f}"] = row
    return sweep


def classify_offline(model_file: Path, labels_file: Path, items: list[tuple[int, bytes]],
                     batch: int, issues: list[str],
                     probabilities_out: dict[int, dict]) -> dict[int, str | None]:
    """Production decode + BlurClassifier in-process, to score candidates without a server.

    Diagnostic only: a release must still pass through the deployed HTTP path.
    """
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.api.v1.blur import _decode_bgr_for_classify
    from src.config import get_settings
    from src.ml.blur.classifier import BlurClassifier

    classifier = BlurClassifier(model_path=str(model_file), class_names_path=str(labels_file))
    out: dict[int, str | None] = {gi: None for gi, _ in items}
    if classifier.session is None:
        issues.append("offline model could not be loaded")
        return out
    dim = get_settings().BLUR_CLASSIFY_DECODE_DIM
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        images = []
        for _, data in chunk:
            try:
                images.append(_decode_bgr_for_classify(data, dim))
            except Exception:
                images.append(None)
        for (gi, _), result in zip(chunk, classifier.classify_batch(images), strict=True):
            if result is None:
                issues.append(f"offline: input {gi} failed")
                continue
            out[gi] = result["predicted_class"]
            probabilities_out[gi] = result["probabilities"]
    return out


def collect_labelled(root: Path, limit: int) -> list[tuple[Path, str, str]]:
    """Find (image_path, label, event) from class folders.

    root/<class>/image is one event named after root; root/<event>/<class>/image
    keeps events apart so over-cull on one new event cannot hide in a pooled rate.
    """
    pairs: list[tuple[Path, str, str]] = []

    def add(class_dir: Path, event: str) -> None:
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                pairs.append((p, class_dir.name, event))

    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if child.name in CLASSES:
            add(child, root.name)
        elif any((child / c).is_dir() for c in CLASSES):
            for c in CLASSES:
                if (child / c).is_dir():
                    add(child / c, child.name)
        else:
            print(f"  (skipping unrecognised dir: {child.name})", file=sys.stderr)
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
    ap.add_argument("--api-key-file", type=Path,
                    help="Untracked X-API-Key file; otherwise use AI_API_KEY environment variable.")
    ap.add_argument("--model-file", type=Path,
                    default=Path("models/blur_classifier/blur_classifier.onnx"))
    ap.add_argument("--model-manifest", type=Path, default=Path("models/manifest.json"))
    ap.add_argument("--report-json", type=Path, default=Path("blur-gate-report.json"))
    ap.add_argument("--cand-dim", type=int, default=1280, help="Client resize long edge.")
    ap.add_argument("--cand-quality", type=int, default=90, help="Client JPEG quality.")
    ap.add_argument("--batch", type=int, default=200, help="Images per stream request.")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of images (0 = all).")
    ap.add_argument("--diagnostic-subset", action="store_true",
                    help="Allow --limit or partial class coverage; output is not a release PASS.")
    ap.add_argument("--min-accuracy", type=float, default=0.97, help="Gate: min overall accuracy.")
    ap.add_argument("--max-miss-rate", type=float, default=0.02,
                    help="Gate: max fraction of blurry photos predicted sharp.")
    ap.add_argument("--max-overcull-rate", type=float, default=0.0,
                    help="Gate: max fraction of sharp photos predicted blurry.")
    ap.add_argument("--min-sharp", type=int, default=0,
                    help="Release minimum of sharp inputs (owner-approved; 0 = none).")
    ap.add_argument("--min-blur", type=int, default=0,
                    help="Release minimum of blurry inputs, all blur classes together.")
    ap.add_argument("--min-events", type=int, default=0,
                    help="Release minimum of distinct events (nested layout).")
    ap.add_argument("--max-photometric-flip-rate", type=float, default=None,
                    help="Gate: max fraction of inputs whose sharp/blurry decision changes "
                         "under brighter/darker/grayscale copies (setting it runs the check).")
    ap.add_argument("--offline", action="store_true",
                    help="Score --model-file in-process with the production decode; never PASS.")
    ap.add_argument("--labels-file", type=Path,
                    default=Path("models/blur_classifier/class_names.json"))
    args = ap.parse_args()

    if (args.batch < 1 or args.limit < 0 or args.cand_dim < 1
            or not 1 <= args.cand_quality <= 100
            or min(args.min_sharp, args.min_blur, args.min_events) < 0
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in (
                args.min_accuracy, args.max_miss_rate, args.max_overcull_rate,
            )) or (args.limit and not args.diagnostic_subset)
            or (args.max_photometric_flip_rate is not None
                and (not math.isfinite(args.max_photometric_flip_rate)
                     or not 0 <= args.max_photometric_flip_rate <= 1))):
        ap.error("invalid gate settings; --limit requires --diagnostic-subset")
    if not args.labelled_dir.is_dir():
        print(f"Not a directory: {args.labelled_dir}", file=sys.stderr)
        return 2

    pairs = collect_labelled(args.labelled_dir, args.limit)
    if not pairs:
        print(f"No labelled images found under {args.labelled_dir}", file=sys.stderr)
        return 2

    issues: list[str] = []
    model = {"version": None, "sha256": None}
    try:
        with args.model_file.open("rb") as f:
            model["sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        artifact = json.loads(args.model_manifest.read_text(encoding="utf-8"))["models"][
            "blur_classifier"
        ]
        model["version"] = artifact["artifact_version"]
        if model["sha256"] != artifact["sha256"] and not args.offline:
            issues.append("model digest differs from manifest")
    except (OSError, ValueError, KeyError, TypeError):
        if not args.offline or model["sha256"] is None:
            issues.append("model file or manifest unavailable/invalid")

    try:
        key = (args.api_key_file.read_text(encoding="utf-8").strip()
               if args.api_key_file else os.environ.get("AI_API_KEY", "").strip())
    except OSError:
        key = ""
        issues.append("API key file unavailable")
    headers = {"X-API-Key": key} if key else {}
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=300.0, pool=10.0)

    print(f"Testing {len(pairs)} labelled images against {args.url}")
    print(f"  candidate: fit {args.cand_dim}px q{args.cand_quality} -> /blur/classify/stream\n")

    items: list[tuple[int, bytes]] = []
    labels: dict[int, str] = {}
    dataset_manifest = []
    for i, (path, label, event) in enumerate(pairs):
        entry = {"path": path.relative_to(args.labelled_dir).as_posix(),
                 "label": label, "event": event, "sha256": None}
        try:
            with path.open("rb") as f:
                entry["sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        except OSError:
            issues.append(f"input {i}: cannot hash source image")
        dataset_manifest.append(entry)
        try:
            items.append((i, resize_bytes(path, args.cand_dim, args.cand_quality)))
            labels[i] = label
        except Exception:
            issues.append(f"input {i}: preparation failed")

    variant_of: dict[int, int] = {}
    if args.max_photometric_flip_rate is not None:
        next_index = len(pairs)
        for i, data in list(items):
            try:
                variants = photometric_variants(data, args.cand_quality)
            except Exception:
                issues.append(f"input {i}: photometric preparation failed")
                continue
            for variant in variants:
                items.append((next_index, variant))
                variant_of[next_index] = i
                next_index += 1

    probabilities: dict[int, dict] = {}
    if args.offline:
        preds = classify_offline(args.model_file, args.labels_file, items, args.batch,
                                 issues, probabilities)
    else:
        with httpx.Client(timeout=timeout) as client:
            preds = classify_stream(client, args.url, headers, items, args.batch, issues,
                                    probabilities)

    scored = [i for i in labels if preds.get(i) in CLASSES]
    n = len(scored)
    correct = sum(1 for i in scored if preds[i] == labels[i])
    is_blur = [label in BLUR_CLASSES for _, label, _ in pairs]
    binary_correct = sum(1 for i in scored if is_blur[i] == (preds[i] in BLUR_CLASSES))
    n_blur = sum(is_blur)
    n_sharp = len(pairs) - n_blur
    missed = [i for i in scored if is_blur[i] and preds[i] == "sharp"]
    overcull = [i for i in scored if not is_blur[i] and preds[i] in BLUR_CLASSES]
    confusion = Counter((labels[i], preds[i]) for i in scored)
    failed = len(pairs) - n
    events: dict[str, dict[str, int]] = {}
    for i, (_, _, event) in enumerate(pairs):
        row = events.setdefault(event, {"sharp": 0, "blur": 0,
                                        "sharp_overcull": 0, "blur_miss": 0})
        row["blur" if is_blur[i] else "sharp"] += 1
    for i in overcull:
        events[pairs[i][2]]["sharp_overcull"] += 1
    for i in missed:
        events[pairs[i][2]]["blur_miss"] += 1

    accuracy = correct / len(pairs)
    binary_accuracy = binary_correct / len(pairs)
    miss_rate = (len(missed) / n_blur) if n_blur else None
    overcull_rate = (len(overcull) / n_sharp) if n_sharp else None
    if not args.diagnostic_subset:
        if not n_sharp or not n_blur:
            issues.append("missing coverage: need both sharp and blurry inputs")
        if (n_sharp < args.min_sharp or n_blur < args.min_blur
                or len(events) < args.min_events):
            issues.append(f"below release minimums: {n_sharp} sharp, {n_blur} blurry, "
                          f"{len(events)} events")
    if failed:
        issues.append(f"{failed} input(s) without valid predictions")

    photometric = None
    if args.max_photometric_flip_rate is not None:
        flipped: dict[int, bool] = {}
        for j, i in variant_of.items():
            if preds.get(i) in CLASSES and preds.get(j) in CLASSES:
                moved = (preds[j] in BLUR_CLASSES) != (preds[i] in BLUR_CLASSES)
                flipped[i] = flipped.get(i, False) or moved
        checked, flips = len(flipped), sum(flipped.values())
        photometric = {"variants": [name for name, _ in PHOTOMETRIC_VARIANTS],
                       "checked": checked, "flipped": flips,
                       "flip_rate": flips / checked if checked else None,
                       "interval_95": wilson(flips, checked)}

    quality_passed = (binary_accuracy >= args.min_accuracy
                      and (miss_rate is not None and miss_rate <= args.max_miss_rate
                           or args.diagnostic_subset and miss_rate is None)
                      and (overcull_rate is not None and overcull_rate <= args.max_overcull_rate
                           or args.diagnostic_subset and overcull_rate is None)
                      and (photometric is None
                           or photometric["flip_rate"] is not None
                           and photometric["flip_rate"] <= args.max_photometric_flip_rate))
    status = "FAIL" if issues or not quality_passed else (
        "DIAGNOSTIC" if args.diagnostic_subset or args.offline else "PASS"
    )
    report = {
        "status": status,
        "counts": {"attempted": len(pairs), "scored": n, "failed": failed},
        "rates": {"accuracy": accuracy, "binary_accuracy": binary_accuracy,
                  "blur_miss": miss_rate, "sharp_overcull": overcull_rate},
        "intervals_95": {"blur_miss": wilson(len(missed), n_blur),
                         "sharp_overcull": wilson(len(overcull), n_sharp)},
        "events": events,
        "photometric": photometric,
        "threshold_sweep": threshold_sweep(labels, probabilities),
        "coverage": {label: sum(pair[1] == label for pair in pairs) for label in CLASSES},
        "confusion": {label: {pred: confusion[(label, pred)] for pred in CLASSES}
                      for label in CLASSES},
        "issues": issues,
        "model": model,
        "preprocessing": {"client_max_dimension": args.cand_dim,
                          "client_jpeg_quality": args.cand_quality,
                          "endpoint": "in-process" if args.offline
                          else "/api/v1/blur/classify/stream"},
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_sha256": hashlib.sha256(json.dumps(
            dataset_manifest, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "runtime_dependencies": {"python": platform.python_version(),
                                 "httpx": httpx.__version__, "pillow": pillow_version},
        "gate_arguments": {"url": args.url, "batch": args.batch, "limit": args.limit,
                           "diagnostic_subset": args.diagnostic_subset,
                           "min_accuracy": args.min_accuracy,
                           "max_miss_rate": args.max_miss_rate,
                           "max_overcull_rate": args.max_overcull_rate,
                           "min_sharp": args.min_sharp, "min_blur": args.min_blur,
                           "min_events": args.min_events, "accuracy_basis": "binary",
                           "max_photometric_flip_rate": args.max_photometric_flip_rate,
                           "offline": args.offline},
    }
    try:
        args.report_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")
    except OSError as e:
        print(f"Cannot write report: {type(e).__name__}", file=sys.stderr)
        return 2

    print("\n" + "=" * 68)
    print("BLUR STREAM GATE — vs GROUND-TRUTH LABELS")
    print("=" * 68)
    print(f"attempted/scored/failed:         {len(pairs)}/{n}/{failed}")
    print(f"binary accuracy (gated):         {binary_correct}/{len(pairs)} = {binary_accuracy:.4f}")
    print(f"4-class accuracy (reported):     {correct}/{len(pairs)} = {accuracy:.4f}")
    print(f"blurry called sharp (dangerous): {len(missed)}/{n_blur} = {miss_rate}")
    print(f"sharp called blurry (over-cull): {len(overcull)}/{n_sharp} = {overcull_rate}")
    if photometric:
        print(f"tone/colour-only flips:          {photometric['flipped']}/"
              f"{photometric['checked']} = {photometric['flip_rate']}")
    print("confusion (label -> predicted):")
    for label in CLASSES:
        print(f"  {label:28s} " + " ".join(str(confusion[(label, pred)]) for pred in CLASSES))
    print("per event (sharp over-cull | blur miss):")
    for event, row in sorted(events.items()):
        print(f"  {event[:30]:30s} {row['sharp_overcull']}/{row['sharp']} | "
              f"{row['blur_miss']}/{row['blur']}")
    if issues:
        print(f"data/stream errors:             {len(issues)}")
        for issue in issues[:10]:
            print(f"  ! {issue}")

    if missed:
        print("\nblurry photos the stream called sharp:")
        for i in missed[:25]:
            print(f"  - {pairs[i][0].name}  ({labels[i]})")
        if len(missed) > 25:
            print(f"  … and {len(missed) - 25} more")

    print("\n" + status
          + f"  (binary accuracy >= {args.min_accuracy}, miss <= {args.max_miss_rate},"
          + f" over-cull <= {args.max_overcull_rate})")
    print(f"report: {args.report_json}")
    print("=" * 68)
    return {"PASS": 0, "FAIL": 1, "DIAGNOSTIC": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
