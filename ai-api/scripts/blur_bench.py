"""Blur speed benchmark — time /blur/classify and /blur/classify/stream over HTTP.

blur_gate.py answers "is it accurate?"; this answers "is it fast enough where it
is deployed?" Point --url at localhost first, then at the Railway deployment,
and compare. Each single-image request is split into server time (the
response's processing_time_ms) and everything else (upload + network + queueing),
so you can see whether the CPU or the wire is the bottleneck.

Usage:
    python scripts/blur_bench.py <image_dir> \
        [--url http://localhost:8000] [--api-key sk_...] [--limit 200] \
        [--dim 1280] [--quality 90] [--concurrency 4] [--batch 200]

--dim 0 sends the original files untouched (what a phone would send without
downscaling). Requires a running ai-api with the blur classifier loaded.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from blur_gate import IMAGE_EXTS, classify_stream, resize_bytes


def pct(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, round(p / 100 * (len(s) - 1)))]


def summarize(label: str, xs: list[float]) -> None:
    print(
        f"  {label:26s} p50 {pct(xs, 50):8.1f} ms   "
        f"p95 {pct(xs, 95):8.1f} ms   max {max(xs):8.1f} ms"
    )


def classify_one(
    client: httpx.Client, base: str, headers: dict, data: bytes
) -> tuple[float, float] | None:
    """One /blur/classify call -> (total_ms, server_ms), or None on failure."""
    t0 = time.perf_counter()
    try:
        r = client.post(
            f"{base}/api/v1/blur/classify",
            headers=headers,
            files={"file": ("img.jpg", data, "image/jpeg")},
        )
    except httpx.HTTPError as e:
        print(f"  ! request failed ({e})", file=sys.stderr)
        return None
    total_ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        print(f"  ! {r.status_code} {r.text[:120]}", file=sys.stderr)
        return None
    return total_ms, r.json()["data"]["processing_time_ms"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Time blur classification over HTTP.")
    ap.add_argument("image_dir", type=Path, help="Folder of photos (searched recursively).")
    ap.add_argument("--url", default="http://localhost:8000", help="ai-api base URL.")
    ap.add_argument("--api-key", default="", help="X-API-Key (omit in DEBUG no-key mode).")
    ap.add_argument("--limit", type=int, default=200, help="Number of photos to use.")
    ap.add_argument(
        "--dim", type=int, default=1280, help="Client resize long edge (0 = send originals)."
    )
    ap.add_argument("--quality", type=int, default=90, help="Client JPEG quality when resizing.")
    ap.add_argument("--concurrency", type=int, default=4, help="Parallel single-image requests.")
    ap.add_argument("--batch", type=int, default=200, help="Images per stream request.")
    args = ap.parse_args()

    paths = sorted(p for p in args.image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)[
        : args.limit
    ]
    if not paths:
        print(f"No images found under {args.image_dir}", file=sys.stderr)
        return 2

    images = [
        p.read_bytes() if args.dim == 0 else resize_bytes(p, args.dim, args.quality) for p in paths
    ]
    avg_kb = sum(map(len, images)) / len(images) / 1024
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=300.0, pool=60.0)

    print(f"{len(images)} photos -> {args.url}")
    print(
        f"  upload size: {'originals' if args.dim == 0 else f'fit {args.dim}px q{args.quality}'}, "
        f"avg {avg_kb:.0f} KB\n"
    )

    with httpx.Client(
        timeout=timeout, limits=httpx.Limits(max_connections=args.concurrency)
    ) as client:
        # Cold start: the first calls pay for a sleeping container / model load.
        t0 = time.perf_counter()
        client.get(f"{args.url}/api/v1/health/ready", headers=headers)
        print(f"health check (cold):         {(time.perf_counter() - t0) * 1000:8.1f} ms")
        first = classify_one(client, args.url, headers, images[0])
        if first is None:
            return 1
        print(f"first classify (warm-up):    {first[0]:8.1f} ms  (server {first[1]:.1f} ms)\n")

        # Mode A — one request per photo, like a phone uploading as it shoots.
        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as pool:
            results = [
                r
                for r in pool.map(lambda d: classify_one(client, args.url, headers, d), images)
                if r
            ]
        wall = time.perf_counter() - t0
        print(f"A) /blur/classify  x{len(images)}, {args.concurrency} in parallel")
        if results:
            summarize("total per request", [t for t, _ in results])
            summarize("server (model + decode)", [s for _, s in results])
            summarize("network + upload + queue", [t - s for t, s in results])
        print(
            f"  throughput: {len(results) / wall:.1f} photos/s   "
            f"failed: {len(images) - len(results)}\n"
        )

        # Mode B — one streaming request per chunk, like the desktop batch sort.
        t0 = time.perf_counter()
        preds = classify_stream(client, args.url, headers, list(enumerate(images)), args.batch)
        wall = time.perf_counter() - t0
        ok = sum(1 for v in preds.values() if v)
        print(f"B) /blur/classify/stream  x{len(images)}, {args.batch} per request")
        print(
            f"  wall time: {wall:.2f} s   throughput: {ok / wall:.1f} photos/s   "
            f"failed: {len(images) - ok}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
