"""Benchmark: C++ extension vs Python/NumPy implementations.

Run: python benchmarks/bench_cpp_vs_python.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

# Ensure the project root is on sys.path so _eventai_cpp can be found.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np

try:
    import _eventai_cpp as cpp

    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("WARNING: C++ extension not found. Only Python benchmarks will run.\n")


def bench(name: str, fn: Callable, repeats: int = 100) -> float:
    """Run fn() `repeats` times and return mean time in ms."""
    # Warmup
    for _ in range(min(5, repeats)):
        fn()

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    elapsed = (time.perf_counter() - start) / repeats * 1000
    print(f"  {name:40s}  {elapsed:8.3f} ms  (avg of {repeats})")
    return elapsed


def bench_laplacian():
    """Benchmark Laplacian variance."""
    print("\n=== Laplacian Variance (256x256 grayscale) ===")
    gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    # Python (OpenCV)
    def py_laplacian():
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    t_py = bench("Python (cv2.Laplacian + .var())", py_laplacian, 500)

    if HAS_CPP:
        t_cpp = bench("C++ (laplacian_variance)", lambda: cpp.laplacian_variance(gray), 500)
        print(f"  Speedup: {t_py / t_cpp:.1f}x")


def bench_fft_hf_ratio():
    """Benchmark FFT HF ratio."""
    print("\n=== FFT HF Ratio (256x256 grayscale) ===")
    gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    def py_fft():
        f = np.fft.fft2(gray)
        fs = np.fft.fftshift(f)
        mag = np.abs(fs)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 8
        mask = np.ones_like(mag)
        mask[cy - r : cy + r, cx - r : cx + r] = 0
        total = np.sum(mag)
        return float(np.sum(mag * mask) / total) if total > 0 else 0.0

    t_py = bench("Python (np.fft.fft2 + mask)", py_fft, 200)

    if HAS_CPP:
        t_cpp = bench("C++ (fft_hf_ratio)", lambda: cpp.fft_hf_ratio(gray), 200)
        print(f"  Speedup: {t_py / t_cpp:.1f}x")


def bench_cosine_topk():
    """Benchmark batch cosine top-K for various database sizes."""
    for N in [100, 1000, 10000, 100000]:
        print(f"\n=== Batch Cosine Top-K (N={N:,}, D=512, K=10) ===")
        query = np.random.randn(512).astype(np.float32)
        query /= np.linalg.norm(query)
        database = np.random.randn(N, 512).astype(np.float32)
        database /= np.linalg.norm(database, axis=1, keepdims=True)

        threshold = 0.0
        top_k = 10
        reps = max(10, 1000 // max(1, N // 1000))

        def py_topk():
            sims = database @ query
            mask = sims >= threshold
            valid = np.where(mask)[0]
            if len(valid) == 0:
                return []
            scores = sims[valid]
            order = np.argsort(scores)[::-1][:top_k]
            return [(int(valid[i]), float(scores[i])) for i in order]

        t_py = bench("Python (NumPy matmul + argsort)", py_topk, reps)

        if HAS_CPP:
            t_cpp = bench(
                "C++ (batch_cosine_topk)",
                lambda: cpp.batch_cosine_topk(query, database, threshold, top_k),
                reps,
            )
            print(f"  Speedup: {t_py / t_cpp:.1f}x")


def bench_batch_blur():
    """Benchmark batch blur metrics."""
    print("\n=== Batch Blur Metrics (100 images, 128x128) ===")
    images = [np.random.randint(0, 256, (128, 128), dtype=np.uint8) for _ in range(100)]

    def py_batch():
        results = []
        for img in images:
            lv = float(cv2.Laplacian(img, cv2.CV_64F).var())
            f = np.fft.fft2(img)
            fs = np.fft.fftshift(f)
            mag = np.abs(fs)
            h, w = img.shape
            cy, cx = h // 2, w // 2
            r = min(h, w) // 8
            mask = np.ones_like(mag)
            mask[cy - r : cy + r, cx - r : cx + r] = 0
            total = np.sum(mag)
            hf = float(np.sum(mag * mask) / total) if total > 0 else 0.0
            results.append((lv, hf))
        return results

    t_py = bench("Python (cv2 + numpy per image)", py_batch, 10)

    if HAS_CPP:
        t_cpp = bench("C++ (batch_blur_metrics)", lambda: cpp.batch_blur_metrics(images), 10)
        print(f"  Speedup: {t_py / t_cpp:.1f}x")


def bench_preprocess():
    """Benchmark preprocessing ops."""
    print("\n=== BGR to Gray (1080x1920x3) ===")
    bgr = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    t_py = bench("Python (cv2.cvtColor)", lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 200)

    if HAS_CPP:
        t_cpp = bench("C++ (bgr_to_gray)", lambda: cpp.bgr_to_gray(bgr), 200)
        print(f"  Speedup: {t_py / t_cpp:.1f}x")

    print("\n=== Resize Gray (1080x1920 -> 270x480) ===")
    gray = np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)
    t_py = bench(
        "Python (cv2.resize)",
        lambda: cv2.resize(gray, (480, 270), interpolation=cv2.INTER_LINEAR),
        200,
    )

    if HAS_CPP:
        t_cpp = bench("C++ (resize_gray)", lambda: cpp.resize_gray(gray, 270, 480), 200)
        print(f"  Speedup: {t_py / t_cpp:.1f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("EventAI C++ Extension Benchmarks")
    if HAS_CPP:
        print(f"Extension version: {cpp.__version__}")
    print("=" * 60)

    bench_laplacian()
    bench_fft_hf_ratio()
    bench_cosine_topk()
    bench_batch_blur()
    bench_preprocess()

    print("\n" + "=" * 60)
    print("Benchmarks complete.")
    print("=" * 60)
