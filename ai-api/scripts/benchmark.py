"""Benchmark script for comparing Python vs C++ performance paths."""
from __future__ import annotations

import time

import cv2
import numpy as np


def benchmark_blur_detection(image_path: str, iterations: int = 100) -> None:
    """Benchmark blur detection performance."""
    from src.ml.blur.detector import BlurDetector

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    detector = BlurDetector()

    start = time.perf_counter()
    for _ in range(iterations):
        detector.detect(image)
    elapsed = time.perf_counter() - start

    print(f"Blur detection: {iterations} iterations in {elapsed:.3f}s")
    print(f"  Average: {elapsed / iterations * 1000:.2f}ms per image")


def benchmark_cosine_similarity(n_database: int = 10000, iterations: int = 100) -> None:
    """Benchmark face matching cosine similarity."""
    from src.ml.faces.matcher import find_matches

    query = np.random.randn(512).astype(np.float32)
    query /= np.linalg.norm(query)

    database = np.random.randn(n_database, 512).astype(np.float32)
    database /= np.linalg.norm(database, axis=1, keepdims=True)

    start = time.perf_counter()
    for _ in range(iterations):
        find_matches(query, database, threshold=0.3, top_k=10)
    elapsed = time.perf_counter() - start

    print(f"Cosine similarity (1 vs {n_database}): {iterations} iterations in {elapsed:.3f}s")
    print(f"  Average: {elapsed / iterations * 1000:.2f}ms per query")


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("EventAI Performance Benchmark")
    print("=" * 60)

    if len(sys.argv) > 1:
        benchmark_blur_detection(sys.argv[1])
    else:
        print("No image path provided, skipping blur benchmark")

    print()
    benchmark_cosine_similarity(n_database=10000)
    benchmark_cosine_similarity(n_database=50000)
