# Python + C++ Integration

## Why C++?

Python is great for writing application logic but slow for tight loops over raw data. Not everything needs C++ though. The rule is:

**Only optimize the inner loops that process raw pixel data or compute distances over large arrays thousands of times per request.**

Most of the "heavy" work (model inference) is already in C++ under the hood via OpenCV and ONNX Runtime. The C++ extensions here target the gaps between those libraries where Python overhead adds up.

## What Gets Accelerated

| Component | Function | Observed speedup | Notes |
|---|---|---|---|
| Laplacian variance (single image) | `laplacian_variance` | ~5.4× | Single-pass sum + sum_sq kernel; biggest win |
| FFT high-frequency ratio | `fft_hf_ratio` | ~1.2× | Radix-2 Cooley-Tukey; ~parity with NumPy FFTPACK |
| BGR → gray, resize, classifier preprocess | `bgr_to_gray`, `resize_gray`, `classify_preprocess` | neutral to slower | OpenCV/NumPy already use hand-tuned SIMD; kept for the GIL-release benefit inside threaded pipelines. **`classify_preprocess` is no longer called by `BlurClassifier`** — it is bilinear, and the classifier needs an area-averaged downscale for accuracy (2026-08-14). Still exported and tested. |

Numbers are from `benchmarks/bench_cpp_vs_python.py` on MSVC 19.50 / Windows / AVX2 / Python 3.12. They will differ on Linux-glibc with GCC.

The blur functions remain exported for benchmarks but are not called by the served detector: the C++ Laplacian excludes border pixels and gives different scores from OpenCV. The detector uses OpenCV/NumPy in every build so optional extension installation cannot change a blur decision.

**Key non-speed benefit:** every C++ function releases the GIL via `py::gil_scoped_release`, so multiple Python threads can overlap computation when callers use the exported extension functions. Served blur detection uses OpenCV/NumPy instead.

## How It Works

### Binding Technology: pybind11

pybind11 is a C++ library that creates Python modules from C++ code. It has first-class support for NumPy arrays -- you can pass a Python NumPy array to C++ and access the underlying memory directly, with zero copying.

### File Structure

```
ai-api/
├── CMakeLists.txt              # Build instructions (C++17, pybind11, AVX2)
├── build_cpp.py                # Build script (auto-detects MSVC on Windows)
│
└── src/cpp/                    # All C++ source files (flat layout)
    ├── bindings.cpp            # pybind11 module definition (glue)
    ├── blur_ops.h              # Blur detection headers
    ├── blur_ops.cpp            # Batch Laplacian variance + FFT HF ratio
    ├── preprocess_ops.h        # Image preprocessing headers
    └── preprocess_ops.cpp      # Fused bgr_to_gray + resize_gray (bilinear)
```

C++ tests are Python-based: `tests/test_cpp_extension.py` (verifies numerical parity with the NumPy fallbacks).

### The Graceful Fallback Pattern

Nothing in `src/` imports the extension today. Any Python file that uses it must follow this pattern:

```python
try:
    from _quickpitik_cpp import laplacian_variance
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

def score(gray):
    if _HAS_CPP:
        # Fast C++ path (only where the output contract is identical)
        return laplacian_variance(gray)

    # Pure NumPy/OpenCV fallback (always works)
    # ...
```

This means:
- **The app always runs**, even without compiled C++ code
- During development, you use the NumPy path (no C++ compiler needed)
- In production Docker builds, C++ is compiled and used automatically
- You can benchmark both paths with `python scripts/benchmark.py`

## Building the C++ Extension

### In Docker (automatic)
The Dockerfile has a build stage that compiles C++ before creating the runtime image. You don't need to do anything.

### Locally (manual)
```bash
# Option A — via the pyproject extras (uses scikit-build-core)
pip install -e ".[cpp]"

# Option B — direct CMake build (helpful on Windows when pip misdetects MSVC)
python build_cpp.py
```

Both paths produce `_quickpitik_cpp.<tag>.pyd` (Windows) or `.so` (Linux/macOS) at the project root where Python can `import _quickpitik_cpp`.

### Requirements
- **Linux**: GCC 10+ or Clang 12+
- **Windows**: MSVC (Visual Studio Build Tools 2022+) — `build_cpp.py` auto-detects `vcvarsall.bat`
- **macOS**: Clang (Xcode Command Line Tools)
- **All**: CMake 3.18+, Ninja (pulled in by the `[cpp]` extra)

OpenCV development headers are **not** required — the extension only uses its own sources plus pybind11's NumPy bindings.

## When to Add Another C++ Function

The rule: profile first, then decide. New C++ is worth adding only when a pure Python version falls behind on a hot path.

1. Run `python benchmarks/bench_cpp_vs_python.py` or `python scripts/benchmark.py` to confirm the Python baseline.
2. Prototype in C++, keeping the NumPy path intact.
3. Re-run benchmarks.
4. If the speedup is below ~2× or the new code can't release the GIL, don't merge it — the maintenance cost isn't worth it.

The existing extension shows the ceiling clearly: NumPy-backed BLAS and OpenCV's SIMD paths are very fast, so naive C++ loops lose on pre-processing and large-N matmul-like workloads. Wins come from fusing passes (single-pass Laplacian variance).
