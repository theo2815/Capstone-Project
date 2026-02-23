#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

namespace eventai {

struct BlurMetrics {
    double laplacian_var;
    double hf_ratio;
};

// Laplacian variance using the kernel [0,1,0; 1,-4,1; 0,1,0].
// Single-pass computation of variance (sum + sum_sq).
double laplacian_variance(py::array_t<uint8_t> gray);

// 2D FFT -> fftshift -> HF/total magnitude ratio.
// Center exclusion radius = min(h, w) / 8.
double fft_hf_ratio(py::array_t<uint8_t> gray);

// Batch wrapper: compute both metrics for each image.
std::vector<BlurMetrics> batch_blur_metrics(
    const std::vector<py::array_t<uint8_t>>& images);

}  // namespace eventai
