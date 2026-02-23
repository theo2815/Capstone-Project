#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace eventai {

// Convert BGR (H, W, 3) uint8 image to grayscale (H, W) uint8.
// Uses luminance weights: 0.114 * B + 0.587 * G + 0.299 * R.
py::array_t<uint8_t> bgr_to_gray(py::array_t<uint8_t> bgr);

// Resize grayscale (H, W) uint8 image using bilinear interpolation.
py::array_t<uint8_t> resize_gray(py::array_t<uint8_t> gray, int new_h,
                                  int new_w);

}  // namespace eventai
