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

// Fused classify preprocessing: center-crop + bilinear resize + BGR->RGB
// normalize to [0,1] float32 + HWC->CHW transpose.
// Returns (1, 3, target_size, target_size) float32 tensor ready for ONNX.
py::array_t<float> classify_preprocess(py::array_t<uint8_t> bgr,
                                        int target_size);

}  // namespace eventai
