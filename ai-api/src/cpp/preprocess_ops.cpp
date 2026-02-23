#include "preprocess_ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eventai {

py::array_t<uint8_t> bgr_to_gray(py::array_t<uint8_t> bgr) {
    auto buf = bgr.unchecked<3>();
    const py::ssize_t H = buf.shape(0);
    const py::ssize_t W = buf.shape(1);

    if (buf.shape(2) != 3) {
        throw std::invalid_argument("Input must be (H, W, 3) BGR image");
    }

    auto result = py::array_t<uint8_t>({H, W});
    auto out = result.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;

        for (py::ssize_t y = 0; y < H; ++y) {
            for (py::ssize_t x = 0; x < W; ++x) {
                // BGR order: 0.114*B + 0.587*G + 0.299*R
                double gray_val = 0.114 * buf(y, x, 0) +
                                  0.587 * buf(y, x, 1) +
                                  0.299 * buf(y, x, 2);
                out(y, x) = static_cast<uint8_t>(
                    std::min(255.0, std::max(0.0, std::round(gray_val))));
            }
        }
    }

    return result;
}

py::array_t<uint8_t> resize_gray(py::array_t<uint8_t> gray, int new_h,
                                  int new_w) {
    auto buf = gray.unchecked<2>();
    const py::ssize_t H = buf.shape(0);
    const py::ssize_t W = buf.shape(1);

    if (new_h <= 0 || new_w <= 0) {
        throw std::invalid_argument("Target dimensions must be positive");
    }

    auto result = py::array_t<uint8_t>({static_cast<py::ssize_t>(new_h),
                                         static_cast<py::ssize_t>(new_w)});
    auto out = result.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;

        const double sy = static_cast<double>(H) / new_h;
        const double sx = static_cast<double>(W) / new_w;

        for (int y = 0; y < new_h; ++y) {
            double src_y = (y + 0.5) * sy - 0.5;
            py::ssize_t y0 = static_cast<py::ssize_t>(std::floor(src_y));
            py::ssize_t y1 = y0 + 1;
            double fy = src_y - y0;

            y0 = std::max(static_cast<py::ssize_t>(0),
                          std::min(y0, H - 1));
            y1 = std::max(static_cast<py::ssize_t>(0),
                          std::min(y1, H - 1));

            for (int x = 0; x < new_w; ++x) {
                double src_x = (x + 0.5) * sx - 0.5;
                py::ssize_t x0 = static_cast<py::ssize_t>(std::floor(src_x));
                py::ssize_t x1 = x0 + 1;
                double fx = src_x - x0;

                x0 = std::max(static_cast<py::ssize_t>(0),
                              std::min(x0, W - 1));
                x1 = std::max(static_cast<py::ssize_t>(0),
                              std::min(x1, W - 1));

                double val = (1.0 - fy) * ((1.0 - fx) * buf(y0, x0) +
                                            fx * buf(y0, x1)) +
                             fy * ((1.0 - fx) * buf(y1, x0) +
                                   fx * buf(y1, x1));
                out(y, x) = static_cast<uint8_t>(
                    std::min(255.0, std::max(0.0, std::round(val))));
            }
        }
    }

    return result;
}

}  // namespace eventai
