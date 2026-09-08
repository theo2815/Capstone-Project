package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.WatermarkSettingsDto
import kotlin.math.roundToInt

// The free-space placement formula for the photographer's own logo, shared by
// the drag preview and mirrored exactly by the backend
// (WatermarkService.watermarkBox) and the website
// (lib/watermark-placement) — so the preview is an accurate promise of what
// gets baked. (0,0) pins the logo to the padded top-left, (1,1) to the padded
// bottom-right; any value keeps it fully in frame at any photo size.
//
// Pure JVM (no Compose) so it is unit-testable directly.

const val WATERMARK_EDGE_PAD_RATIO = 0.025
const val WATERMARK_OPACITY_MIN = 0.05f
const val WATERMARK_OPACITY_MAX = 1f
const val WATERMARK_SCALE_MIN = 0.05f
const val WATERMARK_SCALE_MAX = 0.40f

// Representative 3:2 photos after the compositor's 1280px long-edge cap.
const val WATERMARK_PREVIEW_LONG_EDGE = 1280
const val WATERMARK_PREVIEW_SHORT_EDGE = 853

data class WatermarkBox(val left: Int, val top: Int, val width: Int, val height: Int)

// Use rendered-photo pixels, not screen pixels, for minimums and rounding.
// Fit both axes together when a tall logo or tiny photo would crop the mark.
fun watermarkBox(
    photoW: Int,
    photoH: Int,
    logoAspect: Double,
    scale: Double,
    nx: Double,
    ny: Double,
): WatermarkBox {
    val pad = minOf((photoW * WATERMARK_EDGE_PAD_RATIO).toInt().coerceAtLeast(12), (minOf(photoW, photoH) - 1) / 2)
    val requestedWidth = (photoW * scale).toInt().coerceAtLeast(48)
    val requestedHeight = (requestedWidth / logoAspect).toInt().coerceAtLeast(1)
    val fit = minOf(1.0, (photoW - 2 * pad).toDouble() / requestedWidth, (photoH - 2 * pad).toDouble() / requestedHeight)
    val width = (requestedWidth * fit).toInt().coerceAtLeast(1)
    val height = (requestedHeight * fit).toInt().coerceAtLeast(1)
    return WatermarkBox(
        (pad + nx.coerceIn(0.0, 1.0) * (photoW - width - 2 * pad)).roundToInt(),
        (pad + ny.coerceIn(0.0, 1.0) * (photoH - height - 2 * pad)).roundToInt(),
        width, height,
    )
}

// A dragged top-left in rendered-photo pixels → normalized position. Reuse
// the fitted bounds so size limits and pixel rounding cannot drift on drag.
// Preserve an axis with no free space so shrinking the logo restores its position.
fun positionFromBox(
    photoW: Int,
    photoH: Int,
    logoAspect: Double,
    scale: Double,
    left: Double,
    top: Double,
    currentX: Double,
    currentY: Double,
): Pair<Double, Double> {
    val start = watermarkBox(photoW, photoH, logoAspect, scale, 0.0, 0.0)
    val end = watermarkBox(photoW, photoH, logoAspect, scale, 1.0, 1.0)
    return (if (end.left > start.left) ((left - start.left) / (end.left - start.left)).coerceIn(0.0, 1.0) else currentX) to
        (if (end.top > start.top) ((top - start.top) / (end.top - start.top)).coerceIn(0.0, 1.0) else currentY)
}

// Orientation-aware accessors over the flat wire DTO.
fun WatermarkSettingsDto.pos(landscape: Boolean): Pair<Double, Double> =
    if (landscape) landscapeX to landscapeY else portraitX to portraitY

fun WatermarkSettingsDto.withPos(landscape: Boolean, x: Double, y: Double): WatermarkSettingsDto =
    if (landscape) copy(landscapeX = x, landscapeY = y) else copy(portraitX = x, portraitY = y)
