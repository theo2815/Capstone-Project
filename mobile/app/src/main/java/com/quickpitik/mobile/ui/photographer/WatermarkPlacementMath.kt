package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.WatermarkSettingsDto

// The free-space placement formula for the photographer's own logo, shared by
// the drag preview and mirrored exactly by the backend
// (WatermarkService.drawWatermarkImage) and the website
// (lib/watermark-placement) — so the preview is an accurate promise of what
// gets baked. (0,0) pins the logo to the padded top-left, (1,1) to the padded
// bottom-right; any value keeps it fully in frame at any photo size.
//
// Pure JVM (no Compose) so it is unit-testable directly.

const val WATERMARK_EDGE_PAD_RATIO = 0.025f
const val WATERMARK_OPACITY_MIN = 0.05f
const val WATERMARK_OPACITY_MAX = 1f
const val WATERMARK_SCALE_MIN = 0.05f
const val WATERMARK_SCALE_MAX = 0.40f

data class WatermarkBox(val left: Float, val top: Float, val width: Float, val height: Float)

private fun clamp01(v: Float): Float = v.coerceIn(0f, 1f)

// `logoAspect` = logo natural width / height; `boxW`/`boxH` = preview size in px.
// Height follows the aspect ratio so scaling never distorts the logo.
fun watermarkBox(
    boxW: Float,
    boxH: Float,
    logoAspect: Float,
    scale: Float,
    nx: Float,
    ny: Float,
): WatermarkBox {
    val pad = boxW * WATERMARK_EDGE_PAD_RATIO
    val width = boxW * scale
    val height = if (logoAspect > 0f) width / logoAspect else width
    val freeW = (boxW - width - 2 * pad).coerceAtLeast(0f)
    val freeH = (boxH - height - 2 * pad).coerceAtLeast(0f)
    return WatermarkBox(pad + clamp01(nx) * freeW, pad + clamp01(ny) * freeH, width, height)
}

// Inverse of watermarkBox: a dragged pixel top-left → the normalized (x,y).
fun positionFromBox(
    boxW: Float,
    boxH: Float,
    logoAspect: Float,
    scale: Float,
    left: Float,
    top: Float,
): Pair<Float, Float> {
    val pad = boxW * WATERMARK_EDGE_PAD_RATIO
    val width = boxW * scale
    val height = if (logoAspect > 0f) width / logoAspect else width
    val freeW = (boxW - width - 2 * pad).coerceAtLeast(1e-4f)
    val freeH = (boxH - height - 2 * pad).coerceAtLeast(1e-4f)
    return clamp01((left - pad) / freeW) to clamp01((top - pad) / freeH)
}

// Orientation-aware accessors over the flat wire DTO.
fun WatermarkSettingsDto.pos(landscape: Boolean): Pair<Double, Double> =
    if (landscape) landscapeX to landscapeY else portraitX to portraitY

fun WatermarkSettingsDto.withPos(landscape: Boolean, x: Double, y: Double): WatermarkSettingsDto =
    if (landscape) copy(landscapeX = x, landscapeY = y) else copy(portraitX = x, portraitY = y)
