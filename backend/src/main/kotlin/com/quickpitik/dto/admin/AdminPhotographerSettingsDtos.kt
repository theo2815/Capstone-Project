package com.quickpitik.dto.admin

import com.quickpitik.dto.photographer.PayoutAccountDto
import com.quickpitik.dto.photographer.SocialLinkDto

// Admin-side photographer settings — full review surface for the
// verifications drawer + /admin/photographers/[handle]. Mirrors the FE's
// EffectivePhotographerSettings shape (defined in
// website/src/lib/admin-photographer-view.ts) so the FE adapter is thin.
//
// Cover and watermark fields carry both the s3-presigned URL and the
// gradient/label fallbacks so the FE can render whichever the photographer
// uploaded. Region wraps the three location codes the photographer entered.
//
// Reuses SocialLinkDto + PayoutAccountDto from the photographer-side
// surface — no shape divergence, just a different auth gate.

data class AdminPhotographerSettingsDto(
    val userId: String,
    val handle: String?,
    val brandName: String?,
    val brandColor: String,
    val bio: String,
    val region: AdminPhotographerRegionDto?,
    val cover: AdminPhotographerCoverDto?,
    val watermark: AdminPhotographerWatermarkDto?,
    val socials: List<SocialLinkDto>,
    val payouts: List<PayoutAccountDto>,
)

data class AdminPhotographerRegionDto(
    val regionCode: String,
    val provinceCode: String,
    val city: String?,
)

data class AdminPhotographerCoverDto(
    /** Presigned GET URL when the photographer uploaded an image; null when
     *  only a gradient was picked. */
    val url: String?,
    val gradientFrom: String?,
    val gradientTo: String?,
)

data class AdminPhotographerWatermarkDto(
    /** Field is named `dataUrl` to match the FE's WatermarkPreview shape
     *  inherited from the localStorage prototype. With backend storage the
     *  value is a presigned URL. Null when only a label was set. */
    val dataUrl: String?,
    val label: String?,
)
