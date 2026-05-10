package com.quickpitik.dto.photographer

import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.SocialLink
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.Size
import java.time.OffsetDateTime

// ─── Brand / Handle / Region request DTOs ─────────────────────────────────
// Mirrors website/src/lib/api-photographer-settings.ts BrandPatch.
data class BrandPatchRequest(
    @field:Size(max = 100, message = "brandName must be 0-100 chars")
    val brandName: String? = null,
    val brandColor: String? = null,
    @field:Size(max = 2000, message = "bio must be 0-2000 chars")
    val bio: String? = null,
)

data class HandlePatchRequest(
    @field:NotBlank(message = "handle is required")
    val handle: String,
)

data class RegionPatchRequest(
    @field:NotBlank(message = "regionCode is required")
    val regionCode: String,
    @field:NotBlank(message = "provinceCode is required")
    val provinceCode: String,
)

// ─── Cover / Watermark / QR upload response ───────────────────────────────
// Mirrors website/src/lib/api-photographer-settings.ts MediaUploadResponse.
data class MediaUploadResponseDto(
    val url: String,
    val uploadedAt: OffsetDateTime,
)

// ─── Socials request + wire DTOs ──────────────────────────────────────────
// Mirrors website/src/store/photographer-settings-store.ts SocialLink.
// id is a UUID string; platform is lowercase wire form ("facebook" | ...).
data class SocialLinkDto(
    val id: String,
    val platform: String,
    val url: String,
)

data class CreateSocialRequest(
    @field:NotBlank(message = "platform is required")
    val platform: String,
    @field:NotBlank(message = "url is required")
    val url: String,
)

data class PatchSocialRequest(
    @field:NotBlank(message = "url is required")
    val url: String,
)

fun SocialLink.toDto(): SocialLinkDto =
    SocialLinkDto(
        id = id.toString(),
        platform = platform.wire,
        url = url,
    )

// ─── Payouts request + wire DTOs ──────────────────────────────────────────
// Mirrors website/src/store/photographer-settings-store.ts PayoutAccount + PayoutQr.
//
// PayoutQr.dataUrl is the FE field name from the localStorage prototype where
// a base64 data URL was stored. With backend storage the value is a presigned
// URL — same field name, same <img src> usage on the FE, so no contract change.
data class PayoutQrDto(
    val dataUrl: String,
    val uploadedAt: OffsetDateTime,
)

data class PayoutAccountDto(
    val id: String,
    val method: String,
    val accountNumber: String,
    val accountName: String,
    val qr: PayoutQrDto?,
    val isPrimary: Boolean,
)

data class CreatePayoutRequest(
    @field:NotBlank(message = "method is required")
    val method: String,
    @field:NotBlank(message = "accountNumber is required")
    val accountNumber: String,
    @field:NotBlank(message = "accountName is required")
    val accountName: String,
)

data class PatchPayoutRequest(
    val accountNumber: String? = null,
    val accountName: String? = null,
)

fun PayoutAccount.toDto(qrUrl: String?, qrUploadedAt: OffsetDateTime?): PayoutAccountDto =
    PayoutAccountDto(
        id = id.toString(),
        method = method.wire,
        accountNumber = accountNumber,
        accountName = accountName,
        qr = qrUrl?.let { PayoutQrDto(dataUrl = it, uploadedAt = qrUploadedAt ?: createdAt) },
        isPrimary = isPrimary,
    )

// ─── Verification submit response ─────────────────────────────────────────
// Mirrors website/src/lib/api-photographer-settings.ts VerificationSubmitResponse.
// status is lowercase wire form ("incomplete" | "pending" | "approved" | "rejected").
data class VerificationSubmitResponseDto(
    val status: String,
    val missing: List<String>? = null,
)

// Used by the submit endpoint when the profile is incomplete — lifted out so
// the controller / service can build it without re-parsing entity state.
data class IncompleteFields(
    val missing: List<String>,
) {
    val isComplete: Boolean get() = missing.isEmpty()
}
