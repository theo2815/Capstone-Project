package com.quickpitik.mobile.data.remote

data class PaginatedResponse<T>(
    val items: List<T>,
    val total: Long,
    val offset: Int,
    val limit: Int
)

data class PhotographerEventSummaryDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val location: String,
    val state: String, // "live", "upcoming", "open", "past"
    val photoCount: Int,
    val salesCount: Int,
    val revenueKept: Double,
    val bannerUrl: String? = null
)

data class WeeklyRevenuePointDto(
    val weekOf: String,
    val amount: Double
)

data class EarningsOverviewDto(
    val lifetimeKept: Double,
    val thisWeek: Double,
    val thisMonth: Double,
    val payoutPending: Double,
    val payoutScheduledFor: String?,
    val weeklySeries: List<WeeklyRevenuePointDto>,
    val thisWeekSold: Long,
    val thisMonthSold: Long
)

data class PhotographerPayoutDto(
    val id: String,
    val weekOf: String,
    val amount: Double,
    val status: String,
    val settledAt: String?,
    val method: String,
    val reference: String?,
    val holdReason: String?
)

data class PayoutBalanceDto(
    val unpaidBalance: Double,
    val minimum: Double,
    val hasOpenRequest: Boolean,
    val openRequest: PhotographerPayoutDto?
)

data class PhotographerTransactionDto(
    val id: String,
    val paidAt: String,
    val eventId: String,
    val eventName: String?,
    val eventSlug: String?,
    val photoId: String,
    val buyer: String,
    val amountKept: Double
)

data class TransactionsLedgerResponse(
    val items: List<PhotographerTransactionDto>,
    val total: Long,
    val offset: Int,
    val limit: Int,
    val monthTotals: Map<String, Double>
)

data class VerificationSubmitResponseDto(
    val status: String, // "incomplete", "pending", "approved", "rejected"
    val missing: List<String>?,
    val suspendedAt: String?,
    val suspensionReason: String?
)

data class BrandPatchRequest(
    val brandName: String?,
    val brandColor: String?,
    val bio: String?
)

data class BrandSettingsResponseDto(
    val brandName: String?,
    val brandColor: String?,
    val bio: String?,
    val handle: String?,
    val regionCode: String?,
    val provinceCode: String?,
    val coverUrl: String?,
    val watermarkUrl: String?,
    val avatarUrl: String?
)

data class MediaUploadResponseDto(
    val url: String,
    val uploadedAt: String
)

data class CreatePayoutRequest(
    val method: String,
    val accountNumber: String,
    val accountName: String
)

data class PayoutQrDto(
    val dataUrl: String,
    val uploadedAt: String
)

data class PayoutAccountDto(
    val id: String,
    val method: String,
    val accountNumber: String,
    val accountName: String,
    val qr: PayoutQrDto?,
    val isPrimary: Boolean
)

data class HandlePatchRequest(
    val handle: String
)

data class RegionPatchRequest(
    val regionCode: String,
    val provinceCode: String
)

/**
 * Body for `POST /me/photographer/events/{id}/photos/exists` — the dedup
 * pre-flight. Ask which of these SHA-256s the backend already holds before
 * spending bandwidth uploading them. Backend caps the list at 500 and rejects
 * anything that is not 64 hex characters.
 */
data class PhotoExistsRequest(
    val hashes: List<String>
)

/**
 * One result per requested hash. [status] is one of:
 *  - `new`             — not present for this photographer; upload it
 *  - `same_event`      — already in THIS event; uploading is a no-op, skip
 *  - `different_event` — in another of the photographer's events; an upload
 *                        would be rejected 409 ([eventName] names the holder)
 */
data class PhotoExistsResult(
    val hash: String,
    val status: String,
    val eventName: String? = null
)

data class PhotoExistsResponse(
    val results: List<PhotoExistsResult>
)

/**
 * `GET /api/v1/regions` — the canonical PH region + province list. Mirrors
 * backend dto/reference/RegionDto.
 *
 * Until 2026-08-16 this screen rendered a hardcoded 75-line copy while the
 * website carried a third copy of its own, so a region added backend-side
 * reached neither client. The backend owns the list; both clients read it.
 *
 * Public endpoint, and the response carries `Cache-Control: max-age=1d`.
 * `shortName`/`group` are sent by the backend and kept here to match the wire
 * shape even though this screen currently renders only `name`.
 */
data class RegionDto(
    val code: String,
    val name: String,
    val shortName: String,
    val group: String,
    val provinces: List<ProvinceDto>
)

data class ProvinceDto(
    val code: String,
    val name: String
)

data class CreateSocialRequest(
    val platform: String,
    val url: String
)

data class PatchSocialRequest(
    val url: String
)

data class PatchPayoutRequest(
    val accountNumber: String? = null,
    val accountName: String? = null
)

data class SocialLinkDto(
    val id: String,
    val platform: String,
    val url: String
)

data class PhotographerMessageDto(
    val id: String,
    val kind: String,
    val title: String?,
    val body: String,
    val sourceDecisionId: String?,
    val createdAt: String,
    val readAt: String?
)

// Push frame on /ws/me/photographer/notifications. Built by
// AdminDecisionLogService.pushMessage from the same field set as the REST DTO
// above, so it deserializes straight in. Mirrors RunnerMessageFrame.
data class PhotographerMessageFrame(
    val type: String?,
    val message: PhotographerMessageDto?,
)

data class MarkAllReadResponse(
    val markedRead: Int
)

data class MessageRemovedResponse(
    val removed: Boolean
)

data class PhotographerLibraryPhotoDto(
    val id: String,
    val bib: String?,
    val status: String,
    val salesCount: Int,
    val uploadedAt: String,
    val tone: Int,
    val span: String,
    val thumbnailUrl: String? = null
)

// Presigned URL for the photographer's own un-watermarked original — the
// thumbnailUrl above is the watermarked variant runners see. Mirrors backend
// PhotographerDownloadDto; expiresAt is ISO-8601 and unused today (the link is
// consumed immediately by PhotoDownloader), kept so the shape matches the wire.
data class PhotographerDownloadDto(
    val url: String,
    val expiresAt: String? = null
)

// --- Public photographer profile (GET /public/photographers/{handle}) ---
data class CoverSourceDto(
    val kind: String, // "gradient" | "image"
    val from: String? = null,
    val to: String? = null,
    val url: String? = null
)

data class PhotographerEventCoverageDto(
    val eventSlug: String,
    val state: String,
    val photoCount: Int,
    val salesCount: Int
)

data class PhotographerProfileDto(
    val handle: String? = null,
    val displayName: String? = null,
    val brandColor: String? = null,
    val bio: String? = null,
    val city: String? = null,
    val memberSince: String? = null,
    val cover: CoverSourceDto? = null,
    val watermarkLabel: String? = null,
    val events: List<PhotographerEventCoverageDto> = emptyList()
)
