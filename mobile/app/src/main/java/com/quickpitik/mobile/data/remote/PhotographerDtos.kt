package com.quickpitik.mobile.data.remote

data class PaginatedResponse<T>(
    val items: List<T>,
    val total: Long,
    val offset: Int,
    val limit: Int,
    val snapshotAt: String? = null,
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
    val bannerUrl: String? = null,
    // The caller's own public handle, from the backend — pair with slug to build
    // the shareable gallery URL (quickpitik.com/{handle}/events/{slug}). null
    // until the photographer sets a handle; the share screen gates on it instead
    // of copying a "your-handle" placeholder.
    val handle: String? = null,
    // Photographer-owned events (V46). ownedByMe = the caller created it (and may
    // upload on any date); the rest are lowercase wire enums, defaults keeping the
    // pre-V46 admin-event shape safe. pendingChange is the parked pricing trio on
    // a live event — an untyped JSON blob (Map), matching the backend. reviewStatus:
    // pending | approved | rejected | change_pending.
    val ownedByMe: Boolean = false,
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val watermarkPolicy: String = "platform",
    val reviewStatus: String = "approved",
    val reviewNote: String? = null,
    val pendingChange: Map<String, Any?>? = null,
)

// POST/PATCH /me/photographer/events response (create/edit owned event, V46).
// Superset of the summary + editorial fields + pricePerPhoto. Backend
// PhotographerEventDetailDto.
data class PhotographerEventDetailDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val location: String,
    val bannerUrl: String? = null,
    val state: String,
    val photoCount: Int,
    val salesCount: Int,
    val revenueKept: Double,
    val firstUploadAt: String? = null,
    val lastUploadAt: String? = null,
    // See PhotographerEventSummaryDto.handle.
    val handle: String? = null,
    val ownedByMe: Boolean = false,
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val watermarkPolicy: String = "platform",
    val reviewStatus: String = "approved",
    val reviewNote: String? = null,
    val pendingChange: Map<String, Any?>? = null,
    val description: String = "",
    val organizerName: String = "",
    val pricePerPhoto: Double = 0.0,
)

// Event coupon (V47) — lives on the event, not Settings.
// GET/PUT/DELETE /me/photographer/events/{eventId}/coupon. GET data is null when
// none. Backend CouponDtos.
data class CouponDto(
    val eventId: String,
    val code: String,
    val percentOff: Int,
    val active: Boolean,
    val expiresAt: String? = null,
    val usageLimit: Int? = null,
    val usageCount: Long = 0,
    val updatedAt: String,
)

data class UpsertCouponRequest(
    val code: String,
    val percentOff: Int,
    val active: Boolean = true,
    val expiresAt: String? = null,
    val usageLimit: Int? = null,
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
    // Deprecated backend-side in favour of requestedAt — parsed for
    // compatibility, never rendered.
    val weekOf: String,
    val amount: Double,
    val status: String,
    val settledAt: String?,
    val method: String,
    val reference: String?,
    val holdReason: String?,
    // When the photographer filed this request (the field clients should
    // render — backend EarningsDtos doc-comment).
    val requestedAt: String? = null,
)

data class PayoutBalanceDto(
    val unpaidBalance: Double,
    // Earned but still inside the clearing window — not yet withdrawable.
    // Defaulted for GSON back-compat with older backends that omit it.
    val clearing: Double = 0.0,
    // Tied up in an open request (pending/approved/processing/held).
    val reserved: Double = 0.0,
    // ISO timestamp; non-null while a recent payout-account change still bars
    // requests. The wallet shows a paused banner + disables Request until it passes.
    val accountCooldownUntil: String? = null,
    val minimum: Double,
    val hasOpenRequest: Boolean,
    val openRequest: PhotographerPayoutDto?,
    // Backend-authoritative primary payout account (null = none configured).
    // The request-payout CTA gates on THIS, not on any client-side settings
    // state — the field exists precisely so a client can't offer the CTA and
    // then eat a PAYOUT_NO_ACCOUNT (backend EarningsDtos.kt:147-154).
    val primaryAccount: PayoutBalanceAccountDto? = null
)

data class PayoutBalanceAccountDto(
    val method: String,
    val accountNumber: String,
    val accountName: String,
    val hasQr: Boolean = false,
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
    val missing: List<String>?, // human labels, e.g. "profile picture"
    val missingCodes: List<String>? = null, // stable keys, e.g. "PROFILE_PHOTO"
    val suspendedAt: String?,
    val suspensionReason: String?
)

data class BrandPatchRequest(
    val brandName: String?,
    val brandColor: String?,
    val bio: String?
)

// Placement of the photographer's own logo (V51) — position split per
// orientation, plus shared opacity + size. Doubles as the nested block on the
// brand response AND the PUT body for updateWatermarkSettings (same field set as
// the backend WatermarkSettingsRequest). Defaults reproduce the historical fixed
// bottom-right / 15% width / 70% opacity mark, so an unconfigured photographer
// is unchanged. Position values are normalized free-space fractions in [0,1].
data class WatermarkSettingsDto(
    val opacity: Double = 0.70,
    val scale: Double = 0.15,
    val landscapeX: Double = 1.0,
    val landscapeY: Double = 1.0,
    val portraitX: Double = 1.0,
    val portraitY: Double = 1.0,
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
    val avatarUrl: String?,
    val watermarkSettings: WatermarkSettingsDto? = null,
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
    val thumbnailUrl: String? = null,
    // Soft-delete timestamp (V52); non-null = in the photographer's trash. Null
    // on the active list, set on the ?deleted=true (Deleted tab) list.
    val deletedAt: String? = null,
    val deletionStartedAt: String? = null,
)

// Presigned URL for the photographer's own un-watermarked original — the
// thumbnailUrl above is the watermarked variant runners see. Mirrors backend
// PhotographerDownloadDto; expiresAt is ISO-8601 and unused today (the link is
// consumed immediately by PhotoDownloader), kept so the shape matches the wire.
data class PhotographerDownloadDto(
    val url: String,
    val expiresAt: String? = null
)

// Photo library management (V52). Trash/restore/permanent-delete are bulk; the
// per-photo UI sends a one-element list.
data class PhotoIdsRequest(
    val photoIds: List<String>
)

// trash / restore → how many photos actually moved (ids already in that state
// are no-ops and don't count).
data class PhotoActionResponseDto(
    val affected: Int
)

// permanent-delete → the ids actually purged, and the ones refused because
// they've been sold (kept so buyers can still download them).
data class PermanentDeleteResponseDto(
    val deleted: List<String> = emptyList(),
    val blockedSold: List<String> = emptyList(),
    val blockedPending: List<String> = emptyList(),
    val failed: List<String> = emptyList(),
)

// "Download all" step 1 → a short-lived tokened URL the client streams the ZIP
// (or, for a single-photo event, the raw JPEG) from.
data class PhotographerBundleDto(
    val url: String,
    val expiresAt: String? = null,
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
    val salesCount: Int,
    val event: EventDto? = null,
    val visibility: String = "public",
    val pricingMode: String = "paid",
)

data class PhotographerProfileDto(
    val handle: String? = null,
    val displayName: String? = null,
    val avatarUrl: String? = null,
    val brandColor: String? = null,
    val bio: String? = null,
    val city: String? = null,
    val memberSince: String? = null,
    val cover: CoverSourceDto? = null,
    val watermarkLabel: String? = null,
    val events: List<PhotographerEventCoverageDto> = emptyList()
)

// ── Direct-to-storage upload (backend 2026-09-02) ─────────────────────────
// begin → PUT the file to `uploadUrl` → commit. `mode` = "direct" | "multipart"
// (local-disk dev backend — use the classic endpoint) | "existing" (bytes are
// already in this event; `existing` is the photo, nothing to upload).
data class DirectUploadBeginRequest(
    val contentHash: String,
    val contentType: String,
    val sizeBytes: Long,
)

data class DirectUploadBeginResponse(
    val mode: String? = null,
    val photoId: String? = null,
    val key: String? = null,
    val uploadUrl: String? = null,
    val expiresInSeconds: Long? = null,
    val existing: UploadedPhotoDto? = null,
)

data class DirectUploadCommitRequest(
    val photoId: String,
    val key: String,
    val contentHash: String,
    val contentType: String,
    val sizeBytes: Long? = null,
)
