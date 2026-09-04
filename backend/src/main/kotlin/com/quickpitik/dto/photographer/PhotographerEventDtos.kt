package com.quickpitik.dto.photographer

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import jakarta.validation.constraints.NotBlank
import jakarta.validation.constraints.NotEmpty
import jakarta.validation.constraints.Pattern
import jakarta.validation.constraints.Positive
import jakarta.validation.constraints.Size
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID

// Mirrors website/src/lib/photographer-mock.ts PhotographerEventSummary.
// bannerUrl is the resolved cover (presigned cover_s3_key or legacy banner_url
// fallback) — same value runners see on /events tiles. Carried here so the
// photographer dashboard's covered-event cards render the actual event cover
// instead of the "Banner · soon" placeholder.
data class PhotographerEventSummaryDto(
    val id: UUID,
    val slug: String,
    val name: String,
    val date: LocalDate,
    val location: String,
    val bannerUrl: String?,
    val state: String,
    val photoCount: Int,
    val salesCount: Int,
    val revenueKept: BigDecimal,
    // Photographer-owned events (V46). ownedByMe = the caller created it;
    // the rest are wire-lowercase enums. Defaults keep older clients safe.
    val ownedByMe: Boolean = false,
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val watermarkPolicy: String = "platform",
    val reviewStatus: String = "approved",
    val reviewNote: String? = null,
    val pendingChange: Map<String, Any?>? = null,
)

// Mirrors website/src/lib/api-photographer.ts PhotographerEventDetail
//   = PhotographerEventSummary + firstUploadAt + lastUploadAt.
data class PhotographerEventDetailDto(
    val id: UUID,
    val slug: String,
    val name: String,
    val date: LocalDate,
    val location: String,
    val bannerUrl: String?,
    val state: String,
    val photoCount: Int,
    val salesCount: Int,
    val revenueKept: BigDecimal,
    val firstUploadAt: OffsetDateTime?,
    val lastUploadAt: OffsetDateTime?,
    val ownedByMe: Boolean = false,
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val watermarkPolicy: String = "platform",
    val reviewStatus: String = "approved",
    val reviewNote: String? = null,
    val pendingChange: Map<String, Any?>? = null,
    val description: String = "",
    val organizerName: String = "",
    val pricePerPhoto: BigDecimal = BigDecimal.ZERO,
)

// POST /me/photographer/events (multipart text fields, parsed by the
// controller like the admin create). pricingMode paid|free; a paid event
// needs pricePerPhoto > 0; a free one may pick watermarkPolicy own|none.
data class CreateMyEventRequest(
    @field:NotBlank
    @field:Size(max = 200)
    val title: String,
    @field:NotBlank
    val date: String,
    @field:NotBlank
    @field:Size(max = 200)
    val location: String,
    @field:Size(max = 120)
    val organizerName: String? = null,
    @field:Size(max = 600)
    val description: String? = null,
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val pricePerPhoto: BigDecimal? = null,
    val watermarkPolicy: String? = null,
)

// PATCH /me/photographer/events/{id}. Null/blank = no change. Pricing
// fields on a live event become an edit request (see
// PhotographerOwnedEventService.update); withdrawPendingChange drops it.
data class UpdateMyEventRequest(
    val title: String? = null,
    val date: String? = null,
    val location: String? = null,
    @field:Size(max = 120)
    val organizerName: String? = null,
    @field:Size(max = 600)
    val description: String? = null,
    val visibility: String? = null,
    val pricingMode: String? = null,
    val pricePerPhoto: BigDecimal? = null,
    val watermarkPolicy: String? = null,
    val withdrawPendingChange: Boolean = false,
)

// Mirrors website/src/lib/photographer-mock.ts PhotographerLibraryPhoto.
// status is lowercase wire form; LIVE → "live", HIDDEN → "hidden". Bib is the
// alphabetically-first match if multiple were OCR-detected (matches PhotoDto).
// thumbnailUrl is the watermarked variant (same URL runners see in
// /events/[slug]?browse=1) — the photographer's own dashboard reuses it so
// no second resolver is needed. Clean originals stay behind the explicit
// /me/photographer/photos/{id}/download endpoint.
data class PhotographerLibraryPhotoDto(
    val id: UUID,
    val bib: String?,
    val status: String,
    val salesCount: Int,
    val uploadedAt: OffsetDateTime,
    val tone: Int,
    val span: String,
    val thumbnailUrl: String?,
)

// Mirrors website/src/lib/api-photographer.ts PhotographerDownloadResponse.
data class PhotographerDownloadDto(
    val url: String,
    val expiresAt: OffsetDateTime,
)

// Mirrors website/src/lib/api-photographer.ts UploadedPhoto.
//
// aiDetectionStatus surfaces partial failure of the best-effort ai-api
// faces+bibs pipeline so the photographer dashboard can show a degraded-
// search indicator. null on legacy clients / when both subsystems ran clean
// is treated identically to "ok" (H-5).
//
// Wire values:
//   "ok"                 — faces detect + bibs recognize both succeeded
//   "faces_unavailable"  — faces detect threw; bibs ran cleanly
//   "bibs_unavailable"   — bibs recognize threw; faces ran cleanly
//   "none"               — both threw; runner can find this photo by neither
data class UploadedPhotoDto(
    val id: UUID,
    val status: String,
    val uploadedAt: OffsetDateTime,
    val thumbnailUrl: String,
    val span: String,
    val aiDetectionStatus: String? = null,
)

// Pre-flight duplicate check (dedup Phase 2). The photographer's client hashes
// files locally (SHA-256 of the original bytes — the same identity the upload
// computes server-side) and asks which are already present before spending
// bandwidth. The hash list is capped at the per-batch upload limit (500, mirrors
// the website BATCH_LIMIT). See _journal/2026-06-02-duplicate-upload-detection-plan.
data class PhotoExistsRequest(
    @field:NotEmpty
    @field:Size(max = 500)
    val hashes: List<
        @Pattern(
            regexp = "^[0-9a-fA-F]{64}\$",
            message = "Each hash must be a 64-character hex SHA-256.",
        )
        String,
        >,
)

// One result per requested hash. status:
//   "new"             — not present for this photographer; safe to upload
//   "same_event"      — already in THIS event; an upload is a no-op, skip it
//   "different_event" — already in another of the photographer's events; an
//                       upload would be rejected (eventName names the holder)
data class PhotoExistsResult(
    val hash: String,
    val status: String,
    val eventName: String? = null,
)

data class PhotoExistsResponse(
    val results: List<PhotoExistsResult>,
)

private val phZone: ZoneId = ZoneId.of("Asia/Manila")

internal fun deriveEventState(event: Event, today: LocalDate = LocalDate.now(phZone)): String =
    when (event.status) {
        EventStatus.ACTIVE -> when {
            event.date.isEqual(today) -> "live"
            event.date.isAfter(today) -> "upcoming"
            else -> "open"
        }
        EventStatus.COMPLETED -> "open"
        EventStatus.ARCHIVED -> "past"
        // DRAFT events are filtered server-side; falling through here means a
        // stale row — fall back to the safest non-public state so the FE knows
        // not to surface it as covered.
        EventStatus.DRAFT -> "upcoming"
    }

fun summaryDto(
    event: Event,
    ep: EventPhotographer,
    bannerUrl: String?,
    viewerId: UUID? = null,
): PhotographerEventSummaryDto =
    PhotographerEventSummaryDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        bannerUrl = bannerUrl,
        state = deriveEventState(event),
        photoCount = ep.photoCount,
        salesCount = ep.salesCount,
        revenueKept = ep.revenueKeptPhp,
        ownedByMe = event.createdBy != null && event.createdBy == viewerId,
        visibility = event.visibility.wire,
        pricingMode = event.pricingMode.wire,
        watermarkPolicy = event.watermarkPolicy.wire,
        reviewStatus = event.reviewStatus.wire,
        reviewNote = event.reviewNote,
        pendingChange = event.pendingChange,
    )

fun detailDto(
    event: Event,
    ep: EventPhotographer,
    bannerUrl: String?,
    viewerId: UUID? = null,
): PhotographerEventDetailDto =
    PhotographerEventDetailDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        bannerUrl = bannerUrl,
        state = deriveEventState(event),
        photoCount = ep.photoCount,
        salesCount = ep.salesCount,
        revenueKept = ep.revenueKeptPhp,
        firstUploadAt = ep.firstUploadAt,
        lastUploadAt = ep.lastUploadAt,
        ownedByMe = event.createdBy != null && event.createdBy == viewerId,
        visibility = event.visibility.wire,
        pricingMode = event.pricingMode.wire,
        watermarkPolicy = event.watermarkPolicy.wire,
        reviewStatus = event.reviewStatus.wire,
        reviewNote = event.reviewNote,
        pendingChange = event.pendingChange,
        description = event.description,
        organizerName = event.organizerName,
        pricePerPhoto = event.pricePerPhoto,
    )

fun Photo.toLibraryDto(
    salesCount: Long,
    thumbnailUrlResolver: (Photo) -> String?,
): PhotographerLibraryPhotoDto =
    PhotographerLibraryPhotoDto(
        id = id,
        bib = bibs.minByOrNull { it.bibNumber }?.bibNumber,
        status = when (status) {
            PhotoStatus.LIVE -> "live"
            PhotoStatus.HIDDEN -> "hidden"
            // Processing photos aren't listed yet on the runner-facing grid;
            // surface them to the photographer dashboard as "live" so the
            // upload-just-finished state stays visible.
            PhotoStatus.PROCESSING -> "live"
        },
        salesCount = salesCount.toInt(),
        uploadedAt = uploadedAt,
        tone = tone,
        span = span.wire,
        thumbnailUrl = thumbnailUrlResolver(this),
    )

// ── Direct-to-storage upload (2026-09-02) ─────────────────────────────────
// Two-step alternative to the multipart POST for S3/R2 deployments: the client
// asks for a presigned PUT (begin), streams the original straight to storage,
// then commits. Bytes never cross the backend. `mode` tells the client what to
// do: "direct" (PUT to uploadUrl, then commit), "multipart" (this deployment
// stores on local disk — use the classic endpoint), or "existing" (these bytes
// are already in this event — nothing to upload; `existing` is the photo).
data class DirectUploadBeginRequest(
    @field:Pattern(regexp = "^[0-9a-fA-F]{64}\$", message = "contentHash must be a 64-character hex SHA-256.")
    val contentHash: String,
    @field:NotBlank
    val contentType: String,
    @field:Positive
    val sizeBytes: Long,
)

data class DirectUploadBeginResponse(
    val mode: String,
    val photoId: UUID? = null,
    val key: String? = null,
    val uploadUrl: String? = null,
    val expiresInSeconds: Long? = null,
    val existing: UploadedPhotoDto? = null,
)

data class DirectUploadCommitRequest(
    val photoId: UUID,
    @field:NotBlank
    val key: String,
    @field:Pattern(regexp = "^[0-9a-fA-F]{64}\$", message = "contentHash must be a 64-character hex SHA-256.")
    val contentHash: String,
    @field:NotBlank
    val contentType: String,
)
