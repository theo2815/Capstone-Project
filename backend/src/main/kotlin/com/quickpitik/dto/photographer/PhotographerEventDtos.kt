package com.quickpitik.dto.photographer

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
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
    )

fun detailDto(
    event: Event,
    ep: EventPhotographer,
    bannerUrl: String?,
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
