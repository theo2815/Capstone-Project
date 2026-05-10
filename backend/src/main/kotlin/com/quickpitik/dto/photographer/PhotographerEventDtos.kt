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
data class PhotographerEventSummaryDto(
    val id: UUID,
    val slug: String,
    val name: String,
    val date: LocalDate,
    val location: String,
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
data class PhotographerLibraryPhotoDto(
    val id: UUID,
    val bib: String?,
    val status: String,
    val salesCount: Int,
    val uploadedAt: OffsetDateTime,
    val tone: Int,
    val span: String,
)

// Mirrors website/src/lib/api-photographer.ts PhotographerDownloadResponse.
data class PhotographerDownloadDto(
    val url: String,
    val expiresAt: OffsetDateTime,
)

// Mirrors website/src/lib/api-photographer.ts UploadedPhoto.
data class UploadedPhotoDto(
    val id: UUID,
    val status: String,
    val uploadedAt: OffsetDateTime,
    val thumbnailUrl: String,
    val span: String,
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

fun summaryDto(event: Event, ep: EventPhotographer): PhotographerEventSummaryDto =
    PhotographerEventSummaryDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        state = deriveEventState(event),
        photoCount = ep.photoCount,
        salesCount = ep.salesCount,
        revenueKept = ep.revenueKeptPhp,
    )

fun detailDto(event: Event, ep: EventPhotographer): PhotographerEventDetailDto =
    PhotographerEventDetailDto(
        id = event.id,
        slug = event.slug,
        name = event.name,
        date = event.date,
        location = event.location,
        state = deriveEventState(event),
        photoCount = ep.photoCount,
        salesCount = ep.salesCount,
        revenueKept = ep.revenueKeptPhp,
        firstUploadAt = ep.firstUploadAt,
        lastUploadAt = ep.lastUploadAt,
    )

fun Photo.toLibraryDto(salesCount: Long): PhotographerLibraryPhotoDto =
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
    )
