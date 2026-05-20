package com.quickpitik.dto.photos

import com.quickpitik.entity.Photo
import java.math.BigDecimal
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID

// Mirrors website/src/app/events/[slug]/mock-photos.ts MockPhoto.
//
// `imageUrl` is the watermarked thumbnail every requester sees (photographer
// mark + QuickPitik diagonal tile baked in at upload by WatermarkService).
//
// `cleanUrl` is the presigned URL for the unmodified original (photo.s3Key),
// populated only when the requester owns the photo via a valid DownloadGrant.
// Null for everyone else. FE lightbox uses `cleanUrl ?? imageUrl` so a runner
// browsing /events/[slug] sees a clean preview for any photo they already
// bought — closes G-2.
data class PhotoDto(
    val id: UUID,
    val bib: String?,
    val km: Int?,
    val time: String,
    val tone: Int,
    val span: String,
    val price: BigDecimal,
    val imageUrl: String?,
    val cleanUrl: String? = null,
    val alt: String?,
)

private val timeFormatter = DateTimeFormatter.ofPattern("HH:mm")
private val displayZone: ZoneId = ZoneId.of("Asia/Manila")

fun Photo.toDto(
    thumbnailUrlResolver: (Photo) -> String?,
    cleanUrlResolver: (Photo) -> String? = { null },
): PhotoDto = PhotoDto(
    id = id,
    bib = bibs.minByOrNull { it.bibNumber }?.bibNumber,
    km = km?.toInt(),
    time = (capturedAt ?: uploadedAt).atZoneSameInstant(displayZone).toLocalTime().format(timeFormatter),
    tone = tone,
    span = span.wire,
    price = pricePhp,
    imageUrl = thumbnailUrlResolver(this),
    cleanUrl = cleanUrlResolver(this),
    alt = altText,
)
