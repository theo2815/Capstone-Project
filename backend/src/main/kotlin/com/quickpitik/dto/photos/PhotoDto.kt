package com.quickpitik.dto.photos

import com.quickpitik.entity.Photo
import java.math.BigDecimal
import java.time.format.DateTimeFormatter
import java.util.UUID

// Mirrors website/src/app/events/[slug]/mock-photos.ts MockPhoto.
data class PhotoDto(
    val id: UUID,
    val bib: String?,
    val km: Int?,
    val time: String,
    val tone: Int,
    val span: String,
    val price: BigDecimal,
    val imageUrl: String?,
    val alt: String?,
)

private val timeFormatter = DateTimeFormatter.ofPattern("HH:mm")

fun Photo.toDto(thumbnailUrlResolver: (Photo) -> String?): PhotoDto = PhotoDto(
    id = id,
    bib = bibs.minByOrNull { it.bibNumber }?.bibNumber,
    km = km?.toInt(),
    time = capturedAt?.toLocalTime()?.format(timeFormatter) ?: uploadedAt.toLocalTime().format(timeFormatter),
    tone = tone,
    span = span.wire,
    price = pricePhp,
    imageUrl = thumbnailUrlResolver(this),
    alt = altText,
)
