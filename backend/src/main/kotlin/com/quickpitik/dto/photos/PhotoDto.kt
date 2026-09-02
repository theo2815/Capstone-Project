package com.quickpitik.dto.photos

import com.quickpitik.entity.Photo
import java.math.BigDecimal
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID

// Mirrors website/src/app/events/[slug]/mock-photos.ts MockPhoto.
//
// `imageUrl` is the watermarked preview every requester sees: QuickPitik credit
// tiles + bottom-left credit caption + the photographer's logo, baked in
// off-request by WatermarkService (async since V36), with an XMP credit packet
// and a pHash registered for /public/photos/verify (V42).
//
// `cleanUrl` is the presigned URL for the unmodified original (photo.s3Key),
// populated only when the requester owns the photo via a valid DownloadGrant.
// Null for everyone else. FE lightbox uses `cleanUrl ?? imageUrl` so a runner
// browsing /events/[slug] sees a clean preview for any photo they already
// bought — closes G-2.
//
// `photographerHandle` / `photographerName` attribute the shot to whoever took
// it, so a runner can tap through from a photo to that photographer's public
// profile at /{handle}. The handle is NULL for a photographer who hasn't been
// verified yet (PhotographerSettings.handle is only set during verification),
// and a null handle means "not linkable" — clients must render the name as
// plain text rather than a dead link to /{null}.
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
    val photographerHandle: String? = null,
    val photographerName: String? = null,
)

// Resolved attribution for one photo's photographer. Callers batch-load these
// per page (see PhotoService.resolvePhotographers) so the resolver below stays
// a map lookup instead of a per-photo query.
data class PhotographerRef(
    val handle: String?,
    val name: String?,
)

private val timeFormatter = DateTimeFormatter.ofPattern("HH:mm")
private val displayZone: ZoneId = ZoneId.of("Asia/Manila")

fun Photo.toDto(
    thumbnailUrlResolver: (Photo) -> String?,
    cleanUrlResolver: (Photo) -> String? = { null },
    photographerResolver: (Photo) -> PhotographerRef? = { null },
): PhotoDto {
    val photographer = photographerResolver(this)
    return PhotoDto(
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
        photographerHandle = photographer?.handle,
        photographerName = photographer?.name,
    )
}
