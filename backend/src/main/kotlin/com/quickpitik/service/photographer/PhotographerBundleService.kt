package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PublicProperties
import com.quickpitik.dto.photographer.PhotographerBundleDto
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.io.OutputStream
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

// Streams a ZIP of every ACTIVE original a photographer uploaded to one event —
// the "Download All Photos" endpoint. Modelled on OrderBundleService: a
// read-only prepare() resolves s3 keys + de-duped entry names into a spec, then
// writeTo() streams the bytes one entry at a time (bounded heap — one original
// on the heap at a time, never the whole event) after the transaction closes.
//
// Serves clean originals only (the with-watermark download was deferred, see
// vault backend/decisions). Ownership is intrinsic: findActiveBundleRows only
// returns this photographer's own photos.
@Service
class PhotographerBundleService(
    private val photoRepository: PhotoRepository,
    private val eventRepository: EventRepository,
    private val storageService: StorageService,
    private val tokenService: PhotographerBundleTokenService,
    private val publicProperties: PublicProperties,
) {
    // Called by the authenticated POST: verify there is something to download
    // (an empty set also means "no coverage" here, since the row query is
    // photographer-scoped) and mint the tokened <a>-navigable URL.
    @Transactional(readOnly = true)
    fun prepareDownloadUrl(photographerId: UUID, eventId: UUID): PhotographerBundleDto {
        eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(ErrorCodes.EVENT_NOT_FOUND, "Event not found")
        if (photoRepository.countByEventIdAndPhotographerIdAndDeletedAtIsNull(eventId, photographerId) == 0L) {
            throw NotFoundException(ErrorCodes.PHOTO_NOT_FOUND, "No photos to download")
        }
        val grant = tokenService.issue(photographerId, eventId)
        val base = publicProperties.apiBaseUrl.trimEnd('/')
        val token = URLEncoder.encode(grant.token, StandardCharsets.UTF_8)
        return PhotographerBundleDto(
            url = "$base/photographer-bundles/$eventId?token=$token",
            expiresAt = grant.expiresAt,
        )
    }

    // Stream side (token already verified by the controller). Split from the
    // byte streaming so the JPA transaction closes before the download starts.
    @Transactional(readOnly = true)
    fun prepare(photographerId: UUID, eventId: UUID): BundleSpec {
        val event = eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(ErrorCodes.PHOTO_NOT_FOUND, "Not found")
        val rows = photoRepository.findActiveBundleRows(eventId, photographerId)
        if (rows.isEmpty()) throw NotFoundException(ErrorCodes.PHOTO_NOT_FOUND, "No photos to download")

        val used = mutableSetOf<String>()
        val entries = rows.map { row ->
            val name = uniqueName(used, buildEntryName(event.slug, row.bib, row.id.toString()))
            used += name
            BundleEntryRef(s3Key = row.s3Key, entryName = name)
        }
        // Single-photo events skip the ZIP wrapper — a lone entry downloads as
        // the raw image. Multi-photo events zip under an event-slug filename.
        val isSingle = entries.size == 1
        val ref = eventId.toString().take(8).lowercase()
        return BundleSpec(
            filename = if (isSingle) entries[0].entryName else "quickpitik-${sanitizeFilename(event.slug)}-$ref.zip",
            contentType = if (isSingle) "image/jpeg" else "application/zip",
            entries = entries,
        )
    }

    fun writeTo(spec: BundleSpec, out: OutputStream) {
        if (spec.entries.size == 1) writeSingleTo(spec, out) else writeZipTo(spec, out)
    }

    private fun writeSingleTo(spec: BundleSpec, out: OutputStream) {
        val entry = spec.entries.first()
        storageService.open(entry.s3Key).use { it.transferTo(out) }
        out.flush()
    }

    private fun writeZipTo(spec: BundleSpec, out: OutputStream) {
        // Finish only when ALL entries were copied. Closing ZipOutputStream
        // on failure would otherwise finalize a deceptively valid partial ZIP.
        // The servlet owns out; release the native deflater on either path.
        val zip = object : ZipOutputStream(out) {
            override fun close() { def.end() }
        }
        zip.use {
            spec.entries.forEach { entry ->
                storageService.open(entry.s3Key).use { original ->
                    zip.putNextEntry(ZipEntry(entry.entryName))
                    original.transferTo(zip)
                    zip.closeEntry()
                }
            }
            zip.finish()
            zip.flush()
        }
    }

    private fun buildEntryName(eventSlug: String, bib: String?, photoId: String): String {
        val tag = bib?.takeIf { it.isNotBlank() }?.let { "bib-${sanitizeFilename(it)}" }
            ?: "untagged-${photoId.take(8)}"
        return "quickpitik-${sanitizeFilename(eventSlug)}-$tag.jpg"
    }

    private fun uniqueName(used: Set<String>, base: String): String {
        if (base !in used) return base
        val dot = base.lastIndexOf('.')
        val stem = if (dot > 0) base.substring(0, dot) else base
        val ext = if (dot > 0) base.substring(dot) else ""
        var i = 2
        while ("$stem-$i$ext" in used) i++
        return "$stem-$i$ext"
    }

    private fun sanitizeFilename(s: String): String =
        s.replace(Regex("[^a-zA-Z0-9._-]"), "-").trim('-').ifBlank { "file" }

    data class BundleEntryRef(val s3Key: String, val entryName: String)

    data class BundleSpec(
        val filename: String,
        val contentType: String,
        val entries: List<BundleEntryRef>,
    )
}
