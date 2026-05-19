package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.OrderStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.io.OutputStream
import java.time.OffsetDateTime
import java.util.UUID
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Streams a single ZIP containing every entitled photo in an order.
 *
 * Closes R-1 (per [[ROLE-STATUS]]): replaces the FE-side per-photo iteration
 * loop on /orders/return — Chromium throttles multi-download, mobile Safari
 * drops everything past the first <a download> click, and JSZip in-browser
 * crashes phones on big orders. Server-side ZIP works everywhere.
 *
 * Auth is token-only (not JWT) because the endpoint must be reachable by a
 * top-level <a href> navigation, which carries no Authorization header. The
 * shareToken is unique per order, minted at create time for runners AND
 * guests (V18 backfilled legacy runner orders) — sharing the URL is the
 * same semantic as sharing a Dropbox link.
 *
 * Split into two phases so the JPA transaction closes before the bytes start
 * streaming: `prepare()` runs inside @Transactional and resolves everything
 * the writer needs (s3 keys + filenames); `writeZipTo()` is plain I/O against
 * the storage backend and the response stream. Keeps Hibernate sessions
 * short and lets the bundle stream regardless of TX timeout.
 */
@Service
class OrderBundleService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val photoRepository: PhotoRepository,
    private val eventRepository: EventRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val storageService: StorageService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(readOnly = true)
    fun prepare(orderId: UUID, token: String?): BundleSpec {
        val order = orderRepository.findById(orderId).orElseThrow { orderNotFound() }

        // Anti-IDOR: every failure surfaces NOT_FOUND so an attacker probing
        // by id learns nothing about which ids exist.
        if (token.isNullOrBlank() || order.shareToken == null || order.shareToken != token) {
            throw orderNotFound()
        }
        if (order.status != OrderStatus.PAID && order.status != OrderStatus.FULFILLED) {
            throw orderNotFound()
        }

        val items = orderItemRepository.findByIdOrderId(orderId)
        if (items.isEmpty()) throw orderNotFound()

        val photos = photoRepository.findAllById(items.map { it.id.photoId }).associateBy { it.id }
        val grants = downloadGrantRepository.findByIdOrderId(orderId).associateBy { it.id.photoId }
        val event = eventRepository.findById(order.eventId).orElse(null)
        val eventSlug = event?.slug ?: "quickpitik"

        val now = OffsetDateTime.now()
        val usedNames = mutableSetOf<String>()
        val entries = items.mapNotNull { item ->
            val photo = photos[item.id.photoId] ?: return@mapNotNull null
            val grant = grants[item.id.photoId] ?: return@mapNotNull null
            if (grant.grantedUntil.isBefore(now)) return@mapNotNull null
            val bib = photo.bibs.minByOrNull { it.bibNumber }?.bibNumber
            val name = uniqueName(usedNames, buildEntryName(eventSlug, bib, photo.id.toString()))
            usedNames.add(name)
            BundleEntryRef(s3Key = photo.s3Key, entryName = name)
        }
        if (entries.isEmpty()) throw orderNotFound()

        // Single-photo orders skip the ZIP wrapper: the entry's own filename
        // becomes the response filename and the body is the raw image bytes.
        // Multi-photo orders fall through to a ZIP whose name encodes the
        // event slug + the 8-char order ref. Callers don't branch on URL —
        // contentType + filename + writeTo() dispatch handles it.
        val isSingle = entries.size == 1
        val ref = order.id.toString().take(8).lowercase()
        val filename =
            if (isSingle) entries[0].entryName
            else "quickpitik-${sanitizeFilename(eventSlug)}-$ref.zip"
        val contentType = if (isSingle) "image/jpeg" else "application/zip"
        return BundleSpec(
            orderId = order.id,
            filename = filename,
            contentType = contentType,
            entries = entries,
        )
    }

    // Dispatcher used by the controller. Single-photo orders pass through the
    // photo bytes; multi-photo orders stream a ZIP. The split keeps callers
    // ignorant of the wrapping decision.
    fun writeTo(spec: BundleSpec, out: OutputStream) {
        if (spec.entries.size == 1) writeSingleTo(spec, out) else writeZipTo(spec, out)
    }

    private fun writeSingleTo(spec: BundleSpec, out: OutputStream) {
        val entry = spec.entries.first()
        try {
            val bytes = storageService.getBytes(entry.s3Key)
            out.write(bytes)
            out.flush()
        } catch (ex: Exception) {
            // No partial fallback — a single-photo order with one failing
            // read has nothing to give the caller. Surface as a write
            // failure so the response stream closes and the client retries.
            log.warn(
                "Failed to stream single photo {} for order {}: {}",
                entry.s3Key, spec.orderId, ex.message,
            )
            throw ex
        }
    }

    private fun writeZipTo(spec: BundleSpec, out: OutputStream) {
        ZipOutputStream(out).use { zip ->
            spec.entries.forEach { entry ->
                try {
                    val bytes = storageService.getBytes(entry.s3Key)
                    zip.putNextEntry(ZipEntry(entry.entryName))
                    zip.write(bytes)
                    zip.closeEntry()
                } catch (ex: Exception) {
                    // Per-photo failure is non-fatal — log and continue so
                    // the rest of the order still bundles. A half-written
                    // entry from a failed copy isn't closed here; the next
                    // putNextEntry will roll over cleanly.
                    log.warn(
                        "Skipping {} in bundle {}: {}",
                        entry.s3Key, spec.orderId, ex.message,
                    )
                }
            }
            zip.finish()
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

    // Defensive: strip anything that's not safe inside a ZIP entry path or a
    // download filename. Bibs are runner-input on a future event; event slugs
    // come from admin input — both flow through here.
    private fun sanitizeFilename(s: String): String =
        s.replace(Regex("[^a-zA-Z0-9._-]"), "-").trim('-').ifBlank { "file" }

    private fun orderNotFound(): NotFoundException =
        NotFoundException(code = ErrorCodes.ORDER_NOT_FOUND, message = "Order not found")
}

data class BundleEntryRef(
    val s3Key: String,
    val entryName: String,
)

data class BundleSpec(
    val orderId: UUID,
    val filename: String,
    val contentType: String,
    val entries: List<BundleEntryRef>,
)
