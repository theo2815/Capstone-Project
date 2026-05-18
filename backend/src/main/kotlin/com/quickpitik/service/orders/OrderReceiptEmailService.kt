package com.quickpitik.service.orders

import com.quickpitik.config.ResendProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.entity.Order
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.email.ResendClient
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID

// Builds and sends the post-payment receipt email. Called from
// OrderPaidEmailListener (AFTER_COMMIT, @Async) so a slow Resend request
// doesn't block the PayMongo webhook response (PayMongo retries on >30s
// silence).
//
// Idempotency: orders.email_sent_at is the source of truth. The listener
// short-circuits if it's already stamped. On a failed Resend call we DON'T
// stamp it — PayMongo's webhook retry (if it happens) will re-fire the
// AFTER_COMMIT listener and we'll re-try the send.
//
// Each order produces one email. Multi-event carts produce N receipts —
// clean attribution per event matches the multi-Order split.
@Service
class OrderReceiptEmailService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val photoRepository: PhotoRepository,
    private val eventRepository: EventRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val resendClient: ResendClient,
    private val resendProperties: ResendProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // New TX — this is invoked from a @TransactionalEventListener after the
    // outer webhook TX has already committed, so any reads must run under
    // their own transaction context.
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    fun sendReceiptIfPending(orderId: UUID) {
        val order = orderRepository.findById(orderId).orElse(null)
        if (order == null) {
            log.warn("Receipt skipped — order {} not found", orderId)
            return
        }
        if (order.emailSentAt != null) {
            log.info(
                "Receipt already sent for order {} at {} — skipping",
                orderId,
                order.emailSentAt,
            )
            return
        }
        if (order.recipientEmail.isBlank()) {
            log.warn("Receipt skipped — order {} has no recipient email", orderId)
            return
        }

        val items = orderItemRepository.findByIdOrderId(order.id)
        if (items.isEmpty()) {
            log.warn("Receipt skipped — order {} has no items", orderId)
            return
        }
        val photoIds = items.map { it.id.photoId }
        val photos = photoRepository.findAllById(photoIds).associateBy { it.id }
        val event = eventRepository.findById(order.eventId).orElse(null)
        val grants = downloadGrantRepository.findByIdOrderId(order.id).associateBy { it.id.photoId }

        val eventSlug = event?.slug ?: "quickpitik"
        val downloads = items.mapNotNull { item ->
            val photo = photos[item.id.photoId] ?: return@mapNotNull null
            // No grant ⇒ this order isn't actually entitled yet. Skip the row.
            grants[item.id.photoId] ?: return@mapNotNull null
            val bib = photo.bibs.minByOrNull { it.bibNumber }?.bibNumber
            val baseUrl = storageService.presignedGetUrl(
                photo.s3Key,
                storageProperties.presignedTtl.runnerDownload,
            )
            val filename = buildDownloadFilename(eventSlug, bib, photo.id.toString())
            ReceiptDownloadLink(
                bib = bib,
                url = appendDownloadDisposition(baseUrl, filename),
            )
        }
        if (downloads.isEmpty()) {
            log.warn("Receipt skipped — order {} has no download grants yet", orderId)
            return
        }

        val html = renderHtml(
            order = order,
            eventName = event?.name ?: "QuickPitik",
            downloads = downloads,
        )
        val sender = "${resendProperties.fromName} <${resendProperties.fromAddress}>"
        val subject = buildSubject(event?.name)

        try {
            val response = resendClient.send(
                ResendSendEmailRequest(
                    from = sender,
                    to = listOf(order.recipientEmail),
                    subject = subject,
                    html = html,
                ),
            )
            order.emailSentAt = OffsetDateTime.now()
            orderRepository.save(order)
            log.info(
                "Receipt sent · orderId={} resendId={} to={}",
                orderId,
                response.id,
                order.recipientEmail,
            )
        } catch (ex: Exception) {
            log.error("Receipt send failed · orderId={} err={}", orderId, ex.message, ex)
            // Intentionally do NOT stamp email_sent_at — leaves the door open
            // for a manual reprocess or a future PayMongo webhook re-delivery
            // to try again.
        }
    }

    private fun buildSubject(eventName: String?): String =
        if (eventName != null) "Your QuickPitik receipt · $eventName"
        else "Your QuickPitik receipt"

    private fun renderHtml(
        order: Order,
        eventName: String,
        downloads: List<ReceiptDownloadLink>,
    ): String {
        val ref = order.id.toString().take(8).uppercase()
        val paidAt = (order.paidAt ?: order.createdAt)
            .atZoneSameInstant(DISPLAY_ZONE)
            .format(DATE_FORMATTER)
        val photoWord = if (downloads.size == 1) "photo" else "photos"
        val totalPhp = "₱${order.totalPhp.toPlainString()}"

        val buttonsHtml = downloads.joinToString("") { dl ->
            val label = dl.bib?.let { "BIB $it" } ?: "Untagged"
            """
            <tr><td style="padding:6px 0;">
              <a href="${escapeUrl(dl.url)}" download
                 style="display:inline-block;background:#3b8c5f;color:#fafaf6;
                        text-decoration:none;padding:14px 28px;border-radius:999px;
                        font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;
                        text-transform:uppercase;letter-spacing:0.18em;font-size:12px;
                        font-weight:500;">
                Download · $label  ↓
              </a>
            </td></tr>
            """.trimIndent()
        }

        return """
<!DOCTYPE html>
<html><head><meta charset="utf-8" /><title>QuickPitik receipt</title></head>
<body style="margin:0;padding:0;background:#f7f5ee;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1a1a1a;">
  <table cellspacing="0" cellpadding="0" border="0" width="100%" style="background:#f7f5ee;padding:32px 16px;">
    <tr><td align="center">
      <table cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width:520px;margin:0 auto;background:#fafaf6;border-radius:16px;padding:36px 32px;border:1px solid #e5e2d8;">
        <tr><td>
          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:11px;color:#7a7a7a;margin:0 0 10px;">
            QuickPitik · Payment received
          </p>
          <h1 style="font-family:Georgia,'Times New Roman',serif;font-size:32px;font-weight:500;margin:0 0 18px;letter-spacing:-0.01em;line-height:1.05;color:#1a1a1a;">
            All yours.
          </h1>
          <p style="font-size:15px;line-height:1.6;color:#3a3a3a;margin:0 0 28px;">
            You bought <strong style="font-variant-numeric:tabular-nums;">${downloads.size}</strong> $photoWord from
            <strong>${escapeHtml(eventName)}</strong> for
            <strong style="font-variant-numeric:tabular-nums;">$totalPhp</strong>.
          </p>

          <table cellspacing="0" cellpadding="0" border="0" style="margin:0 0 28px;">
            $buttonsHtml
          </table>

          <hr style="border:none;border-top:1px solid #e5e2d8;margin:28px 0;" />

          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.25em;font-size:11px;color:#7a7a7a;margin:0 0 6px;">
            Reference · <span style="color:#1a1a1a;font-variant-numeric:tabular-nums;">$ref</span>
          </p>
          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.25em;font-size:11px;color:#7a7a7a;margin:0;">
            Paid · <span style="color:#1a1a1a;">${escapeHtml(paidAt)} PHT</span>
          </p>

          <p style="font-size:13px;line-height:1.65;color:#7a7a7a;margin:28px 0 0;">
            The download links above work for 7 days. After that, reply to this email and we&rsquo;ll send you fresh ones.
          </p>
        </td></tr>
      </table>
      <p style="text-align:center;font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:10px;color:#a8a8a8;margin:24px 0 0;">
        QuickPitik · Cebu, Philippines
      </p>
    </td></tr>
  </table>
</body></html>
        """.trimIndent()
    }

    // Friendly filename that lands in the user's downloads folder.
    // Pattern: quickpitik-<event-slug>-<bib-or-photo-prefix>.jpg
    private fun buildDownloadFilename(eventSlug: String, bib: String?, photoId: String): String {
        val tag = bib?.takeIf { it.isNotBlank() }?.let { "bib-$it" }
            ?: "untagged-${photoId.take(8)}"
        return "quickpitik-$eventSlug-$tag.jpg"
    }

    // Append disposition+filename hints to a presigned URL. Works with the
    // existing `?expires=…&method=…` query string already on the URL.
    // StorageDownloadDispositionFilter reads these and stamps the
    // Content-Disposition response header so browsers always download
    // instead of displaying inline.
    private fun appendDownloadDisposition(url: String, filename: String): String {
        val sep = if (url.contains("?")) "&" else "?"
        val encodedName = java.net.URLEncoder.encode(filename, Charsets.UTF_8).replace("+", "%20")
        return "$url${sep}disposition=attachment&filename=$encodedName"
    }

    private fun escapeHtml(s: String): String = s
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")

    // Presigned URLs already URL-encoded by S3/storage; only `&` needs to
    // be `&amp;` to be valid inside an href attribute in HTML email.
    private fun escapeUrl(s: String): String = s.replace("&", "&amp;")

    private data class ReceiptDownloadLink(
        val bib: String?,
        val url: String,
    )

    private companion object {
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        val DATE_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("MMM d · h:mm a")
    }
}
