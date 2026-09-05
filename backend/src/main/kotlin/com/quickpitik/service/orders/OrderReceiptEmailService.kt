package com.quickpitik.service.orders

import com.quickpitik.config.PublicProperties
import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.entity.Order
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.service.email.ResendClient
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID

// Builds and sends the post-payment receipt email. Called from
// OrderPaidEmailListener (AFTER_COMMIT, @Async) so a slow Resend request
// doesn't block the PayMongo webhook response (PayMongo retries on >30s
// silence).
//
// Idempotency: orders.email_sent_at is the source of truth, claimed with a
// conditional UPDATE (claimReceiptSend) rather than a read-check-write, so two
// concurrent webhook retries can't both decide to send. The loser skips. On a
// failed Resend call the claim is released, so PayMongo's webhook retry (if it
// happens) re-fires the AFTER_COMMIT listener and the send is re-tried.
//
// Each order produces one email. Multi-event carts produce N receipts —
// clean attribution per event matches the multi-Order split.
@Service
class OrderReceiptEmailService(
    private val orderRepository: OrderRepository,
    private val orderItemRepository: OrderItemRepository,
    private val eventRepository: EventRepository,
    private val downloadGrantRepository: DownloadGrantRepository,
    private val resendClient: ResendClient,
    private val resendProperties: ResendProperties,
    private val publicProperties: PublicProperties,
    private val orderAccessTokenService: OrderAccessTokenService,
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
        val grants = downloadGrantRepository.findByIdOrderId(order.id).associateBy { it.id.photoId }
        val entitledCount = items.count { item -> grants[item.id.photoId] != null }
        if (entitledCount == 0) {
            log.warn("Receipt skipped — order {} has no download grants yet", orderId)
            return
        }

        val event = eventRepository.findById(order.eventId).orElse(null)
        val bundleUrl = buildBundleUrl(
            orderId,
            orderAccessTokenService.issue(order, OrderCapability.BUNDLE),
        )

        val html = renderHtml(
            order = order,
            eventName = event?.name ?: "QuickPitik",
            entitledCount = entitledCount,
            bundleUrl = bundleUrl,
            listTotal = items.sumOf { it.pricePhpAtPurchase },
            discountTotal = items.sumOf { it.discountPhp },
        )
        val sender = "${resendProperties.fromName} <${resendProperties.fromAddress}>"
        val subject = buildSubject(event?.name)
        val recipient = order.recipientEmail

        // Claim the send slot BEFORE calling Resend, not after. The early
        // emailSentAt check above is only a cheap fast path — two concurrent
        // webhook retries can both pass it, so this conditional UPDATE is what
        // actually decides who sends. Placed here, after every skip condition,
        // so an order that legitimately isn't ready yet (no grants, no items)
        // never burns its one claim.
        if (orderRepository.claimReceiptSend(orderId, OffsetDateTime.now()) == 0) {
            log.info("Receipt already claimed by a concurrent send for order {} — skipping", orderId)
            return
        }

        try {
            val response = resendClient.send(
                ResendSendEmailRequest(
                    from = sender,
                    to = listOf(recipient),
                    subject = subject,
                    html = html,
                ),
            )
            log.info(
                "Receipt sent · orderId={} resendId={} to={}",
                orderId,
                response.id,
                recipient,
            )
        } catch (ex: Exception) {
            log.error("Receipt send failed · orderId={} err={}", orderId, ex.message, ex)
            // Hand the claim back so a manual reprocess or a future PayMongo
            // webhook re-delivery can try again — the same door the old
            // don't-stamp-on-failure behaviour left open.
            orderRepository.releaseReceiptSend(orderId)
        }
    }

    private fun buildSubject(eventName: String?): String =
        if (eventName != null) "Your QuickPitik receipt · $eventName"
        else "Your QuickPitik receipt"

    internal fun renderHtml(
        order: Order,
        eventName: String,
        entitledCount: Int,
        bundleUrl: String,
        listTotal: BigDecimal = order.totalPhp,
        discountTotal: BigDecimal = BigDecimal.ZERO,
    ): String {
        val ref = order.id.toString().take(8).uppercase()
        // A ₱0 order (100% giveaway) confirms without a payment; the receipt
        // says so and shows the list price it waived.
        val free = order.totalPhp.signum() == 0
        val kicker = if (free) "Order confirmed" else "Payment received"
        val paidLabel = if (free) "Confirmed" else "Paid"
        val discountLine = if (discountTotal.signum() > 0) {
            val code = order.couponCode?.let(::escapeHtml) ?: "Photographer discount"
            val charged = if (free) " Nothing was charged." else ""
            """
          <p style="font-size:14px;line-height:1.6;color:#3a3a3a;margin:-20px 0 28px;font-variant-numeric:tabular-nums;">
            List ₱${listTotal.toPlainString()} · $code −₱${discountTotal.toPlainString()} · Total ₱${order.totalPhp.toPlainString()}.$charged
          </p>"""
        } else {
            ""
        }
        val paidAt = (order.paidAt ?: order.createdAt)
            .atZoneSameInstant(DISPLAY_ZONE)
            .format(DATE_FORMATTER)
        val photoWord = if (entitledCount == 1) "photo" else "photos"
        val totalPhp = "₱${order.totalPhp.toPlainString()}"
        val buttonLabel =
            if (entitledCount == 1) "Download your photo  ↓"
            else "Download all $entitledCount photos (.zip)  ↓"

        return """
<!DOCTYPE html>
<html><head><meta charset="utf-8" /><title>QuickPitik receipt</title></head>
<body style="margin:0;padding:0;background:#f7f5ee;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1a1a1a;">
  <table cellspacing="0" cellpadding="0" border="0" width="100%" style="background:#f7f5ee;padding:32px 16px;">
    <tr><td align="center">
      <table cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width:520px;margin:0 auto;background:#fafaf6;border-radius:16px;padding:36px 32px;border:1px solid #e5e2d8;">
        <tr><td>
          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:11px;color:#7a7a7a;margin:0 0 10px;">
            QuickPitik · $kicker
          </p>
          <h1 style="font-family:Georgia,'Times New Roman',serif;font-size:32px;font-weight:500;margin:0 0 18px;letter-spacing:-0.01em;line-height:1.05;color:#1a1a1a;">
            All yours.
          </h1>
          <p style="font-size:15px;line-height:1.6;color:#3a3a3a;margin:0 0 28px;">
            You bought <strong style="font-variant-numeric:tabular-nums;">$entitledCount</strong> $photoWord from
            <strong>${escapeHtml(eventName)}</strong> for
            <strong style="font-variant-numeric:tabular-nums;">$totalPhp</strong>.
          </p>
$discountLine
          <table cellspacing="0" cellpadding="0" border="0" style="margin:0 0 28px;">
            <tr><td style="padding:6px 0;">
              <a href="${escapeUrl(bundleUrl)}" download
                 style="display:inline-block;background:#3b8c5f;color:#fafaf6;
                        text-decoration:none;padding:16px 32px;border-radius:999px;
                        font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;
                        text-transform:uppercase;letter-spacing:0.2em;font-size:13px;
                        font-weight:500;">
                $buttonLabel
              </a>
            </td></tr>
          </table>

          <hr style="border:none;border-top:1px solid #e5e2d8;margin:28px 0;" />

          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.25em;font-size:11px;color:#7a7a7a;margin:0 0 6px;">
            Reference · <span style="color:#1a1a1a;font-variant-numeric:tabular-nums;">$ref</span>
          </p>
          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.25em;font-size:11px;color:#7a7a7a;margin:0;">
            $paidLabel · <span style="color:#1a1a1a;">${escapeHtml(paidAt)} PHT</span>
          </p>

          <p style="font-size:13px;line-height:1.65;color:#7a7a7a;margin:28px 0 0;">
            Tap the button to save the ZIP to your device. iOS Files and Android both unzip on tap. The link works as long as you keep this email.
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

    private fun buildBundleUrl(orderId: UUID, shareToken: String): String {
        val base = publicProperties.apiBaseUrl.trimEnd('/')
        val token = URLEncoder.encode(shareToken, StandardCharsets.UTF_8)
        return "$base/orders/$orderId/download-bundle?token=$token"
    }

    private fun escapeHtml(s: String): String = s
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")

    // Bundle URL is unencoded; only `&` needs to become `&amp;` to be valid
    // inside an href attribute in HTML email.
    private fun escapeUrl(s: String): String = s.replace("&", "&amp;")

    private companion object {
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        val DATE_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("MMM d · h:mm a")
    }
}
