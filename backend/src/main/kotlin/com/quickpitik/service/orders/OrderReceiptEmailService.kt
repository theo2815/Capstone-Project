package com.quickpitik.service.orders

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.entity.Order
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.service.email.ResendClient
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.time.Clock
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
    private val paymongoProperties: PaymongoProperties,
    private val orderAccessTokenService: OrderAccessTokenService,
    private val transactionTemplate: TransactionTemplate,
    private val clock: Clock,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Two phases. Phase A runs in a transaction: every read, every skip
    // condition, and the conditional claim (a @Modifying UPDATE, so it has to
    // be in one). Phase B runs with no connection held, because ResendClient
    // retries up to 3× with Thread.sleep backoff — holding one of Hikari's ten
    // connections across a Resend outage let a handful of concurrent listeners
    // starve the pool at race finish (2026-09-11 architecture audit, P-3).
    //
    // The template's default REQUIRED propagation stands in for the old
    // REQUIRES_NEW: the only caller is OrderPaidEmailListener, an @Async
    // AFTER_COMMIT listener, so there is never an ambient transaction to join.
    fun sendReceiptIfPending(orderId: UUID) {
        val receipt = transactionTemplate.execute { claimReceipt(orderId) } ?: return

        val response = try {
            resendClient.send(
                ResendSendEmailRequest(
                    from = receipt.sender,
                    to = listOf(receipt.recipient),
                    subject = receipt.subject,
                    html = receipt.html,
                ),
            )
        } catch (ex: Exception) {
            log.error("Receipt send failed · orderId={} err={}", orderId, ex.message, ex)
            // Hand the claim back so a manual reprocess or a future PayMongo
            // webhook re-delivery can try again — the same door the old
            // don't-stamp-on-failure behaviour left open. Needs its own
            // transaction: the claim's committed long before we get here.
            transactionTemplate.executeWithoutResult { orderRepository.releaseReceiptSend(orderId) }
            return
        }

        transactionTemplate.executeWithoutResult {
            orderRepository.stampReceiptDelivered(orderId, OffsetDateTime.now(clock))
        }
        log.info(
            "Receipt sent · orderId={} resendId={} to={}",
            orderId,
            response.id,
            receipt.recipient,
        )
    }

    @Scheduled(fixedDelayString = "\${app.email.resend.receipt-sweep-interval-ms:900000}")
    fun recoverStaleReceiptClaims() {
        val cutoff = OffsetDateTime.now(clock).minusMinutes(STALE_CLAIM_MINUTES)
        val stale = transactionTemplate.execute {
            orderRepository.findStaleReceiptClaimIds(cutoff, PageRequest.of(0, STALE_CLAIM_BATCH_SIZE))
        }.orEmpty()
        stale.forEach { orderId ->
            val released = transactionTemplate.execute {
                orderRepository.releaseStaleReceiptSend(orderId, cutoff)
            } == 1
            if (released) sendReceiptIfPending(orderId)
        }
    }

    // Returns the rendered mail once this caller owns the send slot, or null
    // when there is nothing to send. Rendering stays in here so no detached
    // entity is read outside the transaction; it is string building, not I/O.
    private fun claimReceipt(orderId: UUID): PreparedReceipt? {
        val order = orderRepository.findById(orderId).orElse(null)
        if (order == null) {
            log.warn("Receipt skipped — order {} not found", orderId)
            return null
        }
        if (order.emailSentAt != null) {
            log.info(
                "Receipt already sent for order {} at {} — skipping",
                orderId,
                order.emailSentAt,
            )
            return null
        }
        if (order.recipientEmail.isBlank()) {
            log.warn("Receipt skipped — order {} has no recipient email", orderId)
            return null
        }

        val items = orderItemRepository.findByIdOrderId(order.id)
        if (items.isEmpty()) {
            log.warn("Receipt skipped — order {} has no items", orderId)
            return null
        }
        val grants = downloadGrantRepository.findByIdOrderId(order.id).associateBy { it.id.photoId }
        val entitledCount = items.count { item -> grants[item.id.photoId] != null }
        if (entitledCount == 0) {
            log.warn("Receipt skipped — order {} has no download grants yet", orderId)
            return null
        }

        val event = eventRepository.findById(order.eventId).orElse(null)
        val receiptUrl = buildReceiptUrl(order)

        val html = renderHtml(
            order = order,
            eventName = event?.name ?: "QuickPitik",
            entitledCount = entitledCount,
            ctaUrl = receiptUrl,
            listTotal = items.sumOf { it.pricePhpAtPurchase },
            discountTotal = items.sumOf { it.discountPhp },
        )

        // Claim the send slot BEFORE calling Resend, not after. The early
        // emailSentAt check above is only a cheap fast path — two concurrent
        // webhook retries can both pass it, so this conditional UPDATE is what
        // actually decides who sends. Placed here, after every skip condition,
        // so an order that legitimately isn't ready yet (no grants, no items)
        // never burns its one claim.
        if (orderRepository.claimReceiptSend(orderId, OffsetDateTime.now(clock)) == 0) {
            log.info("Receipt already claimed by a concurrent send for order {} — skipping", orderId)
            return null
        }

        return PreparedReceipt(
            sender = "${resendProperties.fromName} <${resendProperties.fromAddress}>",
            recipient = order.recipientEmail,
            subject = buildSubject(event?.name),
            html = html,
        )
    }

    private data class PreparedReceipt(
        val sender: String,
        val recipient: String,
        val subject: String,
        val html: String,
    )

    private fun buildSubject(eventName: String?): String =
        if (eventName != null) "Your QuickPitik receipt · $eventName"
        else "Your QuickPitik receipt"

    internal fun renderHtml(
        order: Order,
        eventName: String,
        entitledCount: Int,
        ctaUrl: String,
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
            if (entitledCount == 1) "Get your photo  ↓"
            else "Get your $entitledCount photos  ↓"

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
              <a href="${escapeUrl(ctaUrl)}"
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
            Tap the button to open your receipt and save your photos. The link works for 90 days.
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

    // The receipt CTA opens the in-app /orders/return page (the same page +
    // URL shape PayMongo's success_url uses) instead of the raw download-bundle
    // API, so the download runs through the website's polished, mobile-safe
    // save flow. Always carries the token — an email can't rely on an
    // authenticated browser — and uses a long-lived RETURN token so the link
    // survives as long as the order (order.tokenExpiresAt), not the 35-minute
    // PayMongo-bounce TTL.
    private fun buildReceiptUrl(order: Order): String {
        val base = paymongoProperties.successUrl.trimEnd('/')
        val separator = if (base.contains("?")) "&" else "?"
        val token = URLEncoder.encode(
            orderAccessTokenService.issueReceiptReturn(order),
            StandardCharsets.UTF_8,
        )
        return "${base}${separator}orderId=${order.id}&token=$token"
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
        const val STALE_CLAIM_BATCH_SIZE = 50
        const val STALE_CLAIM_MINUTES = 10L
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        val DATE_FORMATTER: DateTimeFormatter = DateTimeFormatter.ofPattern("MMM d · h:mm a")
    }
}
