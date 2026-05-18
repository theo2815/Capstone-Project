package com.quickpitik.service.orders

import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

// AFTER_COMMIT receipt dispatcher. Async so a slow Resend call doesn't
// block whatever published the event (PayMongo webhook handler — PayMongo
// retries on >30s ack silence, so we must respond fast).
//
// `@Async` rides Spring's default task executor (see AsyncConfig). On
// Resend failure we log and move on; no automatic retry — the next
// PayMongo webhook re-delivery (if any) will re-publish the event and
// OrderReceiptEmailService.sendReceiptIfPending will try again because
// email_sent_at stays NULL on failure.
@Component
class OrderPaidEmailListener(
    private val orderReceiptEmailService: OrderReceiptEmailService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onOrderPaid(event: OrderPaidEvent) {
        try {
            orderReceiptEmailService.sendReceiptIfPending(event.orderId)
        } catch (ex: Exception) {
            log.error(
                "OrderPaidEmailListener uncaught failure · orderId={} cs={}: {}",
                event.orderId,
                event.checkoutSessionId,
                ex.message,
                ex,
            )
        }
    }
}
