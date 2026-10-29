package com.quickpitik.service.orders

import com.quickpitik.repository.OrderRepository
import com.quickpitik.service.email.EmailService
import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

// Mails the PayMongo refund link once the admin's save has committed. AFTER_COMMIT so a rolled-back
// save never mails; @Async so a Resend outage never blocks the admin request. A send failure is logged
// and dropped: the runner's inbox message is the durable copy and the admin can send the link again.
// Mirrors PayoutAccountChangedEmailListener.
@Component
class RefundLinkSentEmailListener(
    private val orderRepository: OrderRepository,
    private val emailService: EmailService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("emailSending")
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onRefundLinkSent(event: RefundLinkSentEvent) {
        val order = orderRepository.findById(event.orderId).orElse(null)
        if (order == null) {
            log.warn("Refund-link email skipped · order {} not found", event.orderId)
            return
        }
        emailService.sendRefundLinkEmail(order.recipientEmail, event.url, event.amountPhp, event.validityDays)
    }
}
