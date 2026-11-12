package com.quickpitik.service.admin

import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

// PE-2.2 (P7 default): mails the photographer once an admin's payout hold has committed.
// The inbox note (same transaction as the hold) and the balance screen tell them even when Resend is down.
// shortcut: a failed send is logged and dropped; upgrade to a claimed send with a retry sweep if the owner picks P7's alternative.
@Component
class PayoutHoldEmailListener(
    private val userRepository: UserRepository,
    private val emailService: EmailService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("emailSending")
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onPayoutHoldPlaced(event: PayoutHoldPlacedEvent) {
        val user = userRepository.findById(event.photographerId).orElse(null)
        if (user == null) {
            log.warn("Payout hold alert skipped · user {} not found", event.photographerId)
            return
        }
        emailService.sendPayoutHoldAlert(toEmail = user.email, name = user.name)
    }
}
