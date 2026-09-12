package com.quickpitik.service.photographer

import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

// Mails the payout-account-change security alert once the change has committed.
// AFTER_COMMIT so a rolled-back change never alerts; @Async so a Resend outage
// never blocks the settings request. A send failure is logged and dropped — the
// audit row + 24h cooldown are the hard controls, this is the actionable heads-up.
// Mirrors EmailVerificationListener / OrderPaidEmailListener.
@Component
class PayoutAccountChangedEmailListener(
    private val userRepository: UserRepository,
    private val emailService: EmailService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("emailSending")
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onPayoutAccountChanged(event: PayoutAccountChangedEvent) {
        val user = userRepository.findById(event.userId).orElse(null)
        if (user == null) {
            log.warn("Payout-account-change alert skipped · user {} not found", event.userId)
            return
        }
        emailService.sendPayoutAccountChangedAlert(
            toEmail = user.email,
            name = user.name,
            action = event.action,
            method = event.method,
            maskedAccount = event.maskedAccount,
        )
    }
}
