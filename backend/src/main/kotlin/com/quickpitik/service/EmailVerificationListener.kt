package com.quickpitik.service

import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Async
import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener

/**
 * Mails the verification link once the registration has actually committed.
 *
 * Two properties this buys, both of which a direct call from
 * [AuthService.register] would lose:
 *
 *  - **AFTER_COMMIT** — a registration that rolls back (duplicate email losing
 *    the race to the users.email UNIQUE, say) never mails a link for a row that
 *    does not exist.
 *  - **@Async** — registration returns at its own speed. `ResendClient` allows a
 *    15-second read timeout with retries, so an inline send would put a Resend
 *    outage directly into the sign-up latency, or fail the sign-up outright.
 *
 * A send failure is logged and dropped: the account is real and usable either
 * way, since verification is advisory. `POST /auth/resend-verification` is the
 * recovery path, which is why it exists.
 *
 * Mirrors [com.quickpitik.service.orders.OrderPaidEmailListener].
 */
@Component
class EmailVerificationListener(
    private val emailVerificationService: EmailVerificationService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onUserRegistered(event: UserRegisteredEvent) {
        try {
            emailVerificationService.sendVerification(event.userId)
        } catch (ex: Exception) {
            log.error(
                "Verification email dispatch failed · userId={}: {}",
                event.userId,
                ex.message,
                ex,
            )
        }
    }
}
