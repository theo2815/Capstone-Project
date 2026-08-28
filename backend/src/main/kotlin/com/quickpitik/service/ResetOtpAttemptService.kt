package com.quickpitik.service

import com.quickpitik.repository.PasswordResetTokenRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

/**
 * Failed-attempt counter behind the password-reset OTP cap (V37).
 *
 * **Why this is its own bean, and not a method on [PasswordResetService]:** a
 * wrong code ends by throwing, and `PasswordResetService` is annotated
 * `@Transactional` at the class level, so the whole verify transaction rolls
 * back — including any counter increment written inside it. A same-class call
 * also can't change that, because Spring's transaction proxy is bypassed on
 * `this`. So the increment has to happen on a separate bean, in a transaction
 * of its own, which is what [Propagation.REQUIRES_NEW] buys. Get this wrong
 * and the cap is a silent no-op: the counter never advances and a 6-digit
 * code can be brute-forced within the IP budget. Same shape and rationale as
 * [LoginAttemptService]; like there, only an integration test can observe the
 * rollback survival.
 */
@Service
class ResetOtpAttemptService(
    private val repository: PasswordResetTokenRepository,
) {
    // Re-loads the row rather than taking the caller's instance: that instance
    // belongs to a persistence context that is about to be discarded.
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    fun recordFailure(tokenId: UUID) {
        val row = repository.findById(tokenId).orElse(null) ?: return
        row.attempts += 1
        repository.save(row)
    }
}
