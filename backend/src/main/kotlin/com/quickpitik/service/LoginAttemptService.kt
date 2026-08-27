package com.quickpitik.service

import com.quickpitik.config.AuthLockoutProperties
import com.quickpitik.entity.User
import com.quickpitik.repository.UserRepository
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID

/**
 * Consecutive-failure counter behind the temporary account lockout (V29).
 *
 * **Why this is its own bean, and not two methods on [AuthService]:** a failed
 * login ends by throwing, and `AuthService` is annotated `@Transactional` at
 * the class level, so the whole login transaction rolls back — including any
 * counter increment written inside it. A same-class call also can't change
 * that, because Spring's transaction proxy is bypassed on `this`. So the
 * increment has to happen on a separate bean, in a transaction of its own,
 * which is what [Propagation.REQUIRES_NEW] buys. Get this wrong and the
 * feature is a silent no-op: the counter never advances past 1 and nothing
 * ever locks. `AuthLockoutIntegrationTest` is what proves it, since a mock
 * can't observe a rollback.
 */
@Service
class LoginAttemptService(
    private val userRepository: UserRepository,
    private val properties: AuthLockoutProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    /**
     * How long [user] stays locked out, or null if it may sign in.
     *
     * Reads the row already loaded by the caller — no query. An elapsed
     * `lockedUntil` is simply treated as unlocked; it is cleared lazily by the
     * next successful login rather than by a sweeper, because a stale
     * timestamp on a row nobody is authenticating against costs nothing.
     */
    fun lockRemaining(user: User, now: OffsetDateTime = OffsetDateTime.now()): Duration? {
        val until = user.lockedUntil ?: return null
        if (!until.isAfter(now)) return null
        return Duration.between(now, until)
    }

    /**
     * Count one failed password attempt, and lock the account once it reaches
     * `app.auth.lockout.max-attempts`.
     *
     * Runs in its own transaction so it survives the caller's rollback — see
     * the class docblock. Re-loads the user rather than taking the caller's
     * instance for the same reason: that instance belongs to a persistence
     * context that is about to be discarded.
     */
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    fun recordFailure(userId: UUID) {
        val user = userRepository.findById(userId).orElse(null) ?: return
        val now = OffsetDateTime.now()
        // NFR-S-14 window (V34): a streak whose last failure is older than the
        // window restarts at 1 — only failures within `window` of each other
        // accumulate toward a lock.
        val staleStreak = user.lastFailedLoginAt?.isBefore(now.minus(properties.window)) == true
        val attempts = if (staleStreak) 1 else user.failedLoginAttempts + 1
        user.lastFailedLoginAt = now
        if (attempts >= properties.maxAttempts) {
            // Reset the counter as the lock goes on: from here the lock is the
            // state that matters, and leaving the counter at the threshold
            // would re-lock on the very first mistake after it expires.
            user.failedLoginAttempts = 0
            user.lockedUntil = OffsetDateTime.now().plus(properties.duration)
            log.warn("Account locked after {} failed logins · userId={}", attempts, userId)
        } else {
            user.failedLoginAttempts = attempts
        }
        userRepository.save(user)
    }

    /**
     * Clear the streak after a successful sign-in.
     *
     * Also in its own transaction, so a later failure in the login path can't
     * roll the clear back and leave a user one mistake from a lock they had
     * already earned their way out of. The guard keeps the common case — a
     * clean login on a clean account — free of a pointless UPDATE.
     */
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    fun recordSuccess(userId: UUID) {
        val user = userRepository.findById(userId).orElse(null) ?: return
        if (user.failedLoginAttempts == 0 && user.lockedUntil == null && user.lastFailedLoginAt == null) return
        user.failedLoginAttempts = 0
        user.lastFailedLoginAt = null
        user.lockedUntil = null
        userRepository.save(user)
    }
}
