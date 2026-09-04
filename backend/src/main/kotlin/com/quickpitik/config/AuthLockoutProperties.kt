package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties
import java.time.Duration

/**
 * Temporary lockout after consecutive failed logins (V29).
 *
 * Complements the per-IP `app.rate-limit.auth-login` bucket rather than
 * replacing it: that one caps how fast a single host may guess, this one caps
 * how many times a single *account* may be guessed at, no matter how many hosts
 * the attempts come from. It is also unconditional, whereas the bucket is inert
 * while `RATE_LIMIT_ENABLED=false` (the dev default).
 *
 * There is intentionally no `enabled` flag. A lock auto-clears after
 * [duration], and raising [maxAttempts] is the escape hatch if a demo needs
 * one — a second kill switch for a self-healing 15-minute state would be
 * configuration for its own sake.
 */
@ConfigurationProperties(prefix = "app.auth.lockout")
data class AuthLockoutProperties(
    val maxAttempts: Int = 5,
    val duration: Duration = Duration.ofMinutes(15),
    // NFR-S-14: only failures within this window count as one streak. A
    // failure older than [window] restarts the counter at 1 — five typos
    // spread over a month is not an attack.
    val window: Duration = Duration.ofMinutes(15),
)
