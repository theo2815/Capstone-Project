package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// One row serves both phases of a reset (V37): born with codeHash set and
// tokenHash NULL, it rotates into the continuation token on OTP verification.
// confirmReset's findByTokenHash can therefore never match an unverified row.
@Entity
@Table(name = "password_reset_tokens")
class PasswordResetToken(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "user_id", nullable = false)
    val userId: UUID,

    @Column(name = "token_hash", unique = true)
    var tokenHash: String? = null,

    // SHA-256 of the mailed 6-digit code. Non-unique on purpose — a 10^6
    // space collides across users. Nulled when the code is consumed.
    @Column(name = "code_hash")
    var codeHash: String? = null,

    // Failed verify attempts against this code; dead at MAX_OTP_ATTEMPTS.
    @Column(nullable = false)
    var attempts: Int = 0,

    @Column(name = "expires_at", nullable = false)
    var expiresAt: OffsetDateTime,

    @Column(name = "used_at")
    var usedAt: OffsetDateTime? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
) {
    fun isUsable(now: OffsetDateTime = OffsetDateTime.now()): Boolean =
        usedAt == null && expiresAt.isAfter(now)
}
