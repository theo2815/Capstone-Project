package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// Proof-of-inbox token minted at registration (V30). Deliberately carries no
// address of its own — unlike EmailChangeToken, which parks a *pending* address,
// this one confirms the address already on users.email, so redemption has
// nothing to move.
@Entity
@Table(name = "email_verification_tokens")
class EmailVerificationToken(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "user_id", nullable = false)
    val userId: UUID,

    @Column(name = "token_hash", nullable = false, unique = true)
    val tokenHash: String,

    @Column(name = "expires_at", nullable = false)
    val expiresAt: OffsetDateTime,

    @Column(name = "used_at")
    var usedAt: OffsetDateTime? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
) {
    fun isUsable(now: OffsetDateTime = OffsetDateTime.now()): Boolean =
        usedAt == null && expiresAt.isAfter(now)
}
