package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// Pending change of a user's sign-in email. The requested address lives here
// until the token is redeemed — users.email is untouched while a request is
// outstanding, so an unconfirmed (or hostile) request can never lock the owner
// out of their own account.
@Entity
@Table(name = "email_change_tokens")
class EmailChangeToken(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "user_id", nullable = false)
    val userId: UUID,

    @Column(name = "new_email", nullable = false)
    val newEmail: String,

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
