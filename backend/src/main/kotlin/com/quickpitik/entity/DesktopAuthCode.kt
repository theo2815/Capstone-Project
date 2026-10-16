package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// One-time code for connecting the BatchMyPhotos desktop app (V85). Minted by
// the signed-in website, redeemed once by the desktop together with the PKCE
// verifier whose SHA-256 (base64url) is codeChallenge. Mirrors PasswordResetToken.
@Entity
@Table(name = "desktop_auth_codes")
class DesktopAuthCode(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "user_id", nullable = false)
    val userId: UUID,

    @Column(name = "code_hash", nullable = false, unique = true)
    val codeHash: String,

    @Column(name = "code_challenge", nullable = false)
    val codeChallenge: String,

    @Column(name = "expires_at", nullable = false)
    val expiresAt: OffsetDateTime,

    @Column(name = "used_at")
    var usedAt: OffsetDateTime? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime,
) {
    fun isUsable(now: OffsetDateTime): Boolean = usedAt == null && expiresAt.isAfter(now)
}
