package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// One row per admin who has started TOTP enrollment. `secretEnc` is the
// AES-256-GCM-encrypted TOTP secret; `enabled` is true only after the admin
// confirms a code. See TotpService / MfaSecretCipher / AdminMfaService.
@Entity
@Table(name = "admin_mfa")
class AdminMfa(
    @Id
    @Column(name = "user_id", nullable = false, updatable = false)
    val userId: UUID,

    @Column(name = "secret_enc", nullable = false, columnDefinition = "TEXT")
    var secretEnc: String,

    @Column(name = "enabled", nullable = false)
    var enabled: Boolean = false,

    @Column(name = "confirmed_at")
    var confirmedAt: OffsetDateTime? = null,

    @Column(name = "last_used_step")
    var lastUsedStep: Long? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),
)
