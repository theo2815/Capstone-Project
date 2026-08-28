package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.PreUpdate
import jakarta.persistence.Table
import org.hibernate.annotations.DynamicUpdate
import java.time.OffsetDateTime
import java.util.UUID

// @DynamicUpdate: see PhotographerSettings.kt for the full rationale. Avatar
// uploads (POST /me/avatar) fire concurrently with /me/photographer/* writes
// that also load this user into their own TX persistence context via
// PhotographerSettingsService.getOrCreate. Without @DynamicUpdate, any TX
// that flushes the loaded User entity would write a full-row UPDATE with its
// stale avatar_s3_key snapshot, clobbering the avatar upload.
@Entity
@Table(name = "users")
@DynamicUpdate
class User(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(nullable = false, unique = true)
    var email: String,

    @Column(name = "password_hash", nullable = false)
    var passwordHash: String,

    @Column(nullable = false)
    var name: String,

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    var role: Role,

    @Column(name = "avatar_url")
    var avatarUrl: String? = null,

    @Column(name = "avatar_s3_key", length = 512)
    var avatarS3Key: String? = null,

    // Google account link (V38). NULL for password-only accounts; set on the
    // first Google sign-in — either at account creation or when a verified
    // Google email auto-links to an existing row. See GoogleAuthService.
    @Column(name = "google_sub", length = 255, unique = true)
    var googleSub: String? = null,

    // When the address on this row was proven reachable (V30). Advisory —
    // nothing gates on it. NULL means "never confirmed", which is the honest
    // reading for every account that predates the flow.
    @Column(name = "email_verified_at")
    var emailVerifiedAt: OffsetDateTime? = null,

    @Column(name = "suspended_at")
    var suspendedAt: OffsetDateTime? = null,

    @Column(name = "suspension_reason", length = 500)
    var suspensionReason: String? = null,

    // Consecutive failed logins since the last success (V29). Reset to 0 both
    // on a successful login and at the moment a lock is applied — the lock
    // itself is the state that matters from then on. Only LoginAttemptService
    // writes these two; see it for why that has to be a separate bean.
    @Column(name = "failed_login_attempts", nullable = false)
    var failedLoginAttempts: Int = 0,

    // When the streak's most recent failure landed (V34). NFR-S-14 counts
    // "5 failures within 15 min" — a failure older than the window restarts
    // the streak at 1 instead of extending it.
    @Column(name = "last_failed_login_at")
    var lastFailedLoginAt: OffsetDateTime? = null,

    @Column(name = "locked_until")
    var lockedUntil: OffsetDateTime? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "updated_at", nullable = false)
    var updatedAt: OffsetDateTime = OffsetDateTime.now(),
) {
    @PreUpdate
    fun onUpdate() {
        updatedAt = OffsetDateTime.now()
    }
}
