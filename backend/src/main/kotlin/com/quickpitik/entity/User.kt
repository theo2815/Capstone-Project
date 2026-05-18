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

    @Column(name = "suspended_at")
    var suspendedAt: OffsetDateTime? = null,

    @Column(name = "suspension_reason", length = 500)
    var suspensionReason: String? = null,

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
