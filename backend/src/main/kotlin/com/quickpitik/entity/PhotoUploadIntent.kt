package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.DynamicUpdate
import java.time.OffsetDateTime
import java.util.UUID

@Entity
@Table(name = "photo_upload_intents")
@DynamicUpdate
class PhotoUploadIntent(
    @Id
    @Column(name = "photo_id", nullable = false, updatable = false)
    val photoId: UUID = UUID.randomUUID(),

    @Column(name = "photographer_id", nullable = false, updatable = false)
    val photographerId: UUID,

    @Column(name = "event_id", nullable = false, updatable = false)
    val eventId: UUID,

    @Column(name = "request_id", updatable = false)
    val requestId: UUID? = null,

    @Column(name = "content_hash", nullable = false, length = 64, updatable = false)
    val contentHash: String,

    @Column(name = "content_type", nullable = false, length = 32, updatable = false)
    val contentType: String,

    @Column(name = "size_bytes", nullable = false, updatable = false)
    val sizeBytes: Long,

    @Column(name = "staging_key", nullable = false, length = 512, updatable = false)
    val stagingKey: String,

    @Column(name = "original_key", nullable = false, length = 512, updatable = false)
    val originalKey: String,

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    var state: PhotoUploadIntentState = PhotoUploadIntentState.PENDING,

    @Column(name = "lease_token")
    var leaseToken: UUID? = null,

    @Column(name = "lease_until")
    var leaseUntil: OffsetDateTime? = null,

    @Column(name = "result_photo_id")
    var resultPhotoId: UUID? = null,

    @Column(name = "error_code", length = 64)
    var errorCode: String? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "put_expires_at", nullable = false, updatable = false)
    val putExpiresAt: OffsetDateTime,

    @Column(name = "expires_at", nullable = false, updatable = false)
    val expiresAt: OffsetDateTime,

    @Column(name = "updated_at", nullable = false)
    var updatedAt: OffsetDateTime = createdAt,

    @Column(name = "cleanup_after")
    var cleanupAfter: OffsetDateTime? = null,

    @Column(name = "cleanup_checked_at")
    var cleanupCheckedAt: OffsetDateTime? = null,

    @Column(name = "cleaned_at")
    var cleanedAt: OffsetDateTime? = null,
)

enum class PhotoUploadIntentState {
    PENDING,
    VERIFYING,
    COMMITTED,
    REJECTED,
    EXPIRED,
}
