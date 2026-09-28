package com.quickpitik.entity

import com.quickpitik.dto.photos.PhotoMetadataFields
import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.time.OffsetDateTime
import java.util.UUID

@Entity
@Table(name = "photo_metadata")
class PhotoMetadata(
    @Id
    @Column(name = "photo_id", nullable = false, updatable = false)
    val photoId: UUID,

    @Column(name = "original_filename", length = 255)
    var originalFilename: String? = null,

    @Column(name = "original_size_bytes")
    var originalSizeBytes: Long? = null,

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "fields", columnDefinition = "jsonb")
    var fields: PhotoMetadataFields? = null,

    @Enumerated(EnumType.STRING)
    @Column(name = "state", nullable = false, length = 16)
    var state: PhotoMetadataState = PhotoMetadataState.PENDING,

    @Column(name = "extractor_version", nullable = false)
    var extractorVersion: Int = 1,

    @Column(name = "extracted_at")
    var extractedAt: OffsetDateTime? = null,

    @Column(name = "attempts", nullable = false)
    var attempts: Int = 0,

    @Column(name = "next_attempt_at")
    var nextAttemptAt: OffsetDateTime? = null,

    @Column(name = "lease_token")
    var leaseToken: UUID? = null,

    @Column(name = "lease_until")
    var leaseUntil: OffsetDateTime? = null,
)

enum class PhotoMetadataState { PENDING, PROCESSING, READY, EMPTY, FAILED, UNAVAILABLE }
