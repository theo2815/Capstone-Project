package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

@Entity
@Table(name = "user_selfies")
class UserSelfie(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "user_id", nullable = false)
    var userId: UUID,

    @Column(name = "s3_key", nullable = false, length = 512)
    var s3Key: String,

    @Column(name = "is_primary", nullable = false)
    var isPrimary: Boolean = false,

    @Column(name = "quality_score", nullable = false, precision = 5, scale = 4)
    var qualityScore: BigDecimal = BigDecimal.ZERO,

    @Column(name = "uploaded_at", nullable = false)
    var uploadedAt: OffsetDateTime = OffsetDateTime.now(),
)
