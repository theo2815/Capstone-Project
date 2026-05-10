package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.PreUpdate
import jakarta.persistence.Table
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID

@Entity
@Table(name = "photographer_settings")
class PhotographerSettings(
    @Id
    @Column(name = "user_id", nullable = false, updatable = false)
    val userId: UUID,

    @Column(name = "handle", unique = true, length = 40)
    var handle: String? = null,

    @Column(name = "brand_name", length = 100)
    var brandName: String? = null,

    @Column(name = "brand_color", nullable = false, length = 20)
    var brandColor: String = "none",

    @Column(name = "bio", nullable = false, columnDefinition = "TEXT")
    var bio: String = "",

    @Column(name = "region_code", length = 20)
    var regionCode: String? = null,

    @Column(name = "province_code", length = 40)
    var provinceCode: String? = null,

    @Column(name = "city", length = 100)
    var city: String? = null,

    @Column(name = "cover_s3_key", length = 512)
    var coverS3Key: String? = null,

    @Column(name = "cover_gradient_from", length = 20)
    var coverGradientFrom: String? = null,

    @Column(name = "cover_gradient_to", length = 20)
    var coverGradientTo: String? = null,

    @Column(name = "watermark_s3_key", length = 512)
    var watermarkS3Key: String? = null,

    @Column(name = "watermark_label", length = 40)
    var watermarkLabel: String? = null,

    @Enumerated(EnumType.STRING)
    @Column(name = "verification_status", nullable = false, length = 20)
    var verificationStatus: VerificationStatus = VerificationStatus.INCOMPLETE,

    @Column(name = "member_since", nullable = false, updatable = false)
    val memberSince: LocalDate = LocalDate.now(ZoneId.of("Asia/Manila")),

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
