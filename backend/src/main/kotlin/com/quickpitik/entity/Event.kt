package com.quickpitik.entity

import jakarta.persistence.CollectionTable
import jakarta.persistence.Column
import jakarta.persistence.ElementCollection
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.FetchType
import jakarta.persistence.Id
import jakarta.persistence.JoinColumn
import jakarta.persistence.PreUpdate
import jakarta.persistence.Table
import org.hibernate.annotations.JdbcTypeCode
import org.hibernate.type.SqlTypes
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

@Entity
@Table(name = "events")
class Event(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(nullable = false, unique = true, length = 160)
    var slug: String,

    @Column(nullable = false)
    var name: String,

    @Column(nullable = false)
    var date: LocalDate,

    @Column(nullable = false)
    var location: String,

    @Column(name = "cover_s3_key")
    var coverS3Key: String? = null,

    @Column(name = "photo_count", nullable = false)
    var photoCount: Int = 0,

    // Reserved for participant management (roadmap). No participants table
    // exists yet, so this is always 0. The wire field is kept so website
    // (src/types/event.ts) + mobile (RunnerDtos.kt) don't break — both declare
    // it required/non-null. Do not remove without coordinating both clients.
    @Column(name = "participant_count", nullable = false)
    var participantCount: Int = 0,

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    var status: EventStatus,

    @Column(nullable = false, columnDefinition = "TEXT")
    var description: String = "",

    @Column(name = "organizer_name", nullable = false)
    var organizerName: String = "",

    @Column(name = "price_per_photo", nullable = false)
    var pricePerPhoto: BigDecimal = BigDecimal.ZERO,

    @Column(name = "bundle_price")
    var bundlePrice: BigDecimal? = null,

    @Column(name = "bundle_size")
    var bundleSize: Int? = null,

    @ElementCollection(fetch = FetchType.EAGER)
    @CollectionTable(
        name = "event_categories",
        joinColumns = [JoinColumn(name = "event_id")],
    )
    @Column(name = "category", nullable = false, length = 50)
    var categories: MutableSet<String> = mutableSetOf(),

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "updated_at", nullable = false)
    var updatedAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "deleted_at")
    var deletedAt: OffsetDateTime? = null,

    @Column(name = "admin_overrides", nullable = false, columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    var adminOverrides: List<Map<String, Any?>> = emptyList(),

    // ── Photographer-owned events (V46) ──────────────────────────────────
    // Owner photographer; null = platform/admin event (the defaults below
    // reproduce the pre-V46 behaviour exactly).
    @Column(name = "created_by")
    var createdBy: UUID? = null,

    @Enumerated(EnumType.STRING)
    @Column(name = "visibility", nullable = false, length = 10)
    var visibility: EventVisibility = EventVisibility.PUBLIC,

    @Enumerated(EnumType.STRING)
    @Column(name = "pricing_mode", nullable = false, length = 10)
    var pricingMode: EventPricingMode = EventPricingMode.PAID,

    @Enumerated(EnumType.STRING)
    @Column(name = "watermark_policy", nullable = false, length = 10)
    var watermarkPolicy: WatermarkPolicy = WatermarkPolicy.PLATFORM,

    @Enumerated(EnumType.STRING)
    @Column(name = "review_status", nullable = false, length = 16)
    var reviewStatus: EventReviewStatus = EventReviewStatus.APPROVED,

    // The owner's requested pricing trio while CHANGE_PENDING — the live
    // columns above are untouched until an admin approves it.
    @Column(name = "pending_change", columnDefinition = "jsonb")
    @JdbcTypeCode(SqlTypes.JSON)
    var pendingChange: Map<String, Any?>? = null,

    @Column(name = "review_note", length = 500)
    var reviewNote: String? = null,

    @Column(name = "reviewed_at")
    var reviewedAt: OffsetDateTime? = null,

    @Column(name = "reviewed_by")
    var reviewedBy: UUID? = null,
) {
    val isFree: Boolean get() = pricingMode == EventPricingMode.FREE
    @PreUpdate
    fun onUpdate() {
        updatedAt = OffsetDateTime.now()
    }
}
