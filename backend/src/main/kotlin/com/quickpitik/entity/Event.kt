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

    @Column(name = "banner_url")
    var bannerUrl: String? = null,

    @Column(name = "photo_count", nullable = false)
    var photoCount: Int = 0,

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
) {
    @PreUpdate
    fun onUpdate() {
        updatedAt = OffsetDateTime.now()
    }
}
