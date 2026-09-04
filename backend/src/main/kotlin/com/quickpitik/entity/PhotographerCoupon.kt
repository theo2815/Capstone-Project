package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// One coupon per photographer-owned event. `code` stays globally unique
// because a runner types only the code at checkout; event and owner are
// derived from the persisted row, never accepted from the checkout client.
@Entity
@Table(name = "photographer_coupons")
class PhotographerCoupon(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    // Nullable only for disabled V45 rows retained by the migration.
    @Column(name = "event_id", updatable = false)
    val eventId: UUID?,

    @Column(name = "photographer_id", nullable = false, updatable = false)
    val photographerId: UUID,

    @Column(name = "code", nullable = false, length = 16)
    var code: String,

    @Column(name = "percent_off", nullable = false)
    var percentOff: Int,

    @Column(name = "active", nullable = false)
    var active: Boolean = true,

    @Column(name = "expires_at")
    var expiresAt: OffsetDateTime? = null,

    @Column(name = "usage_limit")
    var usageLimit: Int? = null,

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "updated_at", nullable = false)
    var updatedAt: OffsetDateTime = OffsetDateTime.now(),
) {
    fun isLive(now: OffsetDateTime): Boolean = active && (expiresAt?.isAfter(now) ?: true)
}
