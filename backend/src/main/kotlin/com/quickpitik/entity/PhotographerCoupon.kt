package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.time.OffsetDateTime
import java.util.UUID

// One coupon per photographer — the photographer's user id is the primary key.
// `code` is stored UPPERCASE and is globally unique (V45) because a runner
// types only the code at checkout; the owner is derived from it.
@Entity
@Table(name = "photographer_coupons")
class PhotographerCoupon(
    @Id
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

    @Column(name = "created_at", nullable = false, updatable = false)
    val createdAt: OffsetDateTime = OffsetDateTime.now(),

    @Column(name = "updated_at", nullable = false)
    var updatedAt: OffsetDateTime = OffsetDateTime.now(),
) {
    fun isLive(now: OffsetDateTime): Boolean = active && (expiresAt?.isAfter(now) ?: true)
}
