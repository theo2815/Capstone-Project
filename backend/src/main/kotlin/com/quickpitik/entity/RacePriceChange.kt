package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.EnumType
import jakarta.persistence.Enumerated
import jakarta.persistence.Id
import jakarta.persistence.Table
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// SET = the photographer's own change (AA-R11); RESET = an admin reset, CLAMP = a race price change (AA-R12).
enum class RacePriceReason { SET, RESET, CLAMP }

// One row per race price change (V97 race_price_changes). Insert-only.
@Entity
@Table(name = "race_price_changes")
class RacePriceChange(
    @Id
    @Column(nullable = false, updatable = false)
    val id: UUID = UUID.randomUUID(),

    @Column(name = "event_id", nullable = false, updatable = false)
    val eventId: UUID,

    @Column(name = "photographer_id", nullable = false, updatable = false)
    val photographerId: UUID,

    @Column(name = "changed_by", updatable = false)
    val changedBy: UUID?,

    @Column(name = "from_php", nullable = false, precision = 12, scale = 2, updatable = false)
    val fromPhp: BigDecimal,

    @Column(name = "to_php", nullable = false, precision = 12, scale = 2, updatable = false)
    val toPhp: BigDecimal,

    @Enumerated(EnumType.STRING)
    @Column(name = "reason", nullable = false, length = 8, updatable = false)
    val reason: RacePriceReason,

    @Column(name = "changed_at", nullable = false, updatable = false)
    val changedAt: OffsetDateTime,
)
