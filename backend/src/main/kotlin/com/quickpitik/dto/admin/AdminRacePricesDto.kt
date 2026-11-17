package com.quickpitik.dto.admin

import com.quickpitik.entity.RacePriceReason
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

data class AdminRacePricesDto(
    val floorPhp: BigDecimal,
    val ceilingPhp: BigDecimal,
    val prices: List<AdminRacePriceDto>,
    val history: List<AdminRacePriceHistoryDto>,
)

data class AdminRacePriceDto(
    val photographerId: UUID,
    val name: String,
    val handle: String?,
    val effectivePhp: BigDecimal,
    val custom: Boolean,
    val priceChangedAt: OffsetDateTime?,
)

data class AdminRacePriceHistoryDto(
    val photographerId: UUID,
    val changedBy: UUID?,
    val fromPhp: BigDecimal,
    val toPhp: BigDecimal,
    val reason: RacePriceReason,
    val changedAt: OffsetDateTime,
)
