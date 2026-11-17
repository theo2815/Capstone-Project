package com.quickpitik.dto.photographer

import java.math.BigDecimal
import java.time.OffsetDateTime

// PUT /me/photographer/events/{eventId}/price (AA-R11). null = back to the race price.
data class SetRacePriceRequest(
    val pricePhp: BigDecimal? = null,
)

// A photographer's price in a shared race. myPricePhp null = the race price; nextChangeAt null = can change now.
data class RacePriceDto(
    val floorPhp: BigDecimal,
    val ceilingPhp: BigDecimal,
    val myPricePhp: BigDecimal?,
    val effectivePhp: BigDecimal,
    val nextChangeAt: OffsetDateTime?,
)
