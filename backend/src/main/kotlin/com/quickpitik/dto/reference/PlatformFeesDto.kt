package com.quickpitik.dto.reference

import java.math.BigDecimal

data class PlatformFeesDto(
    val photoPricePhp: BigDecimal,
    val platformCutRate: BigDecimal,
    val photographerKeepRate: BigDecimal,
    val couponMaxPercent: Int,
)
