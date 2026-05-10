package com.quickpitik.dto.admin

import java.math.BigDecimal

// Mirrors website/src/lib/api-admin.ts AdminSalesKpis. All numbers in PHP.
data class AdminSalesKpisDto(
    val gmv: BigDecimal,
    val platformRevenue: BigDecimal,
    val refundsIssued: BigDecimal,
    val netPlatformRevenue: BigDecimal,
    val photographerKeep: BigDecimal,
    val totalSalesCount: Long,
)

// Mirrors AdminSalesEventRow.
data class AdminSalesEventRowDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val city: String,
    val status: String,
    val state: String,
    val photoCount: Int,
    val impliedGmv: BigDecimal,
    val impliedCut: BigDecimal,
    val refundsIssued: BigDecimal,
)
