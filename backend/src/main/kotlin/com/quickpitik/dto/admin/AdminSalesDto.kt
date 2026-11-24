package com.quickpitik.dto.admin

import java.math.BigDecimal

// Mirrors website/src/lib/api-admin.ts AdminSalesKpis. All numbers in PHP.
data class AdminSalesKpisDto(
    val gmv: BigDecimal,
    val platformRevenue: BigDecimal,
    val refundsIssued: BigDecimal,
    // FU-3: the number of succeeded refunds in refundsIssued, same window.
    val refundsIssuedCount: Long,
    val netPlatformRevenue: BigDecimal,
    val photographerKeep: BigDecimal,
    val totalSalesCount: Long,
)

// AX-12: GET /admin/sales/trend, oldest week first; weekStart is a Manila Monday.
data class AdminSalesTrendDto(val weeks: List<AdminSalesWeekDto>)

data class AdminSalesWeekDto(
    val weekStart: String,
    val gmv: BigDecimal,
    val platformRevenue: BigDecimal,
    val refundsIssued: BigDecimal,
    val salesCount: Long,
)

// AX-12: GET /admin/sales/top-photographers, highest GMV first.
data class AdminTopPhotographersDto(val items: List<AdminTopPhotographerDto>)

data class AdminTopPhotographerDto(
    val userId: String,
    val handle: String?,
    val displayName: String,
    val gmv: BigDecimal,
    val salesCount: Long,
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
