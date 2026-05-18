package com.quickpitik.mobile.data.remote

data class PaginatedResponse<T>(
    val items: List<T>,
    val total: Long,
    val offset: Int,
    val limit: Int
)

data class PhotographerEventSummaryDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val location: String,
    val state: String, // "live", "upcoming", "open", "past"
    val photoCount: Int,
    val salesCount: Int,
    val revenueKept: Double
)
