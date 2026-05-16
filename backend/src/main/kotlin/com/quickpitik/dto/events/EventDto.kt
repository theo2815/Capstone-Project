package com.quickpitik.dto.events

import com.quickpitik.entity.EventStatus
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID

// Conversion lives in EventDtoMapper — it needs StorageService to presign
// cover_s3_key, which means the mapping is a service, not a plain function.

data class EventDto(
    val id: UUID,
    val slug: String,
    val name: String,
    val date: LocalDate,
    val location: String,
    val bannerUrl: String?,
    val photoCount: Int,
    val participantCount: Int,
    val status: EventStatus,
)

data class EventDetailDto(
    val id: UUID,
    val slug: String,
    val name: String,
    val date: LocalDate,
    val location: String,
    val bannerUrl: String?,
    val photoCount: Int,
    val participantCount: Int,
    val status: EventStatus,
    val description: String,
    val organizerName: String,
    val categories: List<String>,
    val pricePerPhoto: BigDecimal,
    val bundlePrice: BigDecimal?,
    val bundleSize: Int?,
)
