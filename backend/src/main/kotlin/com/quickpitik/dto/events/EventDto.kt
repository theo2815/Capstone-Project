package com.quickpitik.dto.events

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID

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

fun Event.toListDto(): EventDto = EventDto(
    id = id,
    slug = slug,
    name = name,
    date = date,
    location = location,
    bannerUrl = bannerUrl,
    photoCount = photoCount,
    participantCount = participantCount,
    status = status,
)

fun Event.toDetailDto(): EventDetailDto = EventDetailDto(
    id = id,
    slug = slug,
    name = name,
    date = date,
    location = location,
    bannerUrl = bannerUrl,
    photoCount = photoCount,
    participantCount = participantCount,
    status = status,
    description = description,
    organizerName = organizerName,
    categories = categories.sorted(),
    pricePerPhoto = pricePerPhoto,
    bundlePrice = bundlePrice,
    bundleSize = bundleSize,
)
