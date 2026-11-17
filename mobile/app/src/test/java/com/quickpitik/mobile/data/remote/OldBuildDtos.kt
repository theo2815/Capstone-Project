package com.quickpitik.mobile.data.remote

// Copies of four DTO classes as an installed build has them, taken from commit
// d06bb9e9 (2026-11-08, before the race-event fields). Test fixture only: never
// edit these to match today's classes; DtoCompatTest parses today's payloads with them.

data class OldEventDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val location: String,
    val bannerUrl: String?,
    val photoCount: Int,
    val participantCount: Int,
    val status: String,
    val visibility: String = "public",
    val pricingMode: String = "paid",
)

data class OldPhotographerEventSummaryDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val location: String,
    val state: String,
    val photoCount: Int,
    val salesCount: Int,
    val revenueKept: Double,
    val bannerUrl: String? = null,
    val handle: String? = null,
    val ownedByMe: Boolean = false,
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val watermarkPolicy: String = "platform",
    val reviewStatus: String = "approved",
    val reviewNote: String? = null,
    val pendingChange: Map<String, Any?>? = null,
)

data class OldPhotographerEventDetailDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String,
    val location: String,
    val bannerUrl: String? = null,
    val state: String,
    val photoCount: Int,
    val salesCount: Int,
    val revenueKept: Double,
    val firstUploadAt: String? = null,
    val lastUploadAt: String? = null,
    val handle: String? = null,
    val ownedByMe: Boolean = false,
    val visibility: String = "public",
    val pricingMode: String = "paid",
    val watermarkPolicy: String = "platform",
    val reviewStatus: String = "approved",
    val reviewNote: String? = null,
    val pendingChange: Map<String, Any?>? = null,
    val description: String = "",
    val organizerName: String = "",
    val pricePerPhoto: Double = 0.0,
    val deletedAt: String? = null,
)

data class OldCartItemDto(
    val photoId: String,
    val eventId: String,
    val thumbnailUrl: String?,
    val price: Double,
    val bib: String?,
    val eventName: String?,
    val eventSlug: String?,
    val tone: Int?,
    val time: String?
)
