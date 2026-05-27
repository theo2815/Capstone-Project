package com.quickpitik.mobile.data.remote

data class EventDto(
    val id: String,
    val slug: String,
    val name: String,
    val date: String, // "YYYY-MM-DD"
    val location: String,
    val bannerUrl: String?,
    val photoCount: Int,
    val participantCount: Int,
    val status: String // "LIVE", "COMPLETED", etc.
)

data class PhotoDto(
    val id: String,
    val bib: String?,
    val km: Int?,
    val time: String, // "HH:MM"
    val tone: Int,
    val span: String, // e.g. "portrait"
    val price: Double,
    val imageUrl: String?,
    val alt: String?
)

data class SearchByFaceJsonRequest(
    val selfieId: String,
    val offset: Int = 0,
    val limit: Int = 100
)

// F8 (2026-05-27): Mirror of backend dto/runner/RunnerMessageDto. Same shape
// as PhotographerMessageDto + an `orderId` for deep-linking from a dispute
// outcome notification back to the /orders detail page on the website
// (mobile reuses the same field to mark the related order in its inbox UI).
data class RunnerMessageDto(
    val id: String,
    val kind: String,           // snake_case wire form, see backend RunnerMessageKind
    val title: String?,
    val body: String,
    val orderId: String?,
    val sourceDecisionId: String?,
    val createdAt: String,
    val readAt: String?,
)
