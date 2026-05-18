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
