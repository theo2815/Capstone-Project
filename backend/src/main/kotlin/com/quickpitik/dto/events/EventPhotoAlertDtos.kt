package com.quickpitik.dto.events

import java.util.UUID

// Body for POST /events/{slug}/photo-alert. selfieId optional — null means
// "use my primary (or most recent) selfie".
data class PhotoAlertRequest(
    val selfieId: String? = null,
)

// Returned by POST + GET — drives the client's opt-in toggle state.
data class PhotoAlertStatusDto(
    val registered: Boolean,
    val selfieId: UUID?,
)
