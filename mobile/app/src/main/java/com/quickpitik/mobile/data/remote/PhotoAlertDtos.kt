package com.quickpitik.mobile.data.remote

// Runner opt-in for the "your photos are ready" email — mirrors the website's
// api-photo-alert.ts. Register with a specific selfie, or (default null) the
// runner's primary selfie. Gson omits the null field, so an empty body means
// "use my primary selfie" to the backend.
data class PhotoAlertRequest(
    val selfieId: String? = null,
)

data class PhotoAlertStatusDto(
    val registered: Boolean,
    val selfieId: String?,
)
