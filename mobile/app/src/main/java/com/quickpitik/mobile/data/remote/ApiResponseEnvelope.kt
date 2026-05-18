package com.quickpitik.mobile.data.remote

data class ApiResponseEnvelope<T>(
    val success: Boolean,
    val data: T? = null,
    val error: String? = null
)
