package com.quickpitik.dto.auth

data class LogoutRequest(
    val refreshToken: String? = null,
)
