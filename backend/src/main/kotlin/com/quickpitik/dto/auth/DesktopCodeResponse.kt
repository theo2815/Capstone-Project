package com.quickpitik.dto.auth

data class DesktopCodeResponse(
    val code: String,
    val expiresInSeconds: Long,
)
