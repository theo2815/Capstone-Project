package com.quickpitik.dto.auth

data class AuthResponse(
    val accessToken: String,
    val refreshToken: String,
    val user: UserDto,
)
