package com.quickpitik.mobile.data.remote

data class UserDto(
    val id: String,
    val email: String,
    val name: String,
    val role: String,
    val avatarUrl: String? = null,
    val createdAt: String
)

data class LoginRequest(
    val email: String,
    val password: String
)

data class RegisterRequest(
    val name: String,
    val email: String,
    val password: String,
    val role: String // "ADMIN", "PHOTOGRAPHER", or "RUNNER"
)

data class AuthResponse(
    val accessToken: String,
    val refreshToken: String,
    val user: UserDto
)

// Standard Backend Error envelope structure
data class ApiError(
    val code: String,
    val message: String
)

data class ApiErrorEnvelope(
    val success: Boolean,
    val errors: List<ApiError>?
)

data class UploadedPhotoDto(
    val id: String,
    val status: String,
    val uploadedAt: String,
    val thumbnailUrl: String,
    val span: String,
    val aiDetectionStatus: String? = null
)
