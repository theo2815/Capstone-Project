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

// Body for POST /auth/refresh. Mirrors backend dto/auth/RefreshRequest.
data class RefreshRequest(
    val refreshToken: String
)

// Body for POST /auth/forgot-password. Mirrors backend
// dto/auth/ForgotPasswordRequest. The endpoint is anti-enumeration silent —
// it answers with the same generic message whether or not the email exists,
// so the UI must never phrase its success copy as "we found your account".
data class ForgotPasswordRequest(
    val email: String
)

// Body for POST /auth/reset-password. Mirrors backend
// dto/auth/ResetPasswordRequest, which enforces @Size(min = 8) on newPassword;
// validatePassword() gates the same rule client-side first.
data class ResetPasswordRequest(
    val token: String,
    val newPassword: String
)

// Both auth-recovery endpoints return Map<String, String> ({"message": "…"}),
// wrapped by the backend's ResponseEnvelopeAdvice. The screens render their own
// copy rather than the server string, so this exists to give the envelope a
// concrete type — not to be displayed.
data class MessageResponse(
    val message: String
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
