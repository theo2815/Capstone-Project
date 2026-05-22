package com.quickpitik.mobile.data.remote

data class SelfieRefDto(
    val id: String,
    val dataUrl: String,
    val uploadedAt: String,
    val isPrimary: Boolean,
    val qualityScore: Double
)

data class ProfileUpdateRequest(
    val name: String
)

data class PasswordChangeRequest(
    val currentPassword: String,
    val newPassword: String,
    val refreshToken: String? = null
)
