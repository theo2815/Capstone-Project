package com.quickpitik.dto.auth

import jakarta.validation.constraints.NotBlank

// Desktop connect, step 1: the signed-in website asks for a one-time code bound
// to the BatchMyPhotos PKCE challenge (base64url of SHA-256 of the verifier).
data class DesktopCodeRequest(
    @field:NotBlank
    val codeChallenge: String,
)
