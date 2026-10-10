package com.quickpitik.dto.auth

import jakarta.validation.constraints.NotBlank

// Desktop connect, step 2: the app redeems the code with its verifier. Public
// route — the code in the body is the credential.
data class DesktopExchangeRequest(
    @field:NotBlank
    val code: String,
    @field:NotBlank
    val codeVerifier: String,
)
