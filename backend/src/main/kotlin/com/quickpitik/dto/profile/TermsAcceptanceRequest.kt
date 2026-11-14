package com.quickpitik.dto.profile

import jakarta.validation.constraints.NotBlank

// RV-L4: POST /me/terms-acceptance. The version the client showed; it must be the server's current one.
data class TermsAcceptanceRequest(
    @field:NotBlank val version: String? = null,
)
