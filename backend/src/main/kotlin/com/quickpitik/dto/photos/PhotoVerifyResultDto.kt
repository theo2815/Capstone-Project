package com.quickpitik.dto.photos

import java.time.LocalDate

// Answer for POST /api/v1/public/photos/verify. Attribution ONLY — never a
// photo id, a URL, or anything about a runner: the caller already holds the
// image, and this tells them whose it is, nothing more. `confidence` is
// "strong" (≤ half the threshold) or "weak" (within it); null when unmatched.
data class PhotoVerifyResultDto(
    val matched: Boolean,
    val confidence: String? = null,
    val photographerName: String? = null,
    val photographerHandle: String? = null,
    val eventName: String? = null,
    val eventDate: LocalDate? = null,
    val distance: Int? = null,
) {
    companion object {
        val NONE = PhotoVerifyResultDto(matched = false)
    }
}
