package com.quickpitik.dto.auth

/**
 * A bare human-readable acknowledgement, for endpoints whose only meaningful
 * answer is "that worked" — password reset, email confirmation, and friends.
 *
 * Wire shape is `{"message": "…"}`, identical to the `mapOf("message" to …)`
 * these call sites returned before. Mobile already types both recovery
 * endpoints as `ApiResponseEnvelope<MessageResponse>`
 * (`data/remote/QuickPitikApi.kt`); the website ignores the body entirely. So
 * this is a source-level change only, with no client impact.
 */
data class MessageResponse(val message: String)
