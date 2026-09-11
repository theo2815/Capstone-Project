package com.quickpitik.mobile.data.remote

// One reading of the backend's `{success, data, errors}` envelope, for the call
// sites that used to spell it out: `if (success && data != null) … else …` plus
// a catch. Around thirty copies had drifted on which fallback copy they show
// and on whether the backend's own words reach the screen at all.
//
// Use [apiCall] where the catch was RetrofitClient.parseError; use [toResult]
// on its own where the catch must keep the raw exception (the orders poll reads
// a 404 off the HttpException) or revert an optimistic edit first.

/** Success only when the payload arrived - `data` is what the caller asked for. */
fun <T> ApiResponseEnvelope<T>.toResult(fallback: String): Result<T> =
    if (success && data != null) Result.success(data)
    else Result.failure(Exception(error ?: fallback))

/**
 * [block]'s envelope as a [Result]; a thrown failure carries the copy
 * [RetrofitClient.parseError] gives it - the backend's message for a non-2xx,
 * plain language for a dropped connection, never an exception class name.
 *
 * Catches `Exception`, so cancellation lands here too - exactly what the call
 * sites it replaces did; the one place to change it if that ever bites.
 */
suspend fun <T> apiCall(fallback: String, block: suspend () -> ApiResponseEnvelope<T>): Result<T> =
    try {
        block().toResult(fallback)
    } catch (e: Exception) {
        Result.failure(Exception(RetrofitClient.parseError(e)))
    }

/** The user-facing message [toResult] and [apiCall] always put on a failure. */
val Throwable.uiMessage: String get() = message ?: "An unexpected error occurred"
