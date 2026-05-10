package com.quickpitik.exception

import org.springframework.http.HttpStatus

// Generic envelope-friendly exception. Each call site picks the HTTP status +
// canonical error code so the FE only ever sees the shapes documented in the
// FE wiring contract. Existing typed exceptions (NotFoundException, etc.) keep
// their dedicated handlers — ApiException is the catch-all for new statuses
// (415, 413, 429, 422) where adding a one-off exception per code would bloat.
class ApiException(
    val status: HttpStatus,
    val code: String,
    message: String,
    val field: String? = null,
    val retryAfterSeconds: Long? = null,
) : RuntimeException(message)
