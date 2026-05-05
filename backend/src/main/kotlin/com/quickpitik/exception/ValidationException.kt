package com.quickpitik.exception

class ValidationException(
    message: String,
    val code: String = "VALIDATION_FAILED",
    val field: String? = null,
) : RuntimeException(message)
