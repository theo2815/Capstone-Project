package com.quickpitik.exception

import com.quickpitik.common.ErrorCodes

class ValidationException(
    message: String,
    val code: String = ErrorCodes.VALIDATION_ERROR,
    val field: String? = null,
) : RuntimeException(message)
