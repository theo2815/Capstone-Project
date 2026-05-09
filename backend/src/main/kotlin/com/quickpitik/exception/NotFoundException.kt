package com.quickpitik.exception

class NotFoundException(
    message: String,
    val code: String = "NOT_FOUND",
) : RuntimeException(message)
