package com.quickpitik.exception

class ConflictException(
    message: String,
    val code: String = "CONFLICT",
) : RuntimeException(message)
