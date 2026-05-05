package com.quickpitik.exception

class UnauthorizedException(
    message: String = "Unauthorized",
    val code: String = "UNAUTHORIZED",
) : RuntimeException(message)
