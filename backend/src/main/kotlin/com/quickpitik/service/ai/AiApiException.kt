package com.quickpitik.service.ai

import org.springframework.http.HttpStatus

class AiApiException(
    val status: HttpStatus,
    val aiCode: String?,
    message: String,
    cause: Throwable? = null,
) : RuntimeException(message, cause) {
    val isRetryable: Boolean = status.is5xxServerError || status == HttpStatus.TOO_MANY_REQUESTS
}
