package com.quickpitik.exception

import com.quickpitik.common.ApiError
import com.quickpitik.common.ApiResponse
import com.quickpitik.common.ErrorCodes
import com.quickpitik.service.ai.AiApiException
import org.slf4j.LoggerFactory
import org.springframework.core.Ordered
import org.springframework.core.annotation.Order
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
import org.springframework.security.access.AccessDeniedException
import org.springframework.security.authentication.BadCredentialsException
import org.springframework.web.HttpMediaTypeNotSupportedException
import org.springframework.web.bind.MethodArgumentNotValidException
import org.springframework.web.bind.annotation.ExceptionHandler
import org.springframework.web.bind.annotation.RestControllerAdvice
import org.springframework.web.multipart.MaxUploadSizeExceededException
import org.springframework.web.multipart.MultipartException

@RestControllerAdvice
@Order(Ordered.HIGHEST_PRECEDENCE)
class GlobalExceptionHandler {
    private val log = LoggerFactory.getLogger(javaClass)

    @ExceptionHandler(UnauthorizedException::class)
    fun handleUnauthorized(ex: UnauthorizedException): ResponseEntity<ApiResponse<Nothing>> =
        ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(
            ApiResponse.failure(ApiError(code = ex.code, message = ex.message ?: "Unauthorized")),
        )

    @ExceptionHandler(ConflictException::class)
    fun handleConflict(ex: ConflictException): ResponseEntity<ApiResponse<Nothing>> =
        ResponseEntity.status(HttpStatus.CONFLICT).body(
            ApiResponse.failure(ApiError(code = ex.code, message = ex.message ?: "Conflict")),
        )

    @ExceptionHandler(NotFoundException::class)
    fun handleNotFound(ex: NotFoundException): ResponseEntity<ApiResponse<Nothing>> =
        ResponseEntity.status(HttpStatus.NOT_FOUND).body(
            ApiResponse.failure(ApiError(code = ex.code, message = ex.message ?: "Not found")),
        )

    @ExceptionHandler(ValidationException::class)
    fun handleValidation(ex: ValidationException): ResponseEntity<ApiResponse<Nothing>> =
        ResponseEntity.status(HttpStatus.BAD_REQUEST).body(
            ApiResponse.failure(
                ApiError(code = ex.code, message = ex.message ?: "Invalid request", field = ex.field),
            ),
        )

    @ExceptionHandler(MethodArgumentNotValidException::class)
    fun handleMethodArgNotValid(ex: MethodArgumentNotValidException): ResponseEntity<ApiResponse<Nothing>> {
        val errors = ex.bindingResult.fieldErrors.map {
            ApiError(
                code = "VALIDATION_FAILED",
                message = it.defaultMessage ?: "invalid",
                field = it.field,
            )
        }
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(ApiResponse.failure(errors))
    }

    @ExceptionHandler(BadCredentialsException::class)
    fun handleBadCredentials(ex: BadCredentialsException): ResponseEntity<ApiResponse<Nothing>> =
        ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(
            ApiResponse.failure(
                ApiError(code = "INVALID_CREDENTIALS", message = "Invalid email or password"),
            ),
        )

    @ExceptionHandler(AccessDeniedException::class)
    fun handleAccessDenied(ex: AccessDeniedException): ResponseEntity<ApiResponse<Nothing>> =
        ResponseEntity.status(HttpStatus.FORBIDDEN).body(
            ApiResponse.failure(ApiError(code = "FORBIDDEN", message = "Access denied")),
        )

    @ExceptionHandler(MaxUploadSizeExceededException::class)
    fun handleMaxUploadSize(ex: MaxUploadSizeExceededException): ResponseEntity<ApiResponse<Nothing>> =
        ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE).body(
            ApiResponse.failure(
                ApiError(
                    code = ErrorCodes.PAYLOAD_TOO_LARGE,
                    message = "File too large.",
                ),
            ),
        )

    @ExceptionHandler(MultipartException::class)
    fun handleMultipart(ex: MultipartException): ResponseEntity<ApiResponse<Nothing>> {
        // MaxUploadSizeExceededException is a subclass — Spring picks the more specific
        // handler above. This catches malformed-multipart and the like.
        log.warn("Multipart parse failed: {}", ex.message)
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(
            ApiResponse.failure(
                ApiError(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "Multipart upload was malformed.",
                ),
            ),
        )
    }

    @ExceptionHandler(HttpMediaTypeNotSupportedException::class)
    fun handleUnsupportedMediaType(ex: HttpMediaTypeNotSupportedException): ResponseEntity<ApiResponse<Nothing>> =
        ResponseEntity.status(HttpStatus.UNSUPPORTED_MEDIA_TYPE).body(
            ApiResponse.failure(
                ApiError(
                    code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                    message = "Content type not supported.",
                ),
            ),
        )

    /**
     * ai-api failures are not all outages. `AiApiException.status` already
     * carries what actually happened — a 4xx means ai-api understood us and
     * rejected the input (bad image, unprocessable selfie), a 5xx or a
     * connect failure means the service is down. Collapsing both to 503 told
     * the caller to "try again later" for a problem retrying can never fix.
     * Map 4xx → 422 (the input is the problem), everything else → 503.
     */
    @ExceptionHandler(AiApiException::class)
    fun handleAiApi(ex: AiApiException): ResponseEntity<ApiResponse<Nothing>> {
        log.warn("ai-api call failed: status={} code={} message={}", ex.status, ex.aiCode, ex.message)
        val clientFault = ex.status.is4xxClientError
        val status = if (clientFault) HttpStatus.UNPROCESSABLE_ENTITY else HttpStatus.SERVICE_UNAVAILABLE
        val message =
            if (clientFault) "That image could not be processed. Please try a different one."
            else "AI service is temporarily unavailable. Please try again."
        return ResponseEntity.status(status).body(
            ApiResponse.failure(
                ApiError(
                    // F5 (2026-05-27): pass through ai-api's specific code
                    // (LOW_QUALITY, NO_FACES, etc.) when present so the FE can
                    // map it to targeted copy instead of always showing the
                    // generic "AI service down" message. Null aiCode = real
                    // outage → falls back to AI_API_UNAVAILABLE.
                    code = aiErrorCode(ex.aiCode),
                    message = message,
                ),
            ),
        )
    }

    @ExceptionHandler(ApiException::class)
    fun handleApi(ex: ApiException): ResponseEntity<ApiResponse<Nothing>> {
        val builder = ResponseEntity.status(ex.status)
        ex.retryAfterSeconds?.let { builder.header("Retry-After", it.toString()) }
        return builder.body(
            ApiResponse.failure(
                ApiError(code = ex.code, message = ex.message ?: "Error", field = ex.field),
            ),
        )
    }

    @ExceptionHandler(Exception::class)
    fun handleGeneric(ex: Exception): ResponseEntity<ApiResponse<Nothing>> {
        log.error("Unhandled exception", ex)
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
            ApiResponse.failure(ApiError(code = "INTERNAL_ERROR", message = "Internal server error")),
        )
    }

    /**
     * `AiApiException.aiCode` is ai-api's `error.code` verbatim off the wire.
     * Per ai-api/docs/integration-contracts.md, only some of those are stable
     * wire codes — `QuickPitikError` subclasses set `error.code` to the Python
     * *class name* (`ImageValidationError`, `JobNotFoundError`, …). Forwarding
     * that raw leaks ai-api internals into our public envelope and hands the FE
     * a code it can't map. Allowlist the documented codes; collapse everything
     * else (class names, future codes, null) to AI_API_UNAVAILABLE.
     */
    private fun aiErrorCode(aiCode: String?): String =
        if (aiCode in AI_WIRE_CODES) aiCode!! else ErrorCodes.AI_API_UNAVAILABLE

    private companion object {
        // ai-api/docs/integration-contracts.md § Error Handling Contract.
        val AI_WIRE_CODES = setOf(
            "LOW_QUALITY",
            "NO_FACES",
            "NOT_FOUND",
            "INVALID_INPUT",
            "INVALID_WEBHOOK_URL",
            "DELETE_FAILED",
            "EMPTY_BATCH",
            "BATCH_TOO_LARGE",
            "TOO_MANY_JOBS",
            "MODEL_UNAVAILABLE",
            "REQUEST_TIMEOUT",
        )
    }
}
