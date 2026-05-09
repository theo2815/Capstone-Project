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
import org.springframework.web.bind.MethodArgumentNotValidException
import org.springframework.web.bind.annotation.ExceptionHandler
import org.springframework.web.bind.annotation.RestControllerAdvice

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

    @ExceptionHandler(AiApiException::class)
    fun handleAiApi(ex: AiApiException): ResponseEntity<ApiResponse<Nothing>> {
        log.warn("ai-api call failed: status={} code={} message={}", ex.status, ex.aiCode, ex.message)
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(
            ApiResponse.failure(
                ApiError(
                    code = ErrorCodes.AI_API_UNAVAILABLE,
                    message = "AI service is temporarily unavailable. Please try again.",
                ),
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
}
