package com.quickpitik.common

import com.quickpitik.exception.ValidationException
import java.util.UUID

@JvmInline
value class IdempotencyKey(val value: String) {
    companion object {
        const val HEADER = "Idempotency-Key"

        fun parse(raw: String?): IdempotencyKey? {
            val trimmed = raw?.trim().orEmpty()
            if (trimmed.isEmpty()) return null
            return try {
                IdempotencyKey(UUID.fromString(trimmed).toString())
            } catch (_: IllegalArgumentException) {
                throw ValidationException(
                    code = ErrorCodes.INVALID_IDEMPOTENCY_KEY,
                    message = "Idempotency-Key must be a UUID",
                    field = HEADER,
                )
            }
        }
    }
}
