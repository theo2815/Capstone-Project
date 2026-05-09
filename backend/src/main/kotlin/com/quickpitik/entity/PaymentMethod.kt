package com.quickpitik.entity

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException

enum class PaymentMethod(val wire: String) {
    GCASH("gcash"),
    MAYA("maya"),
    CARD("card");

    companion object {
        fun fromWire(raw: String?): PaymentMethod {
            val normalised = raw?.trim()?.lowercase().orEmpty()
            return entries.firstOrNull { it.wire == normalised }
                ?: throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "paymentMethod must be gcash, maya, or card",
                    field = "paymentMethod",
                )
        }
    }
}
