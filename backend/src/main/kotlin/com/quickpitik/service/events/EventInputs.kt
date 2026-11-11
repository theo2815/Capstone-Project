package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ValidationException
import com.quickpitik.service.reference.RegionsService
import java.time.LocalDate
import java.util.UUID

// Field parsing shared by the admin and photographer event writers (V46
// lifted these out of AdminEventService so both create paths agree).
object EventInputs {
    fun parseDate(raw: String): LocalDate =
        runCatching { LocalDate.parse(raw.trim()) }.getOrElse {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "date must be ISO yyyy-MM-dd",
                field = "date",
            )
        }

    // Human-readable slug plus a short random suffix so two events with the
    // same title never collide on the unique index. Immutable after create.
    fun slugify(title: String): String {
        val base = title.trim().lowercase()
            .replace(Regex("[^a-z0-9\\s-]"), "")
            .replace(Regex("\\s+"), "-")
            .replace(Regex("-+"), "-")
            .trim('-')
            .ifBlank { "event" }
        val suffix = UUID.randomUUID().toString().take(6)
        return "$base-$suffix"
    }

    // Lifted from AdminEventService (AA-R3): admin events may leave both blank; races must name both.
    fun validateLocation(regionsService: RegionsService, region: String?, province: String?, required: Boolean) {
        if (!required && region == null && province == null) return
        if (region.isNullOrBlank() || province.isNullOrBlank() || region.length > 20 || province.length > 40 ||
            !regionsService.isValidPair(region, province)) {
            throw ValidationException(code = ErrorCodes.INVALID_REGION,
                message = "Unknown regionCode/provinceCode pair", field = "regionCode")
        }
    }
}
