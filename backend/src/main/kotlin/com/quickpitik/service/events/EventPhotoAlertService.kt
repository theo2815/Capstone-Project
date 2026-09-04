package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.events.PhotoAlertStatusDto
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.entity.EventStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotoAlertRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserSelfieRepository
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.LocalDate
import java.time.ZoneId
import java.util.UUID

// Runner opt-in for the "your photos are ready" email. Reuses the runner's
// existing selfie library — registering just links a chosen (or the primary)
// selfie to the event; the date-based sweep does the matching + send later.
@Service
class EventPhotoAlertService(
    private val eventRepository: EventRepository,
    private val userSelfieRepository: UserSelfieRepository,
    private val alertRepository: EventPhotoAlertRepository,
) {
    @Transactional
    fun register(userId: UUID, slug: String, selfieIdRaw: String?): PhotoAlertStatusDto {
        val event = eventRepository.findBySlugAndDeletedAtIsNull(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")

        val today = LocalDate.now(PH_ZONE)
        if (event.status !in ALERTABLE_STATUSES || today.isAfter(event.date.plusDays(UPLOAD_WINDOW_DAYS))) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.EVENT_NOT_UPLOADABLE,
                message = "This event is not accepting photo alerts.",
            )
        }

        val selfieId = resolveSelfieId(userId, selfieIdRaw)

        val existing = alertRepository.findByEventIdAndUserId(event.id, userId)
        val saved = if (existing != null) {
            existing.selfieId = selfieId // managed entity — flushed on TX commit
            existing
        } else {
            alertRepository.save(
                EventPhotoAlert(eventId = event.id, userId = userId, selfieId = selfieId),
            )
        }
        return PhotoAlertStatusDto(registered = true, selfieId = saved.selfieId)
    }

    @Transactional
    fun optOut(userId: UUID, slug: String): Boolean {
        val event = eventRepository.findBySlugAndDeletedAtIsNull(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        return alertRepository.deleteByEventIdAndUserId(event.id, userId) > 0
    }

    @Transactional(readOnly = true)
    fun status(userId: UUID, slug: String): PhotoAlertStatusDto {
        val event = eventRepository.findBySlugAndDeletedAtIsNull(slug)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val alert = alertRepository.findByEventIdAndUserId(event.id, userId)
        return PhotoAlertStatusDto(registered = alert != null, selfieId = alert?.selfieId)
    }

    // Explicit selfieId → must be the caller's own (IDOR-safe via findByIdAndUserId,
    // same guard as EventPhotoController.searchByFaceJson). None given → primary,
    // else most recent. A runner with no selfie at all can't be matched, so we
    // reject rather than create a dead alert that could never fire.
    private fun resolveSelfieId(userId: UUID, selfieIdRaw: String?): UUID {
        val raw = selfieIdRaw?.trim().orEmpty()
        if (raw.isNotEmpty()) {
            val selfieUuid = runCatching { UUID.fromString(raw) }.getOrNull()
                ?: throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "selfieId must be a UUID",
                    field = "selfieId",
                )
            val selfie = userSelfieRepository.findByIdAndUserId(selfieUuid, userId)
                ?: throw NotFoundException(code = ErrorCodes.SELFIE_NOT_FOUND, message = "Selfie not found")
            return selfie.id
        }
        val fallback = userSelfieRepository.findFirstByUserIdAndIsPrimaryTrue(userId)
            ?: userSelfieRepository.findFirstByUserIdOrderByUploadedAtDesc(userId)
            ?: throw ValidationException(
                code = ErrorCodes.SELFIE_REQUIRED,
                message = "Add a selfie before turning on photo alerts",
                field = "selfie",
            )
        return fallback.id
    }

    private companion object {
        val ALERTABLE_STATUSES = setOf(EventStatus.ACTIVE, EventStatus.COMPLETED)
        val PH_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        const val UPLOAD_WINDOW_DAYS = 3L
    }
}
