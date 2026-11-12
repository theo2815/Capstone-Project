package com.quickpitik.service.removal

import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.AiProperties
import com.quickpitik.config.AiProvider
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.EventStatus
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.LivenessSessionRepository
import com.quickpitik.repository.PhotoRemovalRequestRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.events.EventFaceExpiry
import org.springframework.stereotype.Service
import java.time.Clock
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID

enum class RunnerRemovalBlock {
    EVENT_NOT_ELIGIBLE, AI_OFF, FACE_DATA_EXPIRED, INTAKE_CLOSED, REQUEST_OPEN, TOO_MANY_OPEN, ATTEMPTS_EXHAUSTED,
}

data class RunnerRemovalEligibility(
    val eligible: Boolean,
    val blockedReason: RunnerRemovalBlock?,
    val attemptsLeftToday: Int,
    val emailVerified: Boolean,
    val openRequestId: UUID?,
)

/** Every refusal a runner can hit before any AWS call; first failing rule wins (RD-2). */
@Service
class RunnerRemovalEligibilityService(
    private val eventRepository: EventRepository,
    private val userRepository: UserRepository,
    private val requestRepository: PhotoRemovalRequestRepository,
    private val sessionRepository: LivenessSessionRepository,
    private val eventFaceExpiry: EventFaceExpiry,
    private val aiApiProperties: AiApiProperties,
    private val aiProperties: AiProperties,
    private val props: PhotoRemovalProperties,
    private val clock: Clock,
) {
    fun check(userId: UUID, eventId: UUID): RunnerRemovalEligibility {
        val runner = props.runner
        val dayStart = OffsetDateTime.now(clock.withZone(MANILA)).toLocalDate().atStartOfDay(MANILA).toOffsetDateTime()
        val used = sessionRepository.countByUserIdAndCreatedAtGreaterThanEqual(userId, dayStart)
        val left = (runner.checksPerDay - used).toInt().coerceAtLeast(0)
        val verified = userRepository.findById(userId).map { it.emailVerifiedAt != null }.orElse(false)
        fun result(block: RunnerRemovalBlock?, open: UUID? = null) =
            RunnerRemovalEligibility(block == null, block, left, verified, open)

        // UNLISTED is a visibility, not a status, so private events pass (N3).
        val event = eventRepository.findById(eventId).orElse(null)
        if (event == null || event.deletedAt != null || event.status == EventStatus.DRAFT) {
            return result(RunnerRemovalBlock.EVENT_NOT_ELIGIBLE)
        }
        if (!aiApiProperties.enabled || aiProperties.provider != AiProvider.REKOGNITION) return result(RunnerRemovalBlock.AI_OFF)
        if (eventFaceExpiry.ended(event)) return result(RunnerRemovalBlock.FACE_DATA_EXPIRED)
        if (runner.checksPerDay <= 0) return result(RunnerRemovalBlock.INTAKE_CLOSED)
        if (requestRepository.existsOpenRunnerRequest(userId, eventId)) {
            return result(RunnerRemovalBlock.REQUEST_OPEN, requestRepository.findOpenRunnerRequestId(userId, eventId))
        }
        if (requestRepository.countOpenRunnerRequests(userId) >= runner.maxOpenRequests) return result(RunnerRemovalBlock.TOO_MANY_OPEN)
        if (left <= 0) return result(RunnerRemovalBlock.ATTEMPTS_EXHAUSTED)
        return result(null)
    }

    private companion object {
        // Same Manila day AA's shared liveness service counts against.
        val MANILA: ZoneId = ZoneId.of("Asia/Manila")
    }
}
