package com.quickpitik.service.liveness

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IdentityProperties
import com.quickpitik.dto.photographer.LivenessCredentialsDto
import com.quickpitik.dto.photographer.LivenessSessionDto
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.LivenessSession
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.LivenessSessionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.ai.RekognitionIdentityClient
import org.springframework.beans.factory.ObjectProvider
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import software.amazon.awssdk.core.exception.SdkException
import java.time.Clock
import java.time.Duration
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID

/**
 * Single-use AWS Face Liveness sessions bound to one account, one purpose and (for runner
 * photo removals) one event. Shared by the photographer identity check (AA-I2/AA-I3) and
 * the runner photo-removal plan (RD-2).
 *
 * Claim-then-read, for every purpose: [claim] the session in its own short transaction,
 * then read the result at once, outside any transaction. A session lives about 3 minutes
 * from creation for both starting and reading (AA-0.3b), so clients start one only when the
 * user presses start.
 */
@Service
class LivenessSessionService(
    private val sessionRepository: LivenessSessionRepository,
    private val userRepository: UserRepository,
    private val identityClient: ObjectProvider<RekognitionIdentityClient>,
    private val aiApiProperties: AiApiProperties,
    private val identityProperties: IdentityProperties,
    private val transactionTemplate: TransactionTemplate,
    private val clock: Clock,
) {
    /** Creates the AWS session and its browser keys with no transaction open, then records it under the user lock. */
    fun start(userId: UUID, purpose: LivenessPurpose, eventId: UUID?, dailyLimit: Int, consentVersion: String): LivenessSessionDto {
        if (!aiApiProperties.enabled || identityProperties.livenessRoleArn.isBlank()) throw unavailable()
        val client = identityClient.getIfAvailable() ?: throw unavailable()
        val dayStart = OffsetDateTime.now(clock.withZone(MANILA)).toLocalDate().atStartOfDay(MANILA).toOffsetDateTime()
        // Refuse before AWS is called; counted again under the lock below for parallel starts.
        if (sessionRepository.countByUserIdAndCreatedAtGreaterThanEqual(userId, dayStart) >= dailyLimit) throw exhausted(dayStart)

        val (sessionId, keys) = try {
            val id = client.createLivenessSession()
            id to client.browserCredentials(id)
        } catch (ex: SdkException) {
            throw unavailable()
        }

        val used = transactionTemplate.execute {
            userRepository.lockById(userId)
            val count = sessionRepository.countByUserIdAndCreatedAtGreaterThanEqual(userId, dayStart)
            if (count >= dailyLimit) throw exhausted(dayStart)
            sessionRepository.save(
                LivenessSession(sessionId, userId, purpose, eventId, consentVersion, OffsetDateTime.now(clock)),
            )
            count + 1
        }!!
        return LivenessSessionDto(
            sessionId = sessionId,
            region = identityProperties.livenessRegion,
            attemptsLeftToday = (dailyLimit - used).toInt().coerceAtLeast(0),
            credentials = LivenessCredentialsDto(keys.accessKeyId(), keys.secretAccessKey(), keys.sessionToken(), keys.expiration()),
        )
    }

    /** The caller's unused session for this purpose and event, else 409 LIVENESS_SESSION_INVALID. */
    fun requireOpen(userId: UUID, sessionId: String, purpose: LivenessPurpose, eventId: UUID?): LivenessSession {
        val row = sessionRepository.findBySessionIdAndUserIdAndPurpose(sessionId, userId, purpose)
        if (row == null || row.eventId != eventId || row.consumedAt != null) throw invalidSession()
        return row
    }

    /** Uses the session up. False when it is unknown, someone else's, for another purpose or event, or already used. */
    fun claim(userId: UUID, sessionId: String, purpose: LivenessPurpose, eventId: UUID?): Boolean =
        transactionTemplate.execute {
            sessionRepository.claim(sessionId, userId, purpose.name, eventId, OffsetDateTime.now(clock))
        } == 1

    private fun exhausted(dayStart: OffsetDateTime): ApiException {
        val untilMidnight = Duration.between(OffsetDateTime.now(clock), dayStart.plusDays(1)).seconds
        return ApiException(
            HttpStatus.TOO_MANY_REQUESTS, ErrorCodes.LIVENESS_ATTEMPTS_EXHAUSTED,
            "You have used today's live checks. Try again tomorrow, or have an admin check your ID instead.",
            retryAfterSeconds = untilMidnight.coerceAtLeast(1),
        )
    }

    private fun unavailable() = ApiException(
        HttpStatus.SERVICE_UNAVAILABLE, ErrorCodes.LIVENESS_UNAVAILABLE,
        "The live check is not available right now. You can have an admin check your ID instead.",
    )

    companion object {
        private val MANILA: ZoneId = ZoneId.of("Asia/Manila")

        fun invalidSession() = ApiException(
            HttpStatus.CONFLICT, ErrorCodes.LIVENESS_SESSION_INVALID,
            "This live check can't be used. Please start a new one.",
        )
    }
}
