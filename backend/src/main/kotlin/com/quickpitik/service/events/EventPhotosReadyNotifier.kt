package com.quickpitik.service.events

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.entity.UserSelfie
import com.quickpitik.repository.EventPhotoAlertRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.email.EmailService
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.contentTypeForKey
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.OffsetDateTime
import java.util.UUID

// Per-alert worker for the "your photos are ready" email. Mirrors
// OrderReceiptEmailService: face-match the runner's selfie against the event,
// and if there is >= 1 hit, claim notified_at (conditional UPDATE) and send
// once. Database reads and claim/release updates use short transaction phases;
// storage, face-provider, and email I/O run with no database connection held.
@Service
class EventPhotosReadyNotifier(
    private val alertRepository: EventPhotoAlertRepository,
    private val eventRepository: EventRepository,
    private val userRepository: UserRepository,
    private val userSelfieRepository: UserSelfieRepository,
    private val storageService: StorageService,
    private val photoSearchService: PhotoSearchService,
    private val aiApiProperties: AiApiProperties,
    private val emailService: EmailService,
    private val transactionTemplate: TransactionTemplate,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun notifyIfMatched(alertId: UUID) {
        val prepared = transactionTemplate.execute { prepare(alertId) } ?: return
        val bytes = runCatching { storageService.getBytes(prepared.selfieKey) }.getOrElse {
            log.warn(
                "Photos-ready: selfie bytes unavailable for alert {} (key {}) — retry next sweep",
                prepared.alertId, prepared.selfieKey,
            )
            return
        }

        // One call answers both "ready?" and "how many?" — total is the full
        // match count (PaginatedResponse.of computes it independently of limit).
        val page = try {
            photoSearchService.searchByFace(
                eventId = prepared.eventId,
                selfieBytes = bytes,
                contentType = contentTypeForKey(prepared.selfieKey),
                filename = prepared.selfieKey.substringAfterLast('/'),
                pagination = PaginationParams.of(0, 1),
                requesterUserId = prepared.userId,
                allowFallbackOnError = false,
            )
        } catch (ex: Exception) {
            log.warn(
                "Photos-ready: face search failed for alert {} - retry after rotation: {}",
                prepared.alertId,
                ex.message,
            )
            return
        }
        if (page.total == 0L) return // photos not ready — claim NOT burned, retry next sweep

        val recipient = transactionTemplate.execute { claim(prepared) } ?: return
        try {
            emailService.sendEventPhotosReadyEmail(
                toEmail = recipient.email,
                runnerName = recipient.name,
                eventName = prepared.eventName,
                eventSlug = prepared.eventSlug,
                matchCount = page.total.toInt(),
            )
            log.info("Photos-ready notified · alert={} to={} matches={}", prepared.alertId, recipient.email, page.total)
        } catch (ex: Exception) {
            log.error("Photos-ready send failed · alert={} err={}", prepared.alertId, ex.message, ex)
            // Hand the claim back so the next sweep retries — same door
            // OrderReceiptEmailService.releaseReceiptSend leaves open.
            transactionTemplate.executeWithoutResult { alertRepository.releaseNotify(prepared.alertId) }
        }
    }

    private fun prepare(alertId: UUID): PreparedSearch? {
        val alert = alertRepository.findById(alertId).orElse(null) ?: return null
        if (alert.notifiedAt != null) return null // already sent (cheap fast path)
        if (!aiApiProperties.enabled) return null // no AI => no possible match (sweep also gates)
        // Pending rows are ordered by this stamp, rotating unmatched or broken
        // alerts behind runners the bounded sweep has not checked yet.
        alert.lastCheckedAt = OffsetDateTime.now()

        val event = eventRepository.findById(alert.eventId).orElse(null) ?: return null
        val selfie = resolveSelfie(alert) ?: return null // selfie deleted + no fallback → skip
        return PreparedSearch(alert.id, alert.userId, event.id, selfie.s3Key, event.name, event.slug)
    }

    private fun claim(prepared: PreparedSearch): Recipient? {
        val user = userRepository.findById(prepared.userId).orElse(null) ?: return null
        if (user.email.isBlank()) return null

        // Claim AFTER every skip, exactly like OrderReceiptEmailService — an
        // alert that legitimately isn't ready must never spend its one claim.
        if (alertRepository.claimNotify(prepared.alertId, OffsetDateTime.now()) == 0) {
            log.info("Photos-ready already claimed by a concurrent send for alert {} — skipping", prepared.alertId)
            return null
        }
        return Recipient(user.email, user.name)
    }

    private data class PreparedSearch(
        val alertId: UUID,
        val userId: UUID,
        val eventId: UUID,
        val selfieKey: String,
        val eventName: String,
        val eventSlug: String,
    )

    private data class Recipient(val email: String, val name: String)

    private fun resolveSelfie(alert: EventPhotoAlert): UserSelfie? =
        alert.selfieId?.let { userSelfieRepository.findByIdAndUserId(it, alert.userId) }
            ?: userSelfieRepository.findFirstByUserIdAndIsPrimaryTrue(alert.userId)
            ?: userSelfieRepository.findFirstByUserIdOrderByUploadedAtDesc(alert.userId)
}
