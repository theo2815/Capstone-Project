package com.quickpitik.service.events

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AiApiProperties
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.entity.UserSelfie
import com.quickpitik.repository.EventPhotoAlertRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.EmailService
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.http.MediaType
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

// Per-alert worker for the "your photos are ready" email. Mirrors
// OrderReceiptEmailService: face-match the runner's selfie against the event,
// and if there is >= 1 hit, claim notified_at (conditional UPDATE) and send
// once. REQUIRES_NEW so each alert commits independently within the sweep loop
// and one failure doesn't roll back the others.
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
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    fun notifyIfMatched(alertId: UUID) {
        val alert = alertRepository.findById(alertId).orElse(null) ?: return
        if (alert.notifiedAt != null) return // already sent (cheap fast path)
        if (!aiApiProperties.enabled) return // no AI => no possible match (sweep also gates)
        // Pending rows are ordered by this stamp, rotating unmatched or broken
        // alerts behind runners the bounded sweep has not checked yet.
        alert.lastCheckedAt = OffsetDateTime.now()

        val event = eventRepository.findById(alert.eventId).orElse(null) ?: return

        val selfie = resolveSelfie(alert) ?: return // selfie deleted + no fallback → skip
        val bytes = runCatching { storageService.getBytes(selfie.s3Key) }.getOrElse {
            log.warn(
                "Photos-ready: selfie bytes unavailable for alert {} (key {}) — retry next sweep",
                alert.id, selfie.s3Key,
            )
            return
        }

        // One call answers both "ready?" and "how many?" — total is the full
        // match count (PaginatedResponse.of computes it independently of limit).
        val page = try {
            photoSearchService.searchByFace(
                eventId = event.id,
                selfieBytes = bytes,
                contentType = contentTypeOf(selfie.s3Key),
                filename = selfie.s3Key.substringAfterLast('/'),
                pagination = PaginationParams.of(0, 1),
                requesterUserId = alert.userId,
                allowFallbackOnError = false,
            )
        } catch (ex: Exception) {
            log.warn("Photos-ready: face search failed for alert {} - retry after rotation: {}", alert.id, ex.message)
            return
        }
        if (page.total == 0L) return // photos not ready — claim NOT burned, retry next sweep

        val user = userRepository.findById(alert.userId).orElse(null) ?: return
        if (user.email.isBlank()) return

        // Claim AFTER every skip, exactly like OrderReceiptEmailService — an
        // alert that legitimately isn't ready must never spend its one claim.
        if (alertRepository.claimNotify(alert.id, OffsetDateTime.now()) == 0) {
            log.info("Photos-ready already claimed by a concurrent send for alert {} — skipping", alert.id)
            return
        }

        try {
            emailService.sendEventPhotosReadyEmail(
                toEmail = user.email,
                runnerName = user.name,
                eventName = event.name,
                eventSlug = event.slug,
                matchCount = page.total.toInt(),
            )
            log.info("Photos-ready notified · alert={} to={} matches={}", alert.id, user.email, page.total)
        } catch (ex: Exception) {
            log.error("Photos-ready send failed · alert={} err={}", alert.id, ex.message, ex)
            // Hand the claim back so the next sweep retries — same door
            // OrderReceiptEmailService.releaseReceiptSend leaves open.
            alertRepository.releaseNotify(alert.id)
        }
    }

    private fun resolveSelfie(alert: EventPhotoAlert): UserSelfie? =
        alert.selfieId?.let { userSelfieRepository.findByIdAndUserId(it, alert.userId) }
            ?: userSelfieRepository.findFirstByUserIdAndIsPrimaryTrue(alert.userId)
            ?: userSelfieRepository.findFirstByUserIdOrderByUploadedAtDesc(alert.userId)

    // Same mapping as EventPhotoController.contentTypeOf — two call sites is
    // below the extract-a-shared-util threshold.
    private fun contentTypeOf(key: String): String = when (key.substringAfterLast('.').lowercase()) {
        "jpg", "jpeg" -> MediaType.IMAGE_JPEG_VALUE
        "png" -> MediaType.IMAGE_PNG_VALUE
        "webp" -> "image/webp"
        else -> MediaType.APPLICATION_OCTET_STREAM_VALUE
    }
}
