package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.repository.PhotoRemovalRequestRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import org.slf4j.LoggerFactory
import org.springframework.scheduling.annotation.Async
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID

/** Published only when a removal notice row was newly stored; the email follows once that row has committed. */
data class PhotographerRemovalNoticeStored(val messageId: UUID)

// Emails a photographer about their own removal notices (RD-10, every removal case). The send slot is claimed with a
// conditional UPDATE (as the order receipt), delivery is stamped after Resend accepts, and a failed send leaves the
// dated claim for retryUnsentEmails.
@Component
class PhotographerRemovalEmailListener(
    private val messages: PhotographerMessageRepository,
    private val users: UserRepository,
    private val requests: PhotoRemovalRequestRepository,
    private val email: EmailService,
    private val tx: TransactionTemplate,
    private val clock: Clock,
    private val properties: PhotoRemovalProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("emailSending")
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onNoticeStored(event: PhotographerRemovalNoticeStored) {
        try {
            sendIfPending(event.messageId)
        } catch (ex: Exception) {
            log.warn("Photographer removal email stays unsent for the sweep · messageId={}: {}", event.messageId, ex.message)
        }
    }

    @Scheduled(fixedDelayString = "\${app.photo-removal.email-sweep-interval:PT15M}")
    fun retryUnsentEmails() {
        val now = OffsetDateTime.now(clock)
        val cutoff = now.minus(properties.runner.emailClaimStaleAfter)
        val stale = tx.execute { messages.findStaleEmailClaimIds(cutoff, now.minus(Duration.ofHours(24)), properties.sweepBatchSize) }.orEmpty()
        stale.forEach { id ->
            try {
                if (tx.execute { messages.releaseStaleEmailSend(id, cutoff) } == 1) sendIfPending(id)
            } catch (ex: Exception) {
                log.warn("Photographer removal email retry failed · messageId={}: {}", id, ex.message)
            }
        }
    }

    /** Throws when Resend fails, after the claim, so the claim stays for the sweep. */
    fun sendIfPending(messageId: UUID) {
        val job = tx.execute { claim(messageId) } ?: return
        when (job.kind) {
            PhotographerMessageKind.PHOTO_REMOVAL_REQUEST -> email.sendPhotographerRemovalReviewEmail(job.to, job.eventName, job.body, job.requestId)
            else -> email.sendPhotographerRemovalOutcomeEmail(job.to, job.eventName, job.body, job.requestId)
        }
        tx.executeWithoutResult { messages.stampEmailDelivered(messageId, OffsetDateTime.now(clock)) }
    }

    private class Job(val kind: PhotographerMessageKind, val to: String, val eventName: String, val body: String, val requestId: UUID)

    private fun claim(messageId: UUID): Job? {
        val message = messages.findById(messageId).orElse(null) ?: return null
        val kind = listOf(PhotographerMessageKind.PHOTO_REMOVAL_REQUEST, PhotographerMessageKind.PHOTO_REMOVAL_OUTCOME)
            .firstOrNull { it.wire == message.kindWire } ?: return null
        val requestId = message.photoRemovalRequestId ?: return null
        val request = requests.findById(requestId).orElse(null) ?: return null
        val owner = users.findById(message.photographerId).orElse(null) ?: return null
        if (messages.claimEmailSend(messageId, OffsetDateTime.now(clock)) == 0) return null
        return Job(kind, owner.email, request.eventNameSnapshot, message.body, requestId)
    }
}
