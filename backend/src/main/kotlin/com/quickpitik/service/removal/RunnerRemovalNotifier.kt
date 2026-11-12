package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.RemovalRequestSource
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.repository.PhotoRemovalRequestRepository
import com.quickpitik.repository.RunnerMessageRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.websocket.AdminInboxEvent
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.scheduling.annotation.Async
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Component
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID

/** Published only when a runner status notice was newly stored; the email follows once that row has committed. */
data class RunnerRemovalNoticeStored(val messageId: UUID)

// RD-9: a runner gets one inbox message and one email each time their deletion request reaches a new status.
// The notice key carries the status, so a repeated change event stores nothing; the email claim and sweep copy
// PhotographerRemovalEmailListener (RD-10).
@Component
class RunnerRemovalNotifier(
    private val requests: PhotoRemovalRequestRepository,
    private val runnerQueries: RunnerPhotoRemovalQueryService,
    private val runnerMessages: RunnerMessagesService,
    private val messages: RunnerMessageRepository,
    private val users: UserRepository,
    private val email: EmailService,
    private val publisher: ApplicationEventPublisher,
    private val tx: TransactionTemplate,
    private val clock: Clock,
    private val properties: PhotoRemovalProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    fun onRemovalChanged(event: AdminInboxEvent) {
        if (event.payload["type"] != "photo_removal_changed") return
        val requestId = UUID.fromString(event.payload["entityId"] as String)
        val request = requests.findById(requestId).orElse(null) ?: return
        if (request.source != RemovalRequestSource.RUNNER) return
        val runnerId = request.requesterUserId ?: return // the runner deleted their account; the case carries on
        val detail = runnerQueries.detailOf(request)
        val status = RunnerRemovalStatus.valueOf(detail.status)
        val (title, body) = runnerRemovalCopy(status, detail.counts.hidden)
        val messageId = runnerMessages.pushRemovalUpdate(runnerId, requestId, "removal:runner:$requestId:$status", title, body) ?: return
        publisher.publishEvent(RunnerRemovalNoticeStored(messageId))
    }

    @Async("emailSending")
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onNoticeStored(event: RunnerRemovalNoticeStored) {
        try {
            sendIfPending(event.messageId)
        } catch (ex: Exception) {
            log.warn("Runner removal email stays unsent for the sweep · messageId={}: {}", event.messageId, ex.message)
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
                log.warn("Runner removal email retry failed · messageId={}: {}", id, ex.message)
            }
        }
    }

    /** Throws when Resend fails, after the claim, so the claim stays for the sweep. */
    fun sendIfPending(messageId: UUID) {
        val job = tx.execute { claim(messageId) } ?: return
        email.sendRunnerRemovalUpdateEmail(job.to, job.eventName, job.title, job.body, job.requestId)
        tx.executeWithoutResult { messages.stampEmailDelivered(messageId, OffsetDateTime.now(clock)) }
    }

    private class Job(val to: String, val eventName: String, val title: String, val body: String, val requestId: UUID)

    private fun claim(messageId: UUID): Job? {
        val message = messages.findById(messageId).orElse(null) ?: return null
        if (message.kindWire != RunnerMessageKind.PHOTO_REMOVAL_UPDATE.wire) return null
        val requestId = message.photoRemovalRequestId ?: return null
        val request = requests.findById(requestId).orElse(null) ?: return null
        val runner = users.findById(message.runnerId).orElse(null) ?: return null
        if (messages.claimEmailSend(messageId, OffsetDateTime.now(clock)) == 0) return null
        return Job(runner.email, request.eventNameSnapshot, message.title.orEmpty(), message.body, requestId)
    }
}

/** Title and body per status: counts only, never photos, photographers, scores or the runner's reason. */
internal fun runnerRemovalCopy(status: RunnerRemovalStatus, hidden: Int): Pair<String, String> = when (status) {
    RunnerRemovalStatus.PROCESSING -> "We are processing your photo deletion request" to
        "We could not search the event's photos yet. Our team will check your request and update you here."
    RunnerRemovalStatus.PENDING_ADMIN_REVIEW -> "Your photo deletion request is with our team" to
        if (hidden == 0) "Our team is reviewing your request."
        else "$hidden ${if (hidden == 1) "photo is" else "photos are"} hidden while we review your request."
    RunnerRemovalStatus.ADDITIONAL_VERIFICATION_REQUIRED -> "We need to check your request further" to
        "Our team needs to verify your request before acting on it. We will update you here."
    RunnerRemovalStatus.NO_MATCHES_FOUND -> "We found no photos of you in this event" to
        "Nothing was hidden. If you still see a photo of yourself, you can ask us by email."
    RunnerRemovalStatus.APPROVED -> "Your photo deletion request was approved" to
        "We are deleting the photos in your request. They stay hidden until this is done."
    RunnerRemovalStatus.COMPLETED -> "Your photos have been deleted" to "The photos in your request have been deleted."
    RunnerRemovalStatus.REJECTED -> "We kept the photos in your request" to
        "After review, we kept the photos in your request. You can reply by email within 14 days."
    RunnerRemovalStatus.WITHDRAWN -> "You withdrew your photo deletion request" to
        "Your request is closed. You can send a new one for this event at any time."
}
