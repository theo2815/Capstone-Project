package com.quickpitik.service.relevance

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.entity.PhotographerMessageKind
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

/** Published when a relevance note was stored (or found stored); the e-mail follows once that row has committed. */
data class RelevanceNoticeStored(val messageId: UUID)

// RV-3.1 (A.5, P20): e-mails a photographer their relevance note on the claimed-send path of removal notices
// (PhotographerRemovalEmailListener), with the same sweep cadence, stale-claim age and batch size: the send slot is
// claimed by a conditional UPDATE, delivery is stamped after Resend accepts, and a failed send leaves the dated claim
// for retryUnsentEmails. A second event for the same message finds the slot claimed and sends nothing.
@Component
class RelevanceNoticeEmailListener(
    private val messages: PhotographerMessageRepository,
    private val users: UserRepository,
    private val email: EmailService,
    private val tx: TransactionTemplate,
    private val clock: Clock,
    private val properties: PhotoRemovalProperties,
    private val switch: RelevanceSwitch,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Async("emailSending")
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    fun onNoticeStored(event: RelevanceNoticeStored) {
        try {
            sendIfPending(event.messageId)
        } catch (ex: Exception) {
            log.warn("Relevance notice email stays unsent for the sweep · messageId={}: {}", event.messageId, ex.message)
        }
    }

    @Scheduled(fixedDelayString = "\${app.photo-removal.email-sweep-interval:PT15M}")
    fun retryUnsentEmails() {
        // With the mode off nothing new runs, not even this query.
        if (switch.mode == RelevanceMode.OFF) return
        val now = OffsetDateTime.now(clock)
        val cutoff = now.minus(properties.runner.emailClaimStaleAfter)
        val stale = tx.execute { messages.findStaleRelevanceEmailClaimIds(cutoff, now.minus(Duration.ofHours(24)), properties.sweepBatchSize) }.orEmpty()
        stale.forEach { id ->
            try {
                // Release a stale claim (a never-claimed note has none); sendIfPending then claims before it sends, so a
                // copy that claimed it meanwhile, or the listener, wins and this sends nothing.
                tx.execute { messages.releaseStaleEmailSend(id, cutoff) }
                sendIfPending(id)
            } catch (ex: Exception) {
                log.warn("Relevance notice email retry failed · messageId={}: {}", id, ex.message)
            }
        }
    }

    /** Throws when Resend fails, after the claim, so the claim stays for the sweep. */
    fun sendIfPending(messageId: UUID) {
        val job = tx.execute { claim(messageId) } ?: return
        email.sendRelevanceNotice(job.to, job.name, job.title, job.body, job.caseId, job.appealFooter)
        tx.executeWithoutResult { messages.stampEmailDelivered(messageId, OffsetDateTime.now(clock)) }
    }

    private class Job(val to: String, val name: String, val title: String, val body: String, val caseId: UUID, val appealFooter: Boolean)

    private fun claim(messageId: UUID): Job? {
        val message = messages.findById(messageId).orElse(null) ?: return null
        if (message.kindWire != PhotographerMessageKind.RELEVANCE_NOTICE.wire) return null
        val caseId = message.relevanceCaseId ?: return null
        val owner = users.findById(message.photographerId).orElse(null) ?: return null
        if (messages.claimEmailSend(messageId, OffsetDateTime.now(clock)) == 0) return null
        // V2-T21: the appeal footer goes with V2-T11 to V2-T14 and V2-T23, the notes keyed ":verdict".
        return Job(owner.email, owner.name, message.title.orEmpty(), message.body, caseId,
            message.noticeKey?.endsWith(":verdict") == true)
    }
}
