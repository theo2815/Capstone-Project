package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.PhotoRemovalMessageLink
import org.slf4j.LoggerFactory
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.ZoneOffset
import java.util.UUID

@Service
@Transactional(propagation = Propagation.NEVER)
class PhotoRemovalNotifier(
    private val requests: PhotoRemovalRequestRepository,
    private val reviews: PhotoRemovalReviewRepository,
    private val items: PhotoRemovalItemRepository,
    private val users: UserRepository,
    private val messages: PhotographerMessageRepository,
    private val writer: AdminDecisionLogService,
    private val history: PhotoRemovalHistoryRecorder,
    private val tx: TransactionTemplate,
    private val clock: Clock,
    private val properties: PhotoRemovalProperties,
    private val audit: PhotoRemovalHistoryRepository,
) {
    fun notifyReview(reviewId: UUID) {
        val requestId = reviews.findById(reviewId).orElse(null)?.requestId ?: return
        tx.execute {
            // Same case lock as verification/cancellation serializes review notices and their audit.
            // Load the review only after that lock; its pre-transaction lookup is detached (OSIV is off).
            val request = requests.findForUpdate(requestId) ?: return@execute
            val review = reviews.findByIdAndRequestId(reviewId, request.id) ?: return@execute
            if (request.status != RemovalRequestStatus.OPEN || review.notifiedAt != null || review.unavailableReason != null) return@execute
            val noticeVersion = audit.findReviewNoticeVersion(reviewId)
            val noticeKey = "removal:review:$reviewId:requested" + (noticeVersion?.let { ":$it" } ?: "")
            val commandId = UUID.nameUUIDFromBytes(noticeKey.toByteArray(Charsets.UTF_8))
            val owner = review.ownerId?.let { users.findById(it).orElse(null) }
            if (owner == null || owner.role != Role.PHOTOGRAPHER || owner.suspendedAt != null) {
                review.unavailableReason = if (owner == null) "Owner account unavailable" else "Owner account inactive"
                review.notificationNextAttemptAt = null
                reviews.flush()
                history.record(RemovalHistoryEvent(requestId = request.id, reviewId = reviewId, actorType = RemovalActorType.SYSTEM,
                    action = RemovalHistoryActions.OWNER_UNAVAILABLE, reason = review.unavailableReason,
                    occurredAt = clock.instant(), commandId = commandId))
                return@execute
            }
            val pendingOwnerReviews = items.findPendingOwnerReviewItemIds(request.id).toSet()
            val selected = items.findByRequestIdOrderByIdAsc(request.id).filter {
                it.reviewId == reviewId && it.awaitingDecision(pendingOwnerReviews)
            }
            if (selected.isEmpty()) return@execute
            val message = messages.findByNoticeKey(noticeKey) ?: writer.pushMessage(owner.id, PhotographerMessageKind.PHOTO_REMOVAL_REQUEST,
                body = "${request.publicToOwnerReason}. Please review these selected items: ${selected.joinToString(", ") { it.id.toString() }}.",
                title = "Photo removal review requested",
                photoRemovalLink = PhotoRemovalMessageLink(request.id, noticeKey, clock.instant().atOffset(ZoneOffset.UTC)))
            review.notifiedMessageId = message.id
            review.notifiedAt = message.createdAt.toInstant()
            review.deadline = review.notifiedAt!!.plus(properties.photographerResponseWindow)
            review.notificationNextAttemptAt = null
            reviews.flush()
            history.record(RemovalHistoryEvent(requestId = request.id, reviewId = reviewId, actorType = RemovalActorType.SYSTEM,
                action = RemovalHistoryActions.NOTICE_STORED, occurredAt = review.notifiedAt!!, commandId = commandId,
                metadata = mapOf("messageId" to message.id.toString())))
        }
        notifyOutcomes(reviewId)
    }

    internal fun notifyOutcomes(reviewId: UUID) {
        val requestId = reviews.findById(reviewId).orElse(null)?.requestId ?: return
        tx.executeWithoutResult {
            val request = requests.findForUpdate(requestId) ?: return@executeWithoutResult
            val review = reviews.findByIdAndRequestId(reviewId, requestId) ?: return@executeWithoutResult
            val pending = audit.findPendingOutcomes(reviewId, PageRequest.of(0, 100))
            val currentIds = items.findByRequestIdOrderByIdAsc(requestId).filter { it.reviewId == reviewId }.map { it.id }.toSet()
            val owner = review.ownerId?.let { users.findById(it).orElse(null) }
            pending.groupBy { it.metadata.getValue("decisionVersion") }.forEach { (version, changes) ->
                val key = "removal:review:$reviewId:outcome:$version"
                val commandId = UUID.nameUUIDFromBytes(key.toByteArray(Charsets.UTF_8))
                val scoped = changes.filter { it.itemId in currentIds }
                val available = owner != null && owner.role == Role.PHOTOGRAPHER && owner.suspendedAt == null && scoped.isNotEmpty()
                val message = if (available) messages.findByNoticeKey(key) ?: writer.pushMessage(owner!!.id,
                    PhotographerMessageKind.PHOTO_REMOVAL_OUTCOME,
                    body = scoped.joinToString("; ") { "${it.itemId}: ${outcomeLabel(it.toState)}" },
                    title = "Photo removal review outcome", photoRemovalLink = PhotoRemovalMessageLink(request.id, key, clock.instant().atOffset(ZoneOffset.UTC))) else null
                history.record(RemovalHistoryEvent(requestId, reviewId = reviewId, actorType = RemovalActorType.SYSTEM,
                    action = if (message != null) RemovalHistoryActions.NOTICE_STORED else RemovalHistoryActions.OWNER_UNAVAILABLE,
                    reason = if (message != null) null else "Outcome cannot be delivered to this current owner review",
                    occurredAt = clock.instant(), commandId = commandId,
                    metadata = mapOf("kind" to "photographer_removal_outcome", "noticeKey" to key) +
                        (message?.let { mapOf("messageId" to it.id.toString()) } ?: emptyMap())))
            }
            review.notificationNextAttemptAt = clock.instant()
        }
    }

    private fun outcomeLabel(state: String?) = when (state) {
        "REMOVED" -> "Platform cleanup confirmed"
        "REMOVAL_AUTHORIZED" -> "Removal authorized; cleanup pending"
        "RETAINED" -> "Retained; this request's restriction released"
        "RETAINED_RESTRICTED" -> "Retained under a restricted exception; not erased"
        else -> "More evidence requested; restriction remains"
    }

    @Scheduled(fixedDelayString = "\${app.photo-removal.cleanup-interval:PT1M}")
    fun retryPending() {
        val due = tx.execute { reviews.findPendingNotices(clock.instant(), PageRequest.of(0, 50)).map { it.id } }!!
        due.forEach(::attempt)
    }

    internal fun attempt(reviewId: UUID) {
        try { notifyReview(reviewId) } catch (error: Exception) {
            // The transaction rolled back: no delivered notice/deadline; the bounded sweep retries this review.
            try {
                val requestId = reviews.findById(reviewId).orElse(null)?.requestId ?: return
                tx.execute {
                    val request = requests.findForUpdate(requestId) ?: return@execute
                    val review = reviews.findByIdAndRequestId(reviewId, request.id) ?: return@execute
                    review.notificationNextAttemptAt = clock.instant().plus(properties.cleanupBackoff)
                }
            } catch (retryError: Exception) {
                log.warn("Photo removal review retry time could not be stored: {}", reviewId)
            }
            log.warn("Photo removal review notification remains pending: {}", reviewId)
        }
    }

    companion object { private val log = LoggerFactory.getLogger(PhotoRemovalNotifier::class.java) }
}
