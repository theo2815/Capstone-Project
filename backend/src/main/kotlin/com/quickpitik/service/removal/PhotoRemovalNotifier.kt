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
) {
    fun notifyReview(reviewId: UUID) {
        val requestId = reviews.findById(reviewId).orElse(null)?.requestId ?: return
        tx.execute {
            // Same case lock as verification/cancellation serializes review notices and their audit.
            // Load the review only after that lock; its pre-transaction lookup is detached (OSIV is off).
            val request = requests.findForUpdate(requestId) ?: return@execute
            val review = reviews.findByIdAndRequestId(reviewId, request.id) ?: return@execute
            if (request.status != RemovalRequestStatus.OPEN || review.notifiedAt != null || review.unavailableReason != null) return@execute
            val noticeKey = "removal:review:$reviewId:requested"
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
            val selected = items.findByRequestIdOrderByIdAsc(request.id).filter {
                it.reviewId == reviewId && it.ownerIdSnapshot == owner.id && it.state == RemovalItemState.VERIFIED
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
                    if (request.status == RemovalRequestStatus.OPEN && review.notifiedAt == null && review.unavailableReason == null) {
                        review.notificationNextAttemptAt = clock.instant().plus(properties.cleanupBackoff)
                    }
                }
            } catch (retryError: Exception) {
                log.warn("Photo removal review retry time could not be stored: {}", reviewId)
            }
            log.warn("Photo removal review notification remains pending: {}", reviewId)
        }
    }

    companion object { private val log = LoggerFactory.getLogger(PhotoRemovalNotifier::class.java) }
}
