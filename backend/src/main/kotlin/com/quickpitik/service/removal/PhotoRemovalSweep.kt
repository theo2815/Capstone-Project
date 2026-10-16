package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import org.springframework.context.ApplicationEventPublisher
import org.springframework.data.domain.PageRequest
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.util.UUID

@Service
@Transactional(propagation = Propagation.NEVER)
class PhotoRemovalSweep(
    private val requests: PhotoRemovalRequestRepository,
    private val reviews: PhotoRemovalReviewRepository,
    private val items: PhotoRemovalItemRepository,
    private val audit: PhotoRemovalHistoryRepository,
    private val history: PhotoRemovalHistoryRecorder,
    private val publisher: ApplicationEventPublisher,
    private val tx: TransactionTemplate,
    private val clock: Clock,
    private val properties: PhotoRemovalProperties,
    private val service: PhotoRemovalService,
) {
    @Scheduled(fixedDelayString = "\${app.photo-removal.cleanup-interval:PT1M}")
    fun sweep() { sweepDue(clock.instant(), properties.sweepBatchSize) }

    fun sweepDue(now: Instant, limit: Int): Int {
        if (limit <= 0) return 0
        val bound = limit.coerceAtMost(100)
        val due = tx.execute { requests.findDueActionRequestIds(now, PageRequest.of(0, bound)) }!!
        var processed = 0
        for (requestId in due) {
            if (processed == bound) break
            val remaining = bound - processed
            processed += tx.execute {
                val request = requests.findForUpdate(requestId) ?: return@execute 0
                if (request.status !in setOf(RemovalRequestStatus.OPEN, RemovalRequestStatus.RESOLVING, RemovalRequestStatus.RESOLVED)) return@execute 0
                val selected = items.findByRequestIdOrderByIdAsc(requestId)
                val pendingOwnerReviews = items.findPendingOwnerReviewItemIds(requestId).toSet()
                var actions = 0
                fun record(kind: String, item: PhotoRemovalItem? = null, review: PhotoRemovalReview? = null,
                    dueAt: Instant? = null, reason: String): Boolean {
                    if (actions == remaining) return false
                    val reviewId = review?.id ?: item?.reviewId
                    val action = if (kind == "OWNER_UNAVAILABLE") RemovalHistoryActions.OWNER_UNAVAILABLE else RemovalHistoryActions.REVIEW_OVERDUE
                    if (action == RemovalHistoryActions.OWNER_UNAVAILABLE &&
                        audit.existsByRequestIdAndItemIdAndReviewIdAndAction(requestId, item?.id, reviewId, action)) return false
                    val key = "removal:admin:$kind:${item?.id ?: review!!.id}:${dueAt ?: ""}"
                    val commandId = UUID.nameUUIDFromBytes(key.toByteArray(Charsets.UTF_8))
                    if (audit.findByRequestIdAndCommandId(requestId, commandId).isNotEmpty()) return false
                    history.record(RemovalHistoryEvent(requestId, item?.id, reviewId, item?.photoIdSnapshot,
                        RemovalActorType.SYSTEM, action = action, reason = reason,
                        occurredAt = now, commandId = commandId, metadata = mapOf("kind" to kind)))
                    actions++
                    return true
                }
                for (review in reviews.findByRequestIdOrderByIdAsc(requestId)) {
                    if (request.status == RemovalRequestStatus.RESOLVED) break
                    val undecided = selected.filter { it.reviewId == review.id && it.awaitingDecision(pendingOwnerReviews) }
                    if (undecided.isEmpty()) continue
                    if (review.unavailableReason != null) {
                        record("OWNER_UNAVAILABLE", review = review, reason = "Owner unavailable; admin action required")
                    } else if (!review.overdue && review.notifiedAt != null && review.deadline?.let { it <= now } == true &&
                        undecided.any { it.ownerResponse == null }) {
                        if (record("RESPONSE_OVERDUE", review = review, reason = "Owner response overdue; admin action required")) {
                            review.overdue = true
                            review.overdueAt = now
                        }
                    }
                }
                for (item in selected) {
                    if (request.status != RemovalRequestStatus.RESOLVED && item.reviewId == null && item.awaitingDecision(pendingOwnerReviews))
                        record("OWNER_UNAVAILABLE", item = item, reason = "Photo has no assigned owner; admin action required")
                    if (item.state == RemovalItemState.RETAINED_RESTRICTED && item.retentionReviewAt?.let { it <= now } == true)
                        record("RETENTION_DUE", item = item, dueAt = item.retentionReviewAt, reason = "Restricted retention review due; admin action required")
                }
                if (actions > 0) {
                    reviews.flush()
                    requests.advanceVersion(requestId, now)
                    publisher.publishEvent(removalChangedEvent(requestId, null, now))
                }
                actions
            }!!
        }
        if (processed < bound) {
            val pending = tx.execute { requests.findReconciliationRequestIds(PageRequest.of(0, bound - processed)) }!!
            pending.forEach { id ->
                runCatching { service.reconcile(id) }.onFailure {
                    org.slf4j.LoggerFactory.getLogger(javaClass).warn("Removal reconciliation deferred for case {} ({})", id, it.javaClass.simpleName)
                }
                processed++
            }
        }
        return processed
    }
}

internal fun PhotoRemovalItem.awaitingDecision(pendingOwnerReviews: Set<UUID> = emptySet()) =
    (state == RemovalItemState.VERIFIED && adminDecision in setOf(null, RemovalDecision.MORE_EVIDENCE)) || id in pendingOwnerReviews
