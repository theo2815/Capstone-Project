package com.quickpitik.service.removal

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Component
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.time.Instant
import java.time.temporal.ChronoUnit
import java.util.UUID

data class RemovalHistoryEvent(
    val requestId: UUID, val itemId: UUID? = null, val reviewId: UUID? = null,
    val photoIdSnapshot: UUID? = null, val actorType: RemovalActorType, val actorId: UUID? = null,
    val action: String, val fromState: String? = null, val toState: String? = null, val reason: String? = null,
    val occurredAt: Instant, val commandId: UUID, val metadata: Map<String, String> = emptyMap(),
)

object RemovalHistoryActions {
    const val CREATED = "CREATED"
    const val EVIDENCE_REPLACED = "EVIDENCE_REPLACED"
    const val VERIFIED = "VERIFIED"
    const val NOTICE_STORED = "NOTICE_STORED"
    const val OWNER_UNAVAILABLE = "OWNER_UNAVAILABLE"
    const val OWNER_RESPONDED = "OWNER_RESPONDED"
    const val REVIEW_OVERDUE = "REVIEW_OVERDUE"
    const val ADMIN_DECIDED = "ADMIN_DECIDED"
    const val ERASURE_PROGRESS = "ERASURE_PROGRESS"
    const val EVIDENCE_PURGED = "EVIDENCE_PURGED"
    const val SUPPORT_UPDATED = "SUPPORT_UPDATED"
    const val CANCEL_REQUESTED = "CANCEL_REQUESTED"
    const val RESOLVED = "RESOLVED"
    const val CANCELLED = "CANCELLED"
    val ALL = setOf(CREATED, EVIDENCE_REPLACED, VERIFIED, NOTICE_STORED, OWNER_UNAVAILABLE, OWNER_RESPONDED,
        REVIEW_OVERDUE, ADMIN_DECIDED, ERASURE_PROGRESS, EVIDENCE_PURGED, SUPPORT_UPDATED, CANCEL_REQUESTED, RESOLVED, CANCELLED)
}

@Component
class PhotoRemovalHistoryRecorder(
    private val history: PhotoRemovalHistoryRepository,
    private val requests: PhotoRemovalRequestRepository,
    private val items: PhotoRemovalItemRepository,
    private val reviews: PhotoRemovalReviewRepository,
    private val mapper: ObjectMapper,
) {
    @Transactional(propagation = Propagation.MANDATORY)
    fun record(event: RemovalHistoryEvent) {
        val stored = history.findByRequestIdAndCommandIdAndItemIdAndReviewId(event.requestId, event.commandId, event.itemId, event.reviewId)
        if (stored != null) {
            ensureIdentical(stored, event)
            return
        }
        if (event.action !in RemovalHistoryActions.ALL || event.reason?.isBlank() == true ||
            (event.fromState?.length ?: 0) > 24 || (event.toState?.length ?: 0) > 24 ||
            event.metadata.size > 32 || event.metadata.any { (key, value) ->
                key.length > 64 || value.length > 256 || value.contains("://") ||
                    listOf("selfie", "embedding", "bytes", "payload", "signedurl", "evidence", "secret", "contact", "token")
                        .any { it in key.lowercase().replace("_", "").replace("-", "") }
            }) {
            throw ApiException(HttpStatus.BAD_REQUEST, ErrorCodes.VALIDATION_ERROR, "Invalid photo removal history content")
        }
        if (!requests.existsById(event.requestId)) notFound()
        val item = event.itemId?.let { items.findByIdAndRequestId(it, event.requestId) ?: notFound() }
        if (event.reviewId != null && reviews.findByIdAndRequestId(event.reviewId, event.requestId) == null) notFound()
        if (item != null && (event.photoIdSnapshot != item.photoIdSnapshot ||
                (event.reviewId != null && event.reviewId != item.reviewId))) conflict()

        // PostgreSQL stores microseconds; identical retries must use that same precision.
        val inserted = history.append(UUID.randomUUID(), event.requestId, event.itemId, event.reviewId, event.photoIdSnapshot,
            event.actorType.name, event.actorId, event.action, event.fromState, event.toState, event.reason,
            event.occurredAt.truncatedTo(ChronoUnit.MICROS), event.commandId, mapper.writeValueAsString(event.metadata))
        if (inserted == 0) {
            ensureIdentical(history.findByRequestIdAndCommandIdAndItemIdAndReviewId(event.requestId, event.commandId,
                event.itemId, event.reviewId) ?: conflict(), event)
        }
    }

    private fun ensureIdentical(stored: PhotoRemovalHistory, event: RemovalHistoryEvent) {
        if (stored.photoIdSnapshot != event.photoIdSnapshot || stored.actorType != event.actorType ||
            stored.actorIdSnapshot != event.actorId || stored.action != event.action || stored.fromState != event.fromState ||
            stored.toState != event.toState || stored.reason != event.reason || stored.metadata != event.metadata ||
            stored.occurredAt != event.occurredAt.truncatedTo(ChronoUnit.MICROS)) conflict()
    }

    private fun conflict(): Nothing = throw ApiException(HttpStatus.CONFLICT, ErrorCodes.CONFLICT, "Photo removal command content changed")
    private fun notFound(): Nothing = throw ApiException(HttpStatus.NOT_FOUND, ErrorCodes.NOT_FOUND, "Photo removal record not found")
}
