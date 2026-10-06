package com.quickpitik.service.removal

import com.quickpitik.common.*
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import org.springframework.data.domain.Sort
import org.springframework.data.jpa.domain.Specification
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

@Service
@Transactional(readOnly = true)
class PhotoRemovalQueryService(
    private val requests: PhotoRemovalRequestRepository,
    private val items: PhotoRemovalItemRepository,
    private val reviews: PhotoRemovalReviewRepository,
    private val history: PhotoRemovalHistoryRepository,
) {
    fun adminDetail(adminId: UUID, requestId: UUID): PhotoRemovalRequestDto {
        val request = requests.findById(requestId).orElse(null) ?: removalNotFound()
        val selected = items.findByRequestIdOrderByIdAsc(requestId)
        return request.toAdminDto(selected, reviews.findByRequestIdOrderByIdAsc(requestId))
    }

    fun list(adminId: UUID, eventId: UUID?, status: RemovalRequestStatus?, needsAction: Boolean?, offset: Int?, limit: Int?): RemovalPage<PhotoRemovalRequestDto> {
        val params = removalPagination(offset, limit)
        val filter = Specification<PhotoRemovalRequest> { root, _, cb ->
            val predicates = mutableListOf<jakarta.persistence.criteria.Predicate>()
            eventId?.let { predicates += cb.equal(root.get<UUID>("eventIdSnapshot"), it) }
            status?.let { predicates += cb.equal(root.get<RemovalRequestStatus>("status"), it) }
            needsAction?.let {
                val active = root.get<RemovalRequestStatus>("status").`in`(RemovalRequestStatus.DRAFT, RemovalRequestStatus.OPEN, RemovalRequestStatus.RESOLVING)
                predicates += if (it) active else cb.not(active)
            }
            cb.and(*predicates.toTypedArray())
        }
        val page = requests.findAll(filter, OffsetLimitPageable(params, Sort.by(Sort.Direction.DESC, "createdAt").and(Sort.by("id"))))
        return RemovalPage(page.content.map { it.toAdminDto(items.findByRequestIdOrderByIdAsc(it.id), emptyList()).copy(items = emptyList()) },
            page.totalElements, params.offset, params.limit)
    }

    fun history(adminId: UUID, requestId: UUID, offset: Int?, limit: Int?): RemovalPage<RemovalHistoryDto> {
        if (!requests.existsById(requestId)) removalNotFound()
        val params = removalPagination(offset, limit)
        val rows = history.findByRequestIdOrderByOccurredAtAscIdAsc(requestId, OffsetLimitPageable(params))
        return RemovalPage(rows.map { RemovalHistoryDto(it.id, it.requestId, it.itemId, it.reviewId, it.photoIdSnapshot,
            it.actorType, it.actorIdSnapshot, it.action, it.fromState, it.toState, it.reason, it.occurredAt, it.commandId, it.metadata) },
            history.countByRequestId(requestId), params.offset, params.limit)
    }
}

internal fun removalPagination(offset: Int?, limit: Int?) = PaginationParams((offset ?: 0).coerceAtLeast(0), (limit ?: 50).coerceIn(1, 100))
internal fun removalNotFound(): Nothing = throw ApiException(HttpStatus.NOT_FOUND, ErrorCodes.NOT_FOUND, "Photo removal request not found")
internal fun removalConflict(): Nothing = throw ApiException(HttpStatus.CONFLICT, ErrorCodes.CONFLICT, "Photo removal command changed or is still in progress")
internal fun removalInvalid(message: String): Nothing = throw ApiException(HttpStatus.BAD_REQUEST, ErrorCodes.VALIDATION_ERROR, message)

internal fun PhotoRemovalRequest.toAdminDto(items: List<PhotoRemovalItem>, reviews: List<PhotoRemovalReview>): PhotoRemovalRequestDto {
    fun count(state: RemovalItemState) = items.count { it.state == state }.toLong()
    return PhotoRemovalRequestDto(id, RemovalEventDto(eventIdSnapshot, eventNameSnapshot), status, version, createdAt, updatedAt,
        resolvedAt, RemovalPhaseCounts(count(RemovalItemState.CANDIDATE), count(RemovalItemState.VERIFIED), count(RemovalItemState.EXCLUDED),
            count(RemovalItemState.REMOVAL_AUTHORIZED), count(RemovalItemState.REMOVED), count(RemovalItemState.RETAINED), count(RemovalItemState.RETAINED_RESTRICTED)),
        requesterUserId, requesterContact, supportReference, requesterReason, publicToOwnerReason, creatorId, verifierId,
        verificationMethod, verificationNote, verifiedAt, evidenceKey != null, evidenceContentType, evidenceGeneration,
        evidenceCleanupState, outstandingEvidenceKeys.size,
        items.map { PhotoRemovalItemDto(it.id, it.photoIdSnapshot, it.ownerIdSnapshot, it.ownerNameSnapshot, it.reviewId, it.state,
            it.matchSource, it.matchScore, it.verificationActorId, it.verifiedAt, it.verificationReason, it.ownerResponse,
            it.ownerResponseReason, it.ownerRespondedAt, it.adminDecision, it.adminDecisionReason, it.adminDecisionActorId,
            it.adminDecidedAt, it.retentionReviewAt, it.cleanupJobId, null) },
        reviews.map { PhotoRemovalReviewDto(it.id, it.ownerIdSnapshot, it.notifiedMessageId, it.notifiedAt, it.deadline,
            it.overdue, it.overdueAt, it.unavailableReason) })
}
