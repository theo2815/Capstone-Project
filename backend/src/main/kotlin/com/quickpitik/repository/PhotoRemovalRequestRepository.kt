package com.quickpitik.repository

import com.quickpitik.entity.PhotoRemovalRequest
import com.quickpitik.entity.RemovalRequestStatus
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.JpaSpecificationExecutor
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import jakarta.persistence.LockModeType
import java.time.Instant
import java.util.UUID

interface PhotoRemovalRequestRepository : JpaRepository<PhotoRemovalRequest, UUID>, JpaSpecificationExecutor<PhotoRemovalRequest> {
    companion object {
        // = uq_photo_removal_requests_runner_open (V99): a runner's live request, alias `r`. A withdrawn or no-match
        // case waiting in RESOLVING for its evidence purge no longer counts.
        const val RUNNER_OPEN_SQL = "r.source = 'RUNNER' AND (r.status IN ('DRAFT','OPEN') " +
            "OR (r.status = 'RESOLVING' AND r.close_reason IS NULL))"
    }

    fun findByCreationCommandId(creationCommandId: UUID): PhotoRemovalRequest?

    @Query(value = "SELECT count(*) FROM photo_removal_requests r WHERE r.requester_user_id = :userId AND $RUNNER_OPEN_SQL",
        nativeQuery = true)
    fun countOpenRunnerRequests(userId: UUID): Long

    @Query(value = """SELECT r.id FROM photo_removal_requests r WHERE r.requester_user_id = :userId
        AND r.event_id_snapshot = :eventId AND $RUNNER_OPEN_SQL""", nativeQuery = true)
    fun findOpenRunnerRequestId(userId: UUID, eventId: UUID): UUID?

    @Query(value = """SELECT EXISTS (SELECT 1 FROM photo_removal_requests r WHERE r.requester_user_id = :userId
        AND r.event_id_snapshot = :eventId AND $RUNNER_OPEN_SQL)""", nativeQuery = true)
    fun existsOpenRunnerRequest(userId: UUID, eventId: UUID): Boolean

    // Q13 + R7: an admin kept the photos (RETAIN, nothing removed), or the case ended with every suggestion
    // dismissed. Withdraw, no-match, demote-only and an undecided cancel are not a rejection.
    @Query(value = """SELECT EXISTS (SELECT 1 FROM photo_removal_requests r WHERE r.source = 'RUNNER'
        AND r.requester_user_id = :userId AND r.event_id_snapshot = :eventId AND r.close_reason IS NULL
        AND NOT EXISTS (SELECT 1 FROM photo_removal_items i WHERE i.request_id = r.id AND i.state IN ('REMOVAL_AUTHORIZED','REMOVED'))
        AND (EXISTS (SELECT 1 FROM photo_removal_items i WHERE i.request_id = r.id AND i.admin_decision = 'RETAIN')
            OR (r.status IN ('RESOLVED','CANCELLED')
                AND EXISTS (SELECT 1 FROM photo_removal_history h WHERE h.request_id = r.id AND h.action = 'SUGGESTION_DISMISSED'))))""",
        nativeQuery = true)
    fun existsRejectedRunnerRequest(userId: UUID, eventId: UUID): Boolean

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT r FROM PhotoRemovalRequest r WHERE r.id = :id")
    fun findForUpdate(id: UUID): PhotoRemovalRequest?

    @Query("""SELECT r FROM PhotoRemovalRequest r WHERE EXISTS
        (SELECT i.id FROM PhotoRemovalItem i, PhotoRemovalReview v WHERE i.requestId = r.id AND i.reviewId = v.id AND v.ownerIdSnapshot = :ownerId)
        AND (:eventId IS NULL OR r.eventIdSnapshot = :eventId)
        AND (:status IS NULL OR (CASE WHEN r.status = com.quickpitik.entity.RemovalRequestStatus.RESOLVED AND EXISTS
            (SELECT i.id FROM PhotoRemovalItem i WHERE i.requestId = r.id AND
                (i.state = com.quickpitik.entity.RemovalItemState.REMOVAL_AUTHORIZED OR
                    (i.state = com.quickpitik.entity.RemovalItemState.REMOVED AND NOT EXISTS
                        (SELECT j.id FROM PhotoErasureJob j WHERE j.id = i.cleanupJobId AND j.photoId = i.photoIdSnapshot
                            AND j.outcome = com.quickpitik.dto.photos.PhotoErasureOutcome.COMPLETE AND j.completedAt IS NOT NULL))))
            THEN com.quickpitik.entity.RemovalRequestStatus.RESOLVING ELSE r.status END) = :status)
        AND (:needsAction IS NULL OR (CASE WHEN EXISTS
            (SELECT i.id FROM PhotoRemovalItem i, PhotoRemovalReview v WHERE i.requestId = r.id AND i.reviewId = v.id AND v.ownerIdSnapshot = :ownerId
                AND r.status = com.quickpitik.entity.RemovalRequestStatus.OPEN
                AND i.ownerResponse IS NULL AND ((i.state = com.quickpitik.entity.RemovalItemState.VERIFIED
                    AND (i.adminDecision IS NULL OR i.adminDecision = com.quickpitik.entity.RemovalDecision.MORE_EVIDENCE))
                    OR function('photo_removal_owner_review_pending', i.id) = true))
            THEN true ELSE false END) = :needsAction)
        ORDER BY r.createdAt DESC, r.id""")
    fun findForPhotographer(ownerId: UUID, eventId: UUID?, status: RemovalRequestStatus?, needsAction: Boolean?, pageable: Pageable): Page<PhotoRemovalRequest>

    // Item-only commands must also advance the aggregate version, even with a fixed Clock.
    @Modifying(flushAutomatically = true, clearAutomatically = true)
    @Query("UPDATE PhotoRemovalRequest r SET r.version = r.version + 1, r.updatedAt = :now WHERE r.id = :id")
    fun advanceVersion(id: UUID, now: Instant): Int

    @Query(value = """
        SELECT r.id FROM photo_removal_requests r WHERE r.status IN ('OPEN', 'RESOLVING', 'RESOLVED') AND (
            (r.status <> 'RESOLVED' AND EXISTS (SELECT 1 FROM photo_removal_reviews v WHERE v.request_id = r.id AND (
                (v.deadline <= :now AND v.overdue = false AND v.unavailable_reason IS NULL AND EXISTS
                    (SELECT 1 FROM photo_removal_items i WHERE i.review_id = v.id AND i.owner_response IS NULL AND
                        ((i.state = 'VERIFIED' AND (i.admin_decision IS NULL OR i.admin_decision = 'MORE_EVIDENCE'))
                            OR photo_removal_owner_review_pending(i.id))))
                OR (v.unavailable_reason IS NOT NULL AND EXISTS
                    (SELECT 1 FROM photo_removal_items i WHERE i.review_id = v.id AND
                        ((i.state = 'VERIFIED' AND (i.admin_decision IS NULL OR i.admin_decision = 'MORE_EVIDENCE'))
                            OR photo_removal_owner_review_pending(i.id)))
                    AND NOT EXISTS (SELECT 1 FROM photo_removal_history h WHERE h.request_id = r.id
                        AND h.review_id = v.id AND h.action = 'OWNER_UNAVAILABLE')))))
            OR EXISTS (SELECT 1 FROM photo_removal_items i WHERE i.request_id = r.id AND (
                (r.status <> 'RESOLVED' AND i.review_id IS NULL AND
                    ((i.state = 'VERIFIED' AND (i.admin_decision IS NULL OR i.admin_decision = 'MORE_EVIDENCE'))
                        OR photo_removal_owner_review_pending(i.id)) AND NOT EXISTS
                    (SELECT 1 FROM photo_removal_history h WHERE h.request_id = r.id AND h.item_id = i.id AND h.action = 'OWNER_UNAVAILABLE'))
                OR (i.state = 'RETAINED_RESTRICTED' AND i.retention_review_at <= :now AND NOT EXISTS
                    (SELECT 1 FROM photo_removal_history h WHERE h.request_id = r.id AND h.item_id = i.id
                        AND h.action = 'REVIEW_OVERDUE' AND h.occurred_at >= i.retention_review_at))))
        ) ORDER BY r.created_at, r.id
    """, nativeQuery = true)
    fun findDueActionRequestIds(now: Instant, pageable: Pageable): List<UUID>

    @Query(value = """SELECT r.id FROM photo_removal_requests r
        WHERE r.status IN ('OPEN','RESOLVING','RESOLVED') AND (
            EXISTS (SELECT 1 FROM photo_removal_items i JOIN photo_erasure_jobs j ON j.photo_id = i.photo_id_snapshot
                WHERE i.request_id = r.id AND ((j.outcome = 'COMPLETE' AND j.completed_at IS NOT NULL AND
                    i.state IN ('VERIFIED','REMOVAL_AUTHORIZED','RETAINED','RETAINED_RESTRICTED'))
                    OR (j.outcome <> 'COMPLETE' AND i.state = 'REMOVED')))
            OR (r.status IN ('OPEN','RESOLVING') AND EXISTS (SELECT 1 FROM photo_removal_items i WHERE i.request_id = r.id)
                AND NOT EXISTS (SELECT 1 FROM photo_removal_items i WHERE i.request_id = r.id AND
                    (i.state IN ('CANDIDATE','VERIFIED','REMOVAL_AUTHORIZED') OR photo_removal_owner_review_pending(i.id)))
                AND (r.status = 'OPEN' OR r.evidence_cleanup_state = 'DELETED')))
        ORDER BY r.updated_at, r.id""", nativeQuery = true)
    fun findReconciliationRequestIds(pageable: Pageable): List<UUID>

    fun findByEvidenceCleanupNextAttemptAtLessThanEqualOrderByEvidenceCleanupNextAttemptAtAscIdAsc(now: Instant, pageable: Pageable): List<PhotoRemovalRequest>
}
