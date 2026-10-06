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
    fun findByCreationCommandId(creationCommandId: UUID): PhotoRemovalRequest?

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT r FROM PhotoRemovalRequest r WHERE r.id = :id")
    fun findForUpdate(id: UUID): PhotoRemovalRequest?

    @Query("""SELECT r FROM PhotoRemovalRequest r WHERE EXISTS
        (SELECT i.id FROM PhotoRemovalItem i WHERE i.requestId = r.id AND i.ownerIdSnapshot = :ownerId)
        AND (:eventId IS NULL OR r.eventIdSnapshot = :eventId) AND (:status IS NULL OR r.status = :status)
        AND (:needsAction IS NULL OR (CASE WHEN EXISTS
            (SELECT i.id FROM PhotoRemovalItem i WHERE i.requestId = r.id AND i.ownerIdSnapshot = :ownerId
                AND r.status = com.quickpitik.entity.RemovalRequestStatus.OPEN
                AND i.state = com.quickpitik.entity.RemovalItemState.VERIFIED AND i.ownerResponse IS NULL
                AND (i.adminDecision IS NULL OR i.adminDecision = com.quickpitik.entity.RemovalDecision.MORE_EVIDENCE))
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
                    (SELECT 1 FROM photo_removal_items i WHERE i.review_id = v.id AND i.state = 'VERIFIED'
                        AND i.owner_response IS NULL AND (i.admin_decision IS NULL OR i.admin_decision = 'MORE_EVIDENCE')))
                OR (v.unavailable_reason IS NOT NULL AND EXISTS
                    (SELECT 1 FROM photo_removal_items i WHERE i.review_id = v.id AND i.state = 'VERIFIED'
                        AND (i.admin_decision IS NULL OR i.admin_decision = 'MORE_EVIDENCE'))
                    AND NOT EXISTS (SELECT 1 FROM photo_removal_history h WHERE h.request_id = r.id
                        AND h.review_id = v.id AND h.action = 'OWNER_UNAVAILABLE')))))
            OR EXISTS (SELECT 1 FROM photo_removal_items i WHERE i.request_id = r.id AND (
                (r.status <> 'RESOLVED' AND i.state = 'VERIFIED' AND i.owner_id_snapshot IS NULL
                    AND (i.admin_decision IS NULL OR i.admin_decision = 'MORE_EVIDENCE') AND NOT EXISTS
                    (SELECT 1 FROM photo_removal_history h WHERE h.request_id = r.id AND h.item_id = i.id AND h.action = 'OWNER_UNAVAILABLE'))
                OR (i.state = 'RETAINED_RESTRICTED' AND i.retention_review_at <= :now AND NOT EXISTS
                    (SELECT 1 FROM photo_removal_history h WHERE h.request_id = r.id AND h.item_id = i.id
                        AND h.action = 'REVIEW_OVERDUE' AND h.occurred_at >= i.retention_review_at))))
        ) ORDER BY r.created_at, r.id
    """, nativeQuery = true)
    fun findDueActionRequestIds(now: Instant, pageable: Pageable): List<UUID>

    fun findByEvidenceCleanupNextAttemptAtLessThanEqualOrderByEvidenceCleanupNextAttemptAtAscIdAsc(now: Instant, pageable: Pageable): List<PhotoRemovalRequest>
}
