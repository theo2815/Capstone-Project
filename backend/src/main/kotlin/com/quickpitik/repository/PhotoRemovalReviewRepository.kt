package com.quickpitik.repository

import com.quickpitik.entity.PhotoRemovalReview
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import java.time.Instant
import java.util.UUID

interface PhotoRemovalReviewRepository : JpaRepository<PhotoRemovalReview, UUID> {
    fun findByIdAndRequestId(id: UUID, requestId: UUID): PhotoRemovalReview?
    fun findByRequestIdOrderByIdAsc(requestId: UUID): List<PhotoRemovalReview>
    fun findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(requestId: UUID, ownerIdSnapshot: UUID): List<PhotoRemovalReview>

    @Query(value = """
        SELECT v.* FROM photo_removal_reviews v JOIN photo_removal_requests r ON r.id = v.request_id
        WHERE COALESCE(v.notification_next_attempt_at, r.updated_at) <= :now AND (
            (r.status = 'OPEN' AND v.notified_at IS NULL AND v.unavailable_reason IS NULL AND EXISTS
                (SELECT 1 FROM photo_removal_items i WHERE i.review_id = v.id AND (i.state = 'VERIFIED' OR photo_removal_owner_review_pending(i.id))))
            OR EXISTS (SELECT 1 FROM photo_removal_history h WHERE h.review_id = v.id
                AND h.action IN ('ADMIN_DECIDED','ERASURE_PROGRESS') AND h.metadata->>'decisionVersion' IS NOT NULL
                AND NOT EXISTS (SELECT 1 FROM photo_removal_history n WHERE n.review_id = v.id
                    AND n.action IN ('NOTICE_STORED','OWNER_UNAVAILABLE')
                    AND n.metadata->>'noticeKey' = 'removal:review:' || v.id || ':outcome:' || (h.metadata->>'decisionVersion'))))
        ORDER BY COALESCE(v.notification_next_attempt_at, r.updated_at), v.id
    """, nativeQuery = true)
    fun findPendingNotices(now: Instant, pageable: Pageable): List<PhotoRemovalReview>

    @Query("""
        SELECT v FROM PhotoRemovalReview v, PhotoRemovalRequest r
        WHERE v.requestId = r.id AND r.status IN
            (com.quickpitik.entity.RemovalRequestStatus.OPEN, com.quickpitik.entity.RemovalRequestStatus.RESOLVING)
            AND v.notifiedAt IS NOT NULL AND v.deadline <= :now
            AND v.overdue = false AND v.unavailableReason IS NULL
        ORDER BY v.deadline, v.id
    """)
    fun findDueReviews(now: Instant, pageable: Pageable): List<PhotoRemovalReview>
}
