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

    @Query("""
        SELECT v FROM PhotoRemovalReview v, PhotoRemovalRequest r
        WHERE v.requestId = r.id AND r.status = com.quickpitik.entity.RemovalRequestStatus.OPEN
            AND COALESCE(v.notificationNextAttemptAt, r.updatedAt) <= :now
            AND v.notifiedAt IS NULL AND v.unavailableReason IS NULL
        ORDER BY COALESCE(v.notificationNextAttemptAt, r.updatedAt), v.id
    """)
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
