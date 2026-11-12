package com.quickpitik.repository

import com.quickpitik.entity.PhotoRemovalItem
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.domain.Pageable
import java.time.Instant
import java.util.UUID

interface PhotoRemovalItemRepository : JpaRepository<PhotoRemovalItem, UUID> {
    @Query(value = "SELECT id FROM photo_removal_items WHERE request_id = :requestId AND photo_removal_owner_review_pending(id)", nativeQuery = true)
    fun findPendingOwnerReviewItemIds(requestId: UUID): List<UUID>

    fun findByIdAndRequestId(id: UUID, requestId: UUID): PhotoRemovalItem?
    fun findByPhotoIdSnapshotAndStateInOrderByAdminDecidedAtAscIdAsc(photoIdSnapshot: UUID,
        states: Collection<com.quickpitik.entity.RemovalItemState>): List<PhotoRemovalItem>
    fun findByRequestIdOrderByIdAsc(requestId: UUID): List<PhotoRemovalItem>
    @Query("""SELECT i FROM PhotoRemovalItem i, PhotoRemovalReview r WHERE i.requestId = :requestId
        AND i.reviewId = r.id AND r.ownerIdSnapshot = :ownerIdSnapshot ORDER BY i.id""")
    fun findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(requestId: UUID, ownerIdSnapshot: UUID): List<PhotoRemovalItem>

    @Query("SELECT DISTINCT i.requestId FROM PhotoRemovalItem i WHERE i.photoIdSnapshot IN :photoIds ORDER BY i.requestId")
    fun findOverlappingRequestIds(photoIds: Collection<UUID>, pageable: Pageable): List<UUID>

    // Auto-hides (system-verified items: no verification actor) of runner requests for one event, per photographer.
    @Query("""SELECT new com.quickpitik.repository.OwnerCount(i.ownerIdSnapshot, count(i))
        FROM PhotoRemovalItem i, PhotoRemovalRequest r WHERE i.requestId = r.id
        AND r.source = com.quickpitik.entity.RemovalRequestSource.RUNNER AND r.eventIdSnapshot = :eventId
        AND i.ownerIdSnapshot IN :ownerIds AND i.verificationActorId IS NULL AND i.verifiedAt >= :since
        GROUP BY i.ownerIdSnapshot ORDER BY i.ownerIdSnapshot""")
    fun countRunnerAutoHidesSince(eventId: UUID, ownerIds: Collection<UUID>, since: Instant): List<OwnerCount>

    // RD-5: item counts per (state, admin decision), the input of the runner status and counts.
    @Query("""SELECT new com.quickpitik.repository.RequestStateCount(i.requestId, i.state, i.adminDecision, count(i))
        FROM PhotoRemovalItem i WHERE i.requestId IN :ids GROUP BY i.requestId, i.state, i.adminDecision""")
    fun countStatesByRequestIds(ids: Collection<UUID>): List<RequestStateCount>
}

data class OwnerCount(val ownerId: UUID, val count: Long)
data class RequestStateCount(val requestId: UUID, val state: com.quickpitik.entity.RemovalItemState,
    val decision: com.quickpitik.entity.RemovalDecision?, val count: Long)
