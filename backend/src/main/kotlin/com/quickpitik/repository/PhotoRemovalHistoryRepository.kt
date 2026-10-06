package com.quickpitik.repository

import com.quickpitik.entity.PhotoRemovalHistory
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.Repository
import java.time.Instant
import java.util.UUID

// Deliberately excludes JpaRepository's save/update/delete surface.
interface PhotoRemovalHistoryRepository : Repository<PhotoRemovalHistory, UUID> {
    fun findByRequestIdAndCommandId(requestId: UUID, commandId: UUID): List<PhotoRemovalHistory>
    fun findByRequestIdAndCommandIdAndItemIdAndReviewId(requestId: UUID, commandId: UUID, itemId: UUID?, reviewId: UUID?): PhotoRemovalHistory?
    fun findByRequestIdOrderByOccurredAtAscIdAsc(requestId: UUID, pageable: Pageable): List<PhotoRemovalHistory>
    fun countByRequestId(requestId: UUID): Long
    fun existsByRequestIdAndAction(requestId: UUID, action: String): Boolean
    fun existsByRequestIdAndItemIdAndReviewIdAndAction(requestId: UUID, itemId: UUID?, reviewId: UUID?, action: String): Boolean

    @Modifying
    @Query(value = """
        INSERT INTO photo_removal_history (id, request_id, item_id, review_id, photo_id_snapshot,
            actor_type, actor_id_snapshot, action, from_state, to_state, reason, occurred_at, command_id, metadata)
        VALUES (:id, :requestId, :itemId, :reviewId, :photoIdSnapshot, :actorType, :actorId, :action,
            :fromState, :toState, :reason, :occurredAt, :commandId, CAST(:metadata AS jsonb))
        ON CONFLICT ON CONSTRAINT uq_photo_removal_history_action DO NOTHING
    """, nativeQuery = true)
    fun append(id: UUID, requestId: UUID, itemId: UUID?, reviewId: UUID?, photoIdSnapshot: UUID?,
        actorType: String, actorId: UUID?, action: String, fromState: String?, toState: String?, reason: String?,
        occurredAt: Instant, commandId: UUID, metadata: String): Int
}
