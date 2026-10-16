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
    @Query(value = """SELECT max(CAST(metadata->>'noticeVersion' AS bigint)) FROM photo_removal_history
        WHERE review_id = :reviewId AND action = 'VERIFIED' AND metadata->>'kind' = 'owner_reverification'""", nativeQuery = true)
    fun findReviewNoticeVersion(reviewId: UUID): Long?

    @Query(value = """WITH pending AS (SELECT h.* FROM photo_removal_history h WHERE h.review_id = :reviewId
        AND h.action IN ('ADMIN_DECIDED','ERASURE_PROGRESS') AND h.metadata->>'decisionVersion' IS NOT NULL
        AND NOT EXISTS (SELECT 1 FROM photo_removal_history n WHERE n.review_id = h.review_id
            AND n.action IN ('NOTICE_STORED','OWNER_UNAVAILABLE')
            AND n.metadata->>'noticeKey' = 'removal:review:' || h.review_id || ':outcome:' || (h.metadata->>'decisionVersion'))
        ), first_version AS (SELECT metadata->>'decisionVersion' AS version FROM pending
            ORDER BY CAST(metadata->>'decisionVersion' AS bigint) LIMIT 1)
        SELECT p.* FROM pending p WHERE p.metadata->>'decisionVersion' = (SELECT version FROM first_version)
        ORDER BY p.occurred_at, p.id""", nativeQuery = true)
    fun findPendingOutcomes(reviewId: UUID, pageable: Pageable): List<PhotoRemovalHistory>

    @Query(value = """
        SELECT EXISTS (SELECT 1 FROM photo_removal_history WHERE photo_id_snapshot = :photoId
            AND action = 'NOTICE_STORED' AND metadata->>'kind' = 'guest_buyer_removal' AND metadata->>'orderId' = :orderId)
    """, nativeQuery = true)
    fun hasGuestBuyerNotice(photoId: UUID, orderId: String): Boolean

    @Query(value = """
        SELECT h.id AS "noticeId", o.id AS "orderId", h.photo_id_snapshot AS "photoId",
            h.occurred_at AS "occurredAt", o.recipient_email AS "recipientEmail",
            CASE WHEN EXISTS (SELECT 1 FROM photo_removal_history s WHERE s.action = 'SUPPORT_UPDATED'
                AND s.request_id = h.request_id AND s.metadata->>'noticeId' = h.id::text
                AND s.metadata->>'orderId' = o.id::text AND s.metadata->>'photoId' = h.photo_id_snapshot::text)
                THEN 'COMPLETED' ELSE 'PENDING' END AS status
        FROM photo_removal_history h JOIN orders o ON o.id::text = h.metadata->>'orderId'
        WHERE h.request_id = :requestId AND h.action = 'NOTICE_STORED' AND h.metadata->>'kind' = 'guest_buyer_removal'
        ORDER BY h.occurred_at, h.id
    """, nativeQuery = true)
    fun findGuestBuyerNotices(requestId: UUID): List<GuestBuyerRemovalNoticeRow>
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

interface GuestBuyerRemovalNoticeRow {
    val noticeId: UUID
    val orderId: UUID
    val photoId: UUID
    val occurredAt: Instant
    val status: String
    val recipientEmail: String
}
