package com.quickpitik.repository

import jakarta.persistence.LockModeType
import org.springframework.data.jpa.repository.Lock
import com.quickpitik.entity.Flag
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface FlagRepository : JpaRepository<Flag, UUID> {

    @Query(
        """
        SELECT f FROM Flag f
        WHERE (:statusWire IS NULL OR f.statusWire = :statusWire)
          AND (:query = '' OR LOWER(f.reason) LIKE LOWER(CONCAT('%', :query, '%')) OR LOWER(f.note) LIKE LOWER(CONCAT('%', :query, '%')))
        ORDER BY f.createdAt DESC, f.id ASC
        """,
        countQuery = """
        SELECT COUNT(f) FROM Flag f
        WHERE (:statusWire IS NULL OR f.statusWire = :statusWire)
          AND (:query = '' OR LOWER(f.reason) LIKE LOWER(CONCAT('%', :query, '%')) OR LOWER(f.note) LIKE LOWER(CONCAT('%', :query, '%')))
        """,
    )
    fun pageForAdmin(
        @Param("statusWire") statusWire: String?,
        @Param("query") query: String?,
        pageable: Pageable,
    ): Page<Flag>

    @Query("SELECT f.targetId FROM Flag f WHERE f.id = :id AND f.targetKindWire = 'photo'")
    fun findPhotoId(@Param("id") id: UUID): UUID?

    // PE-2.2: the photographer behind a relevance flag, read before any lock (the twin of findPhotoId).
    @Query("SELECT f.targetId FROM Flag f WHERE f.id = :id AND f.targetKindWire = 'user' AND f.reason = :reason")
    fun findRelevanceUserTarget(@Param("id") id: UUID, @Param("reason") reason: String): UUID?

    // PE-2.2: the read behind PayoutRequestService.relevanceHoldOpen, the one home of the hold rule.
    @Query(
        """
        SELECT COUNT(f) > 0 FROM Flag f
        WHERE f.targetKindWire = 'user' AND f.targetId = :userId AND f.reason = :reason
          AND f.statusWire IN ('open', 'escalated') AND f.payoutHoldAt IS NOT NULL
        """,
    )
    fun existsOpenPayoutHold(@Param("userId") userId: UUID, @Param("reason") reason: String): Boolean

    // PE-2.2: open relevance flags for a page of photographers, held or not (the admin payout list).
    @Query(
        """
        SELECT f FROM Flag f
        WHERE f.targetKindWire = 'user' AND f.targetId IN :userIds AND f.reason = :reason
          AND f.statusWire IN ('open', 'escalated')
        """,
    )
    fun findOpenByUserTargetsAndReason(
        @Param("userIds") userIds: Collection<UUID>,
        @Param("reason") reason: String,
    ): List<Flag>

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT f FROM Flag f WHERE f.id = :id")
    fun findByIdForUpdate(@Param("id") id: UUID): Flag?

    fun countByStatusWire(statusWire: String): Long

    // SystemFlagService's duplicate lookup. Derived (no @Query): no null bind can hit the PG 18 typing trap.
    fun findByTargetKindWireAndTargetIdAndReasonAndStatusWireIn(
        targetKindWire: String,
        targetId: UUID,
        reason: String,
        statusWires: Collection<String>,
    ): List<Flag>

    // AA-R9: accounts with a live report on one target (the hide threshold).
    @Query(
        """
        SELECT COUNT(DISTINCT f.reporterId) FROM Flag f
        WHERE f.targetKindWire = :targetKind AND f.targetId = :targetId
          AND f.reason = 'user_report' AND f.statusWire IN ('open', 'escalated')
        """,
    )
    fun countDistinctReporters(@Param("targetKind") targetKind: String, @Param("targetId") targetId: UUID): Long

    fun existsByTargetKindWireAndTargetIdAndStatusWireAndIdNot(
        targetKindWire: String,
        targetId: UUID,
        statusWire: String,
        id: UUID,
    ): Boolean
}
