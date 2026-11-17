package com.quickpitik.repository

import com.quickpitik.entity.Dispute
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

interface DisputeRepository : JpaRepository<Dispute, UUID> {
    fun findByOrderId(orderId: UUID): List<Dispute>

    // Batch fetch for OrderService.hydrateList — one round-trip for the whole
    // page of orders instead of N. Service layer groups by orderId.
    fun findByOrderIdIn(orderIds: Collection<UUID>): List<Dispute>

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT d FROM Dispute d WHERE d.id = :id")
    fun findByIdForUpdate(@Param("id") id: UUID): Dispute?

    fun findByRefundStatusInOrderByRefundRequestedAtAsc(
        refundStatuses: Collection<String>,
        pageable: Pageable,
    ): List<Dispute>

    fun findByProviderRefundId(providerRefundId: String): Dispute?

    // Approved QR Ph refunds nobody sent a link for, paid (or created) before windowCutoff.
    // A sent link may have been claimed, so those rows are never auto-cancelled.
    @Query(
        """
        SELECT d FROM Dispute d, Order o
        WHERE o.id = d.orderId
          AND d.refundStatus = 'manual_review'
          AND o.paymentMethodWire = 'qrph'
          AND COALESCE(o.paidAt, o.createdAt) < :windowCutoff
          AND d.refundLinkSentAt IS NULL
        ORDER BY d.refundRequestedAt ASC
        """,
    )
    fun findDueForAutoCancel(
        @Param("windowCutoff") windowCutoff: OffsetDateTime,
        pageable: Pageable,
    ): List<Dispute>

    @Query(
        """
        SELECT d FROM Dispute d
        WHERE d.orderId = :orderId
          AND d.photoId = :photoId
          AND d.statusWire IN ('open', 'escalated')
        """,
    )
    fun findOpenForOrderPhoto(
        @Param("orderId") orderId: UUID,
        @Param("photoId") photoId: UUID,
    ): Dispute?

    fun countByStatusWire(statusWire: String): Long

    @Query(
        """
        SELECT COALESCE(SUM(d.refundAmountPhp), 0) FROM Dispute d
        WHERE d.refundStatus = 'succeeded'
          AND d.refundedAt >= :from AND d.refundedAt < :to
        """,
    )
    fun sumSucceededRefundsInWindow(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): BigDecimal

    @Query(
        value = """
        SELECT o.event_id, COALESCE(SUM(d.refund_amount_php), 0)
        FROM disputes d
        JOIN orders o ON o.id = d.order_id
        WHERE d.refund_status = 'succeeded'
        GROUP BY o.event_id
        """,
        nativeQuery = true,
    )
    fun sumSucceededRefundsByEvent(): List<Array<Any>>

    // Admin queue list and search. `q` is '' for no search, never null (a null bind inside
    // LOWER(CONCAT(...)) is typed bytea by Postgres). Names, e-mails, the handle and the event
    // name match as case-insensitive "contains"; ids and PayMongo ids match exactly.
    // EXISTS keeps one row per dispute however many payments or joins match.
    @Query(
        """
        SELECT d FROM Dispute d
        WHERE d.statusWire IN :statusWires AND ($ADMIN_SEARCH)
        ORDER BY d.openedAt DESC, d.id ASC
        """,
        countQuery = """
        SELECT COUNT(d) FROM Dispute d
        WHERE d.statusWire IN :statusWires AND ($ADMIN_SEARCH)
        """,
    )
    fun pageForAdmin(
        @Param("statusWires") statusWires: Collection<String>,
        @Param("q") q: String,
        pageable: Pageable,
    ): Page<Dispute>

    // The admin's "this runner's other requests": rows per runner and status (idx_disputes_runner).
    // The alias is `n`, clear of HQL's `count` keyword.
    @Query(
        """
        SELECT d.runnerId AS runnerId, d.statusWire AS statusWire, COUNT(d) AS n FROM Dispute d
        WHERE d.runnerId IN :runnerIds
        GROUP BY d.runnerId, d.statusWire
        """,
    )
    fun countByRunnerAndStatus(@Param("runnerIds") runnerIds: Collection<UUID>): List<RunnerDisputeCount>
}

interface RunnerDisputeCount {
    val runnerId: UUID
    val statusWire: String
    val n: Long
}

private const val CONTAINS_Q ="LIKE LOWER(CONCAT('%', :q, '%'))"

// The `q` predicates of pageForAdmin, shared by its row and count queries.
private const val ADMIN_SEARCH = """
    :q = ''
    OR CAST(d.id AS String) = LOWER(:q)
    OR CAST(d.orderId AS String) = LOWER(:q)
    OR EXISTS (SELECT 1 FROM User u WHERE u.id = d.runnerId
        AND (LOWER(u.name) $CONTAINS_Q OR LOWER(u.email) $CONTAINS_Q))
    OR EXISTS (SELECT 1 FROM Order o WHERE o.id = d.orderId
        AND d.runnerId IS NULL AND LOWER(o.recipientEmail) $CONTAINS_Q)
    OR EXISTS (SELECT 1 FROM Photo p, PhotographerSettings ps WHERE p.id = d.photoId
        AND ps.userId = p.photographerId AND LOWER(ps.handle) $CONTAINS_Q)
    OR EXISTS (SELECT 1 FROM Order o, Event e WHERE o.id = d.orderId
        AND e.id = o.eventId AND LOWER(e.name) $CONTAINS_Q)
    OR EXISTS (SELECT 1 FROM Payment pm WHERE pm.orderId = d.orderId
        AND (pm.providerPaymentId = :q OR pm.providerRef = :q))
"""
