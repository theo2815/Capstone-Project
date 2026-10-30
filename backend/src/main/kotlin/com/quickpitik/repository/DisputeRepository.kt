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

    // Approved QR Ph refunds nobody finished: paid (or created) before windowCutoff and no link newer than linkCutoff.
    @Query(
        """
        SELECT d FROM Dispute d, Order o
        WHERE o.id = d.orderId
          AND d.refundStatus = 'manual_review'
          AND d.refundRequestedBy IS NOT NULL
          AND o.paymentMethodWire = 'qrph'
          AND COALESCE(o.paidAt, o.createdAt) < :windowCutoff
          AND (d.refundLinkSentAt IS NULL OR d.refundLinkSentAt < :linkCutoff)
        ORDER BY d.refundRequestedAt ASC
        """,
    )
    fun findDueForAutoCancel(
        @Param("windowCutoff") windowCutoff: OffsetDateTime,
        @Param("linkCutoff") linkCutoff: OffsetDateTime,
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

    @Query(
        """
        SELECT d FROM Dispute d
        WHERE (:statusWire IS NULL OR d.statusWire = :statusWire)
        ORDER BY d.openedAt DESC, d.id ASC
        """,
        countQuery = """
        SELECT COUNT(d) FROM Dispute d
        WHERE (:statusWire IS NULL OR d.statusWire = :statusWire)
        """,
    )
    fun pageForAdmin(
        @Param("statusWire") statusWire: String?,
        pageable: Pageable,
    ): Page<Dispute>
}
