package com.quickpitik.repository

import com.quickpitik.entity.Dispute
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
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
