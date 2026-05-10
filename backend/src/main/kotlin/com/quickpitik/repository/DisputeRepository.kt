package com.quickpitik.repository

import com.quickpitik.entity.Dispute
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface DisputeRepository : JpaRepository<Dispute, UUID> {
    fun findByOrderId(orderId: UUID): List<Dispute>

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
