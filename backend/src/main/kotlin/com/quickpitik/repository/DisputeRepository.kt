package com.quickpitik.repository

import com.quickpitik.entity.Dispute
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
}
