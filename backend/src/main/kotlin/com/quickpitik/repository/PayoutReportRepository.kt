package com.quickpitik.repository

import com.quickpitik.entity.PayoutReport
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface PayoutReportRepository : JpaRepository<PayoutReport, UUID> {

    // The OPEN-row gate behind REPORT_ALREADY_OPEN — service checks before the
    // insert so we can return a clean 409 without relying on the DB
    // DataIntegrityViolation surfacing through the partial unique index.
    @Query(
        """
        SELECT r FROM PayoutReport r
        WHERE r.payoutId = :payoutId
          AND r.photographerId = :photographerId
          AND r.statusWire = 'open'
        """,
    )
    fun findOpenForCycle(
        @Param("payoutId") payoutId: String,
        @Param("photographerId") photographerId: UUID,
    ): PayoutReport?

    @Query(
        """
        SELECT r FROM PayoutReport r
        WHERE r.photographerId = :photographerId
          AND (:cycleId IS NULL OR r.payoutId = :cycleId)
          AND (:statusWire IS NULL OR r.statusWire = :statusWire)
        ORDER BY r.openedAt DESC, r.id ASC
        """,
    )
    fun findForPhotographer(
        @Param("photographerId") photographerId: UUID,
        @Param("cycleId") cycleId: String?,
        @Param("statusWire") statusWire: String?,
    ): List<PayoutReport>

    // Admin queue list — across all photographers, optional status filter.
    @Query(
        """
        SELECT r FROM PayoutReport r
        WHERE (:statusWire IS NULL OR r.statusWire = :statusWire)
        ORDER BY r.openedAt DESC, r.id ASC
        """,
        countQuery = """
        SELECT COUNT(r) FROM PayoutReport r
        WHERE (:statusWire IS NULL OR r.statusWire = :statusWire)
        """,
    )
    fun pageForAdmin(
        @Param("statusWire") statusWire: String?,
        pageable: Pageable,
    ): Page<PayoutReport>
}
