package com.quickpitik.repository

import com.quickpitik.entity.PayoutCycle
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.math.BigDecimal
import java.util.UUID

interface PayoutCycleRepository : JpaRepository<PayoutCycle, String> {

    fun findByIdAndPhotographerId(id: String, photographerId: UUID): PayoutCycle?

    @Query(
        """
        SELECT c FROM PayoutCycle c
        WHERE c.photographerId = :photographerId
        ORDER BY c.weekOf DESC, c.id ASC
        """,
        countQuery = """
        SELECT COUNT(c) FROM PayoutCycle c
        WHERE c.photographerId = :photographerId
        """,
    )
    fun pageForPhotographer(
        @Param("photographerId") photographerId: UUID,
        pageable: Pageable,
    ): Page<PayoutCycle>

    // Sum of cycles in a non-paid state — what the FE labels payoutPending.
    // Held cycles are intentionally excluded so the chip only shows what's
    // actually moving through the queue (held = stuck, surfaces elsewhere).
    @Query(
        """
        SELECT COALESCE(SUM(c.amountPhp), 0) FROM PayoutCycle c
        WHERE c.photographerId = :photographerId
          AND c.statusWire IN ('scheduled', 'pending')
        """,
    )
    fun sumOpenAmount(@Param("photographerId") photographerId: UUID): BigDecimal

    // The next non-paid cycle, ordered by settled_at ASC NULLS LAST then week_of —
    // FE renders payoutScheduledFor from this row.
    @Query(
        """
        SELECT c FROM PayoutCycle c
        WHERE c.photographerId = :photographerId
          AND c.statusWire IN ('scheduled', 'pending')
        ORDER BY (CASE WHEN c.settledAt IS NULL THEN 1 ELSE 0 END) ASC,
                 c.settledAt ASC,
                 c.weekOf ASC
        """,
    )
    fun findNextUpcoming(
        @Param("photographerId") photographerId: UUID,
        pageable: Pageable,
    ): List<PayoutCycle>

    fun countByStatusWire(statusWire: String): Long

    // Photographer-initiated payout request flow:
    //   - countByPhotographerId drives the next sequence number when building
    //     a new cycle id (PAY-{HANDLE}-{seq:003}).
    //   - findFirstByPhotographerIdAndStatusWireIn finds an open cycle so we
    //     can enforce "one open request at a time."
    //   - sumPaidByPhotographer subtracts from lifetime kept to compute the
    //     photographer's unpaid balance for the ₱500 threshold check.
    fun countByPhotographerId(photographerId: UUID): Long

    fun findFirstByPhotographerIdAndStatusWireIn(
        photographerId: UUID,
        statusWires: Collection<String>,
    ): PayoutCycle?

    @Query(
        """
        SELECT COALESCE(SUM(c.amountPhp), 0) FROM PayoutCycle c
        WHERE c.photographerId = :photographerId
          AND c.statusWire = 'paid'
        """,
    )
    fun sumPaidByPhotographer(@Param("photographerId") photographerId: UUID): BigDecimal

    // Admin payout queue list. The FE filter is one of pending_review |
    // approved | held | paid; we map "pending_review" → DB "pending" since
    // the DB enum is shared with the photographer-facing surface (which
    // uses "pending" / "scheduled" — admin only acts on "pending" rows).
    @Query(
        """
        SELECT c FROM PayoutCycle c
        WHERE (:statusWire IS NULL OR c.statusWire = :statusWire)
          AND (:search = '' OR LOWER(c.id) LIKE CONCAT('%', :search, '%'))
        ORDER BY c.weekOf DESC, c.id ASC
        """,
        countQuery = """
        SELECT COUNT(c) FROM PayoutCycle c
        WHERE (:statusWire IS NULL OR c.statusWire = :statusWire)
          AND (:search = '' OR LOWER(c.id) LIKE CONCAT('%', :search, '%'))
        """,
    )
    fun pageForAdmin(
        @Param("statusWire") statusWire: String?,
        @Param("search") search: String,
        pageable: Pageable,
    ): Page<PayoutCycle>
}
