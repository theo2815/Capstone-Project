package com.quickpitik.repository

import com.quickpitik.entity.PayoutCycle
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.math.BigDecimal
import java.util.UUID

interface PayoutCycleRepository : JpaRepository<PayoutCycle, String>, PayoutCycleInsert {

    @Query("SELECT c.photographerId FROM PayoutCycle c WHERE c.id = :id")
    fun findPhotographerIdById(@Param("id") id: String): UUID?

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT c FROM PayoutCycle c WHERE c.id = :id")
    fun lockById(@Param("id") id: String): PayoutCycle?

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT c FROM PayoutCycle c WHERE c.id = :id AND c.photographerId = :photographerId")
    fun lockByIdAndPhotographerId(@Param("id") id: String, @Param("photographerId") photographerId: UUID): PayoutCycle?

    fun existsByAccountSnapshotQrKey(key: String): Boolean

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
          AND c.statusWire IN ('scheduled', 'pending', 'approved', 'processing')
        """,
    )
    fun sumOpenAmount(@Param("photographerId") photographerId: UUID): BigDecimal

    // The next non-paid cycle, ordered by settled_at ASC NULLS LAST then week_of —
    // FE renders payoutScheduledFor from this row.
    @Query(
        """
        SELECT c FROM PayoutCycle c
        WHERE c.photographerId = :photographerId
          AND c.statusWire IN ('scheduled', 'pending', 'approved', 'processing')
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

    // Database-wide payout number for new cycle ids (V87). Unique across
    // photographers by construction, unlike a per-photographer count.
    @Query(value = "SELECT nextval('payout_cycle_number_seq')", nativeQuery = true)
    fun nextCycleNumber(): Long

    // Photographer-initiated payout request flow:
    //   - findFirstByPhotographerIdAndStatusWireIn finds an open cycle so we
    //     can enforce "one open request at a time."
    //   - sumPaidByPhotographer subtracts from lifetime kept to compute the
    //     photographer's unpaid balance for the ₱500 threshold check.
    fun countByPhotographerId(photographerId: UUID): Long

    fun findFirstByPhotographerIdAndStatusWireIn(
        photographerId: UUID,
        statusWires: Collection<String>,
    ): PayoutCycle?

    // The PayMongo transfer webhook resolves a cycle by the id it stored at
    // disbursement time (Phase 11).
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    fun findByProviderTransferId(providerTransferId: String): PayoutCycle?

    @Query(
        """
        SELECT COALESCE(SUM(c.amountPhp), 0) FROM PayoutCycle c
        WHERE c.photographerId = :photographerId
          AND c.statusWire = 'paid'
        """,
    )
    fun sumPaidByPhotographer(@Param("photographerId") photographerId: UUID): BigDecimal

    // Amount tied up in OPEN cycles — reserved against the available balance so a
    // photographer can't request funds already committed to a pending/held cycle.
    // Includes 'held' (still committed), unlike sumOpenAmount which is a display
    // figure that hides held. Phase 2's terminal states (cancelled/rejected/
    // failed/returned) must stay OUT of this list — they release the reservation.
    @Query(
        """
        SELECT COALESCE(SUM(c.amountPhp), 0) FROM PayoutCycle c
        WHERE c.photographerId = :photographerId
          AND c.statusWire IN ('scheduled', 'pending', 'approved', 'processing', 'held')
        """,
    )
    fun sumReservedByPhotographer(@Param("photographerId") photographerId: UUID): BigDecimal

    // Admin payout queue list. The service maps the FE's statuses to DB wires
    // ("pending_review" → "pending"; the DB enum is shared with the photographer
    // surface) and passes every wire for "no filter" (AX-10: a tab asks for its group).
    @Query(
        """
        SELECT c FROM PayoutCycle c
        WHERE c.statusWire IN :statusWires AND $ADMIN_PAYOUT_SEARCH
        ORDER BY c.weekOf DESC, c.id ASC
        """,
        countQuery = "SELECT COUNT(c) FROM PayoutCycle c WHERE c.statusWire IN :statusWires AND $ADMIN_PAYOUT_SEARCH",
    )
    fun pageForAdmin(
        @Param("statusWires") statusWires: Collection<String>,
        @Param("search") search: String,
        pageable: Pageable,
    ): Page<PayoutCycle>

    // The admin payouts tab counts (AX-10): the list's own search, one row per status present.
    @Query(
        "SELECT new com.quickpitik.repository.StatusCountRow(c.statusWire, COUNT(c)) FROM PayoutCycle c " +
            "WHERE $ADMIN_PAYOUT_SEARCH GROUP BY c.statusWire",
    )
    fun countByStatusForAdmin(@Param("search") search: String): List<StatusCountRow>
}

// The admin payout search (FT-28): the payout id, or its photographer's name, email (FU-2), brand
// name or handle, the handle typed with or without "@". Columns as the admin user search uses them.
private const val ADMIN_PAYOUT_SEARCH = """(:search = ''
    OR LOWER(c.id) LIKE CONCAT('%', :search, '%') ESCAPE '\'
    OR EXISTS (SELECT 1 FROM User u WHERE u.id = c.photographerId
        AND (LOWER(u.name) LIKE CONCAT('%', :search, '%') ESCAPE '\'
            OR LOWER(u.email) LIKE CONCAT('%', :search, '%') ESCAPE '\'))
    OR EXISTS (SELECT 1 FROM PhotographerSettings s WHERE s.userId = c.photographerId
        AND (LOWER(s.brandName) LIKE CONCAT('%', :search, '%') ESCAPE '\'
            OR CONCAT('@', LOWER(s.handle)) LIKE CONCAT('%', :search, '%') ESCAPE '\')))"""

// One status and how many rows have it; the admin payouts and flags counts routes (AX-10).
data class StatusCountRow(val status: String, val total: Long)
