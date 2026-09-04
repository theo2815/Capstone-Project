package com.quickpitik.repository

import com.quickpitik.entity.Transaction
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

interface TransactionRepository : JpaRepository<Transaction, UUID> {

    fun findByOrderIdAndPhotoIdAndIsRefund(
        orderId: UUID,
        photoId: UUID,
        isRefund: Boolean,
    ): Transaction?

    @Query(
        """
        SELECT t FROM Transaction t
        WHERE t.photographerId = :photographerId
        ORDER BY t.paidAt DESC, t.id ASC
        """,
        countQuery = """
        SELECT COUNT(t) FROM Transaction t
        WHERE t.photographerId = :photographerId
        """,
    )
    fun pageForPhotographer(
        @Param("photographerId") photographerId: UUID,
        pageable: Pageable,
    ): Page<Transaction>

    // SUM over a window. COALESCE on the JPQL side keeps the return non-null
    // when the photographer has no rows in the window so callers don't have to
    // unwrap an Optional<BigDecimal>.
    @Query(
        """
        SELECT COALESCE(SUM(t.amountKeptPhp), 0) FROM Transaction t
        WHERE t.photographerId = :photographerId
        """,
    )
    fun sumLifetime(@Param("photographerId") photographerId: UUID): BigDecimal

    @Query(
        """
        SELECT COALESCE(SUM(t.amountKeptPhp), 0) FROM Transaction t
        WHERE t.photographerId = :photographerId
          AND t.paidAt >= :from AND t.paidAt < :to
        """,
    )
    fun sumWindow(
        @Param("photographerId") photographerId: UUID,
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): BigDecimal

    // Sales count (non-refund rows only) per window — drives thisWeekSold /
    // thisMonthSold on the dashboard chip.
    @Query(
        """
        SELECT COUNT(t) FROM Transaction t
        WHERE t.photographerId = :photographerId
          AND t.paidAt >= :from AND t.paidAt < :to
          AND t.isRefund = false
        """,
    )
    fun countSalesWindow(
        @Param("photographerId") photographerId: UUID,
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): Long

    // Returns rows of (paid_at, amount_kept_php) within a window for in-memory
    // weekly bucketing — the 12-week series is small enough that fetching the
    // raw rows beats writing portable date-trunc SQL across H2 / Postgres.
    @Query(
        """
        SELECT t.paidAt, t.amountKeptPhp FROM Transaction t
        WHERE t.photographerId = :photographerId
          AND t.paidAt >= :from AND t.paidAt < :to
        ORDER BY t.paidAt ASC
        """,
    )
    fun pickAmountsForBucketing(
        @Param("photographerId") photographerId: UUID,
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): List<Array<Any>>

    // Admin sales aggregates — across all photographers in the window.
    @Query(
        """
        SELECT COALESCE(SUM(t.amountKeptPhp), 0) FROM Transaction t
        WHERE t.paidAt >= :from AND t.paidAt < :to
        """,
    )
    fun sumPhotographerKeepInWindow(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): BigDecimal

    // Coupon discounts in the window (V45), net of refunds like kept. Added
    // back to kept before dividing by the keep rate, or the reconstructed
    // gross — and with it the platform fee — under-reports by the discount.
    @Query(
        """
        SELECT COALESCE(SUM(t.discountPhp), 0) FROM Transaction t
        WHERE t.paidAt >= :from AND t.paidAt < :to
        """,
    )
    fun sumDiscountsInWindow(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): BigDecimal

    // Per-photographer net kept amount + non-refund sale count for a single
    // week window. Drives the admin "Generate cycles" action — one row per
    // photographer who had at least ₱0.01 net activity in the window. Refund
    // rows reduce amountKeptPhp (they're inserted as negative); itemCount
    // counts non-refund rows only. Returns Array<Any> of (photographerId,
    // netAmount, saleCount) — Kotlin caller widens the types.
    @Query(
        """
        SELECT t.photographerId,
               COALESCE(SUM(t.amountKeptPhp), 0),
               SUM(CASE WHEN t.isRefund = false THEN 1 ELSE 0 END)
        FROM Transaction t
        WHERE t.paidAt >= :from AND t.paidAt < :to
        GROUP BY t.photographerId
        HAVING COALESCE(SUM(t.amountKeptPhp), 0) > 0
        """,
    )
    fun aggregateByPhotographerInWindow(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): List<Array<Any>>

    // Refund total in a window — only the negative-amount refund rows. The
    // sum is positive (we negate the negatives) so the FE can render it as
    // a peso amount without sign juggling.
    @Query(
        """
        SELECT COALESCE(-SUM(t.amountKeptPhp), 0) FROM Transaction t
        WHERE t.isRefund = true
          AND t.paidAt >= :from AND t.paidAt < :to
        """,
    )
    fun sumRefundsInWindow(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): BigDecimal

    @Query(
        """
        SELECT COUNT(t) FROM Transaction t
        WHERE t.isRefund = false
          AND t.paidAt >= :from AND t.paidAt < :to
        """,
    )
    fun countSalesInWindow(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): Long

    // Daily counts of decisions in a window — used for the kpi-trend chart.
    @Query(
        value = """
        SELECT paid_at::date AS day,
               COUNT(*) FILTER (WHERE is_refund = false) AS sales,
               COUNT(*) FILTER (WHERE is_refund = true) AS refunds
        FROM transactions
        WHERE paid_at >= :from AND paid_at < :to
        GROUP BY day
        ORDER BY day
        """,
        nativeQuery = true,
    )
    fun salesAndRefundCountsByDayBetween(
        @Param("from") from: OffsetDateTime,
        @Param("to") to: OffsetDateTime,
    ): List<Array<Any>>
}
