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

    // Withdrawable ("cleared") kept amount: undisputed sales that have aged
    // past the clearing cutoff, PLUS all refund rows (refunds bite immediately
    // — we never wait to claw back money that's been returned). Drives the
    // payout available balance. COALESCE keeps it non-null.
    @Query(
        """
        SELECT COALESCE(SUM(t.amountKeptPhp), 0) FROM Transaction t
        WHERE t.photographerId = :photographerId
          AND (
            t.isRefund = true OR (
              t.isRefund = false
              AND t.paidAt <= :cutoff
              AND NOT EXISTS (
                SELECT d.id FROM Dispute d
                WHERE d.orderId = t.orderId
                  AND d.photoId = t.photoId
                  AND d.statusWire IN ('open', 'escalated')
              )
            )
          )
        """,
    )
    fun sumCleared(
        @Param("photographerId") photographerId: UUID,
        @Param("cutoff") cutoff: OffsetDateTime,
    ): BigDecimal

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
    // back to kept before subtracting from list gross, or the platform fee
    // over-reports by the discount.
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

    // GMV at the list price buyers were charged from (M-16), not kept / keep
    // rate — the rate is live config and would restate history. A refund row
    // takes back the share of the list price it reversed: its (kept + discount)
    // over its sale's. A refund row with no sale row (refund_of NULL) adds
    // nothing here, as its sale never added either.
    // ponytail: such a row still lowers the kept sum, so platform revenue
    // over-reports by its reversal; exclude it from the kept/discount sums too
    // if `is_refund AND refund_of IS NULL` rows ever exist in prod.
    @Query(
        value = """
        SELECT ROUND(COALESCE(SUM(oi.price_php_at_purchase * CASE
                   WHEN t.is_refund = false THEN 1
                   ELSE (t.amount_kept_php + t.discount_php)
                        / NULLIF(s.amount_kept_php + s.discount_php, 0)
               END), 0), 2)
        FROM transactions t
        JOIN order_items oi ON oi.order_id = t.order_id AND oi.photo_id = t.photo_id
        LEFT JOIN transactions s ON s.id = t.refund_of
        WHERE t.paid_at >= :from AND t.paid_at < :to
        """,
        nativeQuery = true,
    )
    fun sumListGrossInWindow(
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
