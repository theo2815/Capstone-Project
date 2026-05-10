package com.quickpitik.repository

import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface EventPhotographerRepository : JpaRepository<EventPhotographer, EventPhotographerId> {

    @Query(
        """
        SELECT ep FROM EventPhotographer ep
        WHERE ep.id.photographerId = :photographerId
          AND (:withUploadsOnly = false OR ep.photoCount > 0)
        ORDER BY ep.lastUploadAt DESC NULLS LAST, ep.joinedAt DESC, ep.id.eventId ASC
        """,
        countQuery = """
        SELECT COUNT(ep) FROM EventPhotographer ep
        WHERE ep.id.photographerId = :photographerId
          AND (:withUploadsOnly = false OR ep.photoCount > 0)
        """,
    )
    fun searchForPhotographer(
        @Param("photographerId") photographerId: UUID,
        @Param("withUploadsOnly") withUploadsOnly: Boolean,
        pageable: Pageable,
    ): Page<EventPhotographer>

    fun findAllByIdPhotographerId(photographerId: UUID): List<EventPhotographer>

    // Per-event earnings list — filters out events with no revenue so the FE
    // tile doesn't show ₱0 rows (matches the mock filter
    // `PHOTOGRAPHER_EVENTS.filter(e => e.revenueKept > 0)`). Sorted by revenue
    // DESC so highest-grossing events surface first.
    @Query(
        """
        SELECT ep FROM EventPhotographer ep
        WHERE ep.id.photographerId = :photographerId
          AND ep.revenueKeptPhp > 0
        ORDER BY ep.revenueKeptPhp DESC, ep.id.eventId ASC
        """,
        countQuery = """
        SELECT COUNT(ep) FROM EventPhotographer ep
        WHERE ep.id.photographerId = :photographerId
          AND ep.revenueKeptPhp > 0
        """,
    )
    fun pageEarningsForPhotographer(
        @Param("photographerId") photographerId: UUID,
        pageable: Pageable,
    ): Page<EventPhotographer>

    // Admin sales-by-event — sums photographer-keep across all photographers
    // covering an event so the admin tile shows event-level (not per-row)
    // numbers. impliedGmv is reconstructed from amount_kept / keep_rate
    // service-side.
    @Query(
        value = """
        SELECT t.event_id AS event_id,
               COALESCE(SUM(CASE WHEN t.is_refund = false THEN t.amount_kept_php ELSE 0 END), 0)
                   AS implied_cut,
               COALESCE(-SUM(CASE WHEN t.is_refund = true  THEN t.amount_kept_php ELSE 0 END), 0)
                   AS refunds,
               COUNT(*) FILTER (WHERE t.is_refund = false) AS sales_count
        FROM transactions t
        GROUP BY t.event_id
        """,
        nativeQuery = true,
    )
    fun salesAggregatesByEvent(): List<Array<Any>>
}
