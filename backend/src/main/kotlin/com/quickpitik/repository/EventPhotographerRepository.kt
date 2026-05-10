package com.quickpitik.repository

import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
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

    // Atomic upsert for the per-(event, photographer) upload counters. Replaces
    // the read-modify-write `findById → orElseGet { new } → save` pattern that
    // (a) lost concurrent increments on photo_count and (b) crashed with a PK
    // violation when two first-uploads from the same photographer raced for the
    // same event (H-3 + M-6). joined_at is left to the column DEFAULT now() on
    // insert and untouched on conflict — matches @Column(updatable = false) on
    // the entity. sales_count and revenue_kept_php also rely on column defaults.
    @Modifying
    @Query(
        value = """
        INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at)
        VALUES (:eventId, :photographerId, 1, :now, :now)
        ON CONFLICT (event_id, photographer_id) DO UPDATE SET
            photo_count     = event_photographer.photo_count + 1,
            first_upload_at = COALESCE(event_photographer.first_upload_at, EXCLUDED.first_upload_at),
            last_upload_at  = EXCLUDED.last_upload_at
        """,
        nativeQuery = true,
    )
    fun upsertOnUpload(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("now") now: OffsetDateTime,
    ): Int
}
