package com.quickpitik.repository

import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

interface EventPhotographerRepository : JpaRepository<EventPhotographer, EventPhotographerId> {

    @Query("""
        SELECT ep.id.photographerId FROM EventPhotographer ep
        JOIN Event e ON e.id = ep.id.eventId
        JOIN User u ON u.id = ep.id.photographerId
        JOIN PhotographerSettings s ON s.userId = ep.id.photographerId
        WHERE ep.id.eventId = :eventId AND ep.id.photographerId IN :photographerIds
          AND ep.metadataPublic = true AND e.deletedAt IS NULL
          AND e.status <> com.quickpitik.entity.EventStatus.DRAFT
          AND u.suspendedAt IS NULL
          AND s.verificationStatus = com.quickpitik.entity.VerificationStatus.APPROVED
    """)
    fun findPublicMetadataPhotographerIds(eventId: UUID, photographerIds: Collection<UUID>): List<UUID>

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

    // Pageable-bounded: the sole caller is the UNAUTHENTICATED public profile,
    // which caps at its own MAX_PUBLIC_EVENTS (newest coverage first).
    fun findAllByIdPhotographerId(photographerId: UUID, pageable: Pageable): List<EventPhotographer>

    // Positive-revenue legacy rows stay visible; settled own-photo purchases
    // also retain giveaways and fully refunded sales. EXISTS avoids multiplying
    // revenue through item joins.
    @Query(
        """
        SELECT ep FROM EventPhotographer ep
        WHERE ep.id.photographerId = :photographerId
          AND (ep.revenueKeptPhp > 0 OR EXISTS (
              SELECT oi.id.photoId FROM OrderItem oi, Photo p, Order o
              WHERE oi.id.photoId = p.id
                AND oi.id.orderId = o.id
                AND p.photographerId = :photographerId
                AND p.eventId = ep.id.eventId
                AND o.eventId = ep.id.eventId
                AND (o.paidAt IS NOT NULL OR o.status IN (
                    com.quickpitik.entity.OrderStatus.PAID,
                    com.quickpitik.entity.OrderStatus.FULFILLED,
                    com.quickpitik.entity.OrderStatus.REFUNDED
                ))
          ))
        ORDER BY ep.revenueKeptPhp DESC, ep.id.eventId ASC
        """,
        countQuery = """
        SELECT COUNT(ep) FROM EventPhotographer ep
        WHERE ep.id.photographerId = :photographerId
          AND (ep.revenueKeptPhp > 0 OR EXISTS (
              SELECT oi.id.photoId FROM OrderItem oi, Photo p, Order o
              WHERE oi.id.photoId = p.id
                AND oi.id.orderId = o.id
                AND p.photographerId = :photographerId
                AND p.eventId = ep.id.eventId
                AND o.eventId = ep.id.eventId
                AND (o.paidAt IS NOT NULL OR o.status IN (
                    com.quickpitik.entity.OrderStatus.PAID,
                    com.quickpitik.entity.OrderStatus.FULFILLED,
                    com.quickpitik.entity.OrderStatus.REFUNDED
                ))
          ))
        """,
    )
    fun pageEarningsForPhotographer(
        @Param("photographerId") photographerId: UUID,
        pageable: Pageable,
    ): Page<EventPhotographer>

    // Admin sales-by-event — sums photographer-keep across all photographers
    // covering an event so the admin tile shows event-level (not per-row)
    // numbers. list_gross is what the sold items listed at (order_items), so
    // impliedGmv never depends on today's keep rate (M-16).
    @Query(
        value = """
        SELECT t.event_id AS event_id,
               COALESCE(SUM(CASE WHEN t.is_refund = false THEN t.amount_kept_php ELSE 0 END), 0)
                   AS photographer_keep,
               COUNT(*) FILTER (WHERE t.is_refund = false) AS sales_count,
               COALESCE(SUM(CASE WHEN t.is_refund = false THEN t.discount_php ELSE 0 END), 0)
                   AS discounts,
               COALESCE(SUM(CASE WHEN t.is_refund = false THEN oi.price_php_at_purchase ELSE 0 END), 0)
                   AS list_gross
        FROM transactions t
        LEFT JOIN order_items oi ON oi.order_id = t.order_id AND oi.photo_id = t.photo_id
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
        INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at, metadata_public)
        SELECT :eventId, :photographerId, 1, :now, :now,
               CASE WHEN (SELECT created_by FROM events WHERE id = :eventId) = :photographerId
                    THEN false
                    ELSE COALESCE((SELECT metadata_public_by_default FROM photographer_settings WHERE user_id = :photographerId), false)
               END
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

    @Modifying
    @Query(
        "UPDATE EventPhotographer ep SET ep.metadataPublic = :value " +
            "WHERE ep.id.eventId = :eventId AND ep.id.photographerId = :photographerId",
    )
    fun updateMetadataVisibility(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("value") value: Boolean,
    ): Int

    @Modifying
    @Query(
        value = """
        INSERT INTO event_photographer (event_id, photographer_id, metadata_public)
        VALUES (:eventId, :photographerId, :value)
        ON CONFLICT (event_id, photographer_id) DO UPDATE SET
            metadata_public = EXCLUDED.metadata_public
        """,
        nativeQuery = true,
    )
    fun upsertMetadataVisibility(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("value") value: Boolean,
    ): Int

    @Modifying
    @Query("UPDATE EventPhotographer ep SET ep.photoCount = GREATEST(0, ep.photoCount + :delta) " +
        "WHERE ep.id.eventId = :eventId AND ep.id.photographerId = :photographerId")
    fun adjustPhotoCount(@Param("eventId") eventId: UUID, @Param("photographerId") photographerId: UUID,
        @Param("delta") delta: Int): Int

    @Modifying
    @Query(
        value = """
        INSERT INTO event_photographer (event_id, photographer_id, sales_count, revenue_kept_php)
        VALUES (:eventId, :photographerId, GREATEST(0, :saleDelta), :keptDelta)
        ON CONFLICT (event_id, photographer_id) DO UPDATE SET
            sales_count = GREATEST(0, event_photographer.sales_count + :saleDelta),
            revenue_kept_php = event_photographer.revenue_kept_php + :keptDelta
        """,
        nativeQuery = true,
    )
    fun adjustSalesCounters(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("saleDelta") saleDelta: Int,
        @Param("keptDelta") keptDelta: BigDecimal,
    ): Int
}
