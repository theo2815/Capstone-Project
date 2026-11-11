package com.quickpitik.repository

import com.quickpitik.entity.BatchRelevance
import jakarta.persistence.LockModeType
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

// PE-1 photo-to-event relevance. batch_relevance and batch_relevance_photos are written only by
// service/relevance. The relevance reads of photos live here too, not in PhotoRepository: they only
// read, and PhotoRepository is edited by other lanes at the same time.
interface BatchRelevanceRepository : JpaRepository<BatchRelevance, UUID> {

    // Idempotent: a second call, or a second instance, inserts nothing. Makes no row for a suspended
    // (or anonymised) photographer or a deleted event, so a tick that listed the batch before the
    // anonymise script committed cannot bring the row back (R2-38).
    @Modifying
    @Query(value = """
        INSERT INTO batch_relevance (event_id, photographer_id)
        SELECT e.id, u.id FROM events e, users u
        WHERE e.id = :eventId AND e.deleted_at IS NULL
          AND u.id = :photographerId AND u.suspended_at IS NULL
        ON CONFLICT (event_id, photographer_id) DO NOTHING
    """, nativeQuery = true)
    fun ensureBatch(@Param("eventId") eventId: UUID, @Param("photographerId") photographerId: UUID): Int

    fun findByEventIdAndPhotographerId(eventId: UUID, photographerId: UUID): BatchRelevance?

    // The batches the sweep should work on (plan 5.4). Quiet for :quietCutoff, then one of: no row yet;
    // a different newest upload than the one scored; the event's date edited since scoring; or, before
    // the final pass, unsettled work or closed uploads once :rescoreCutoff has passed. Uploads close
    // (PhotoUploadService.gate) after date + 3 days for a shared race and one month for a
    // photographer's own event; the caller passes both cutoff dates from today in Manila.
    @Query(value = """
        SELECT ep.event_id AS "eventId", ep.photographer_id AS "photographerId",
               ep.last_upload_at AS "lastUploadAt", e.date AS "eventDate",
               (e.created_by IS NULL) AS "sharedRace"
        FROM event_photographer ep
        JOIN events e ON e.id = ep.event_id AND e.deleted_at IS NULL
        JOIN users u ON u.id = ep.photographer_id AND u.suspended_at IS NULL
        LEFT JOIN batch_relevance br ON br.event_id = ep.event_id AND br.photographer_id = ep.photographer_id
        WHERE ep.last_upload_at <= :quietCutoff
          AND (br.id IS NULL
               OR br.scored_upload_at IS DISTINCT FROM ep.last_upload_at
               OR (br.scored_at IS NOT NULL AND br.scored_event_date IS DISTINCT FROM e.date)
               OR (br.final_scored_at IS NULL AND br.scored_at <= :rescoreCutoff
                   AND (br.metadata_pending > 0 OR br.photos_processing > 0
                        OR (e.created_by IS NULL AND e.date <= :adminClosedOnOrBefore)
                        OR (e.created_by IS NOT NULL AND e.date <= :ownerClosedOnOrBefore))))
        ORDER BY br.last_swept_at ASC NULLS FIRST, ep.last_upload_at ASC
        LIMIT :limit
    """, nativeQuery = true)
    fun findDueBatches(
        @Param("quietCutoff") quietCutoff: OffsetDateTime,
        @Param("rescoreCutoff") rescoreCutoff: OffsetDateTime,
        @Param("adminClosedOnOrBefore") adminClosedOnOrBefore: LocalDate,
        @Param("ownerClosedOnOrBefore") ownerClosedOnOrBefore: LocalDate,
        @Param("limit") limit: Int,
    ): List<DueBatch>

    // The scoring lock (plan 5.3): two instances never score one batch at once.
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT b FROM BatchRelevance b WHERE b.id = :id")
    fun lockById(@Param("id") id: UUID): BatchRelevance?

    @Modifying
    @Query(value = "UPDATE batch_relevance SET last_swept_at = :now WHERE id = :id", nativeQuery = true)
    fun touchSwept(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    // Looks up the batch's next :limit photos, in upload order, and stores one row each (plan 5.2, D20).
    // The match is the nearest photo by another photographer in the same event uploaded strictly earlier,
    // over three pairs (clean/clean, centre/centre, clean/marked), kept when within :weakMax bits; the
    // tight/weak cut is made at scoring. The exact match is the earliest same-file photo by another
    // photographer in any event (deleted events too). Idempotent: ON CONFLICT means a second run, or a
    // second instance, inserts nothing. Returns the rows inserted.
    // `= eligible look-up photo`: LIVE, not trashed, removal NONE, has a clean hash, not yet looked up.
    @Modifying
    @Query(value = """
        INSERT INTO batch_relevance_photos
            (photo_id, batch_id, looked_up_at, match_photo_id, match_distance, match_pair, exact_photo_id)
        SELECT l.id, :batchId, :now, m.id, m.d,
               CASE WHEN m.id IS NULL THEN NULL
                    WHEN m.d = m.d_clean THEN 'CLEAN'
                    WHEN m.d = m.d_centre THEN 'CENTRE'
                    ELSE 'MARKED' END,
               x.id
        FROM (
            SELECT p.id, p.event_id, p.photographer_id, p.uploaded_at, p.content_hash, p.phash_clean, p.phash_centre
            FROM photos p
            WHERE p.event_id = :eventId
              AND p.photographer_id = :photographerId
              AND p.status = 'LIVE' AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
              AND p.phash_clean IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM batch_relevance_photos d WHERE d.photo_id = p.id)
            ORDER BY p.uploaded_at, p.id
            LIMIT :limit
        ) l
        LEFT JOIN LATERAL (
            SELECT o.id, c.d_clean, c.d_centre, c.d_marked,
                   LEAST(c.d_clean, c.d_centre, c.d_marked) AS d
            FROM photos o
            CROSS JOIN LATERAL (
                SELECT bit_count(CAST((l.phash_clean  # o.phash_clean)  AS bit(64))) AS d_clean,
                       bit_count(CAST((l.phash_centre # o.phash_centre) AS bit(64))) AS d_centre,
                       bit_count(CAST((l.phash_clean  # o.phash)        AS bit(64))) AS d_marked
            ) c
            WHERE o.event_id = l.event_id
              AND o.photographer_id <> l.photographer_id
              AND o.deleted_at IS NULL AND o.removal_state = 'NONE'
              AND o.uploaded_at < l.uploaded_at
              AND LEAST(c.d_clean, c.d_centre, c.d_marked) <= :weakMax
            ORDER BY LEAST(c.d_clean, c.d_centre, c.d_marked), o.uploaded_at, o.id
            LIMIT 1
        ) m ON TRUE
        LEFT JOIN LATERAL (
            SELECT o.id
            FROM photos o
            WHERE o.content_hash = l.content_hash
              AND o.photographer_id <> l.photographer_id
              AND o.deleted_at IS NULL AND o.removal_state = 'NONE'
              AND o.uploaded_at < l.uploaded_at
            ORDER BY o.uploaded_at, o.id
            LIMIT 1
        ) x ON TRUE
        ON CONFLICT (photo_id) DO NOTHING
    """, nativeQuery = true)
    fun lookUpNext(
        @Param("batchId") batchId: UUID,
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("limit") limit: Int,
        @Param("weakMax") weakMax: Int,
        @Param("now") now: OffsetDateTime,
    ): Int

    // 0 means every eligible photo has a look-up row and the batch can be scored. The service never infers
    // this from lookUpNext's count: another instance may have inserted the same rows.
    // `= eligible look-up photo` (as lookUpNext).
    @Query(value = """
        SELECT count(*) FROM photos p
        WHERE p.event_id = :eventId
          AND p.photographer_id = :photographerId
          AND p.status = 'LIVE' AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
          AND p.phash_clean IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM batch_relevance_photos d WHERE d.photo_id = p.id)
    """, nativeQuery = true)
    fun countPendingLookups(@Param("eventId") eventId: UUID, @Param("photographerId") photographerId: UUID): Long

    // The copy counts of one batch over its non-trashed photos not under a removal review (plan 5.3).
    // A match is tight within :tight bits and weak above it up to :weak.
    // `= batch evidence photo`: not trashed, removal NONE, LIVE or HIDDEN (a hidden batch keeps its evidence).
    @Query(value = """
        SELECT count(*) FILTER (WHERE p.status = 'LIVE') AS "photosLive",
               count(*) FILTER (WHERE p.status = 'PROCESSING') AS "photosProcessing",
               count(*) FILTER (WHERE p.status IN ('LIVE', 'HIDDEN') AND p.indexing_status = 'INDEXED') AS "indexedPhotos",
               count(r.photo_id) FILTER (WHERE p.status IN ('LIVE', 'HIDDEN')) AS "lookups",
               count(*) FILTER (WHERE p.status IN ('LIVE', 'HIDDEN') AND r.match_distance <= :tight) AS "tightHits",
               count(*) FILTER (WHERE p.status IN ('LIVE', 'HIDDEN')
                                  AND r.match_distance > :tight AND r.match_distance <= :weak) AS "weakHits",
               count(*) FILTER (WHERE p.status IN ('LIVE', 'HIDDEN') AND r.exact_photo_id IS NOT NULL) AS "exactMatches"
        FROM photos p
        LEFT JOIN batch_relevance_photos r ON r.photo_id = p.id
        WHERE p.event_id = :eventId
          AND p.photographer_id = :photographerId
          AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
    """, nativeQuery = true)
    fun batchCounts(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("tight") tight: Int,
        @Param("weak") weak: Int,
    ): BatchCounts

    // R2-23: call first inside each look-up transaction with app.relevance.lookup-timeout; the limit ends
    // with the transaction (SET LOCAL), and a timeout fails only that batch's step.
    @Query(value = "SELECT set_config('statement_timeout', CAST(:ms AS text), true)", nativeQuery = true)
    fun setLocalStatementTimeout(@Param("ms") ms: Long): String
}

interface BatchCounts {
    val photosLive: Long
    val photosProcessing: Long
    val indexedPhotos: Long
    val lookups: Long
    val tightHits: Long
    val weakHits: Long
    val exactMatches: Long
}

// lastUploadAt keeps microseconds: the due rule compares it with scored_upload_at for equality.
interface DueBatch {
    val eventId: UUID
    val photographerId: UUID
    val lastUploadAt: Instant
    val eventDate: LocalDate
    val sharedRace: Boolean
}
