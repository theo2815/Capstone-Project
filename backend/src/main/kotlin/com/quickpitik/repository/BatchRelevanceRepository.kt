package com.quickpitik.repository

import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.BatchRelevance
import com.quickpitik.entity.PhotoMetadataState
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
    // the final pass, unsettled work or closed uploads once :rescoreCutoff has passed; or, when
    // :raiseUnflagged (flag mode with the flags switch on), a Low batch not yet linked to a flag, such as one
    // that went Low in shadow mode (PE-2.1, plan 5.9). Uploads close
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
                        OR (e.created_by IS NOT NULL AND e.date <= :ownerClosedOnOrBefore)))
               OR (:raiseUnflagged AND br.level = 'LOW' AND br.flag_id IS NULL))
        ORDER BY br.last_swept_at ASC NULLS FIRST, ep.last_upload_at ASC
        LIMIT :limit
    """, nativeQuery = true)
    fun findDueBatches(
        @Param("quietCutoff") quietCutoff: OffsetDateTime,
        @Param("rescoreCutoff") rescoreCutoff: OffsetDateTime,
        @Param("adminClosedOnOrBefore") adminClosedOnOrBefore: LocalDate,
        @Param("ownerClosedOnOrBefore") ownerClosedOnOrBefore: LocalDate,
        @Param("limit") limit: Int,
        @Param("raiseUnflagged") raiseUnflagged: Boolean,
    ): List<DueBatch>

    // R2-10: another photographer's batch in the event whose capture dates fit the current event date.
    // Without one, a date-Low batch may mean the event's date is wrong, so no automatic note is sent.
    @Query(value = """
        SELECT EXISTS (SELECT 1 FROM batch_relevance
                       WHERE event_id = :eventId AND photographer_id <> :photographerId
                         AND scored_event_date = :eventDate AND date_low = false)
    """, nativeQuery = true)
    fun existsOtherDateClear(
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("eventDate") eventDate: LocalDate,
    ): Boolean

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

    // The batch's capture times for the date rule (plan 5.3 item 2), read through the entity mapping and
    // never by a SQL cast: the stored JSON shape of takenAtLocal is Hibernate's to choose.
    // `= batch evidence photo` (as batchCounts).
    @Query("""
        SELECT m.photoId AS photoId, m.state AS state, m.fields AS fields
        FROM PhotoMetadata m, Photo p
        WHERE p.id = m.photoId AND p.eventId = :eventId AND p.photographerId = :photographerId
            AND p.deletedAt IS NULL AND p.removalState = com.quickpitik.entity.RemovalState.NONE
            AND p.status IN (com.quickpitik.entity.PhotoStatus.LIVE, com.quickpitik.entity.PhotoStatus.HIDDEN)
        ORDER BY p.uploadedAt, m.photoId
    """)
    fun metadataRows(@Param("eventId") eventId: UUID, @Param("photographerId") photographerId: UUID): List<BatchMetadataRow>

    // Photos whose XMP marker names a photo of another photographer, in any event and of any status
    // (plan 5.3 item 3). Evidence only (C16). `<>` drops a named photo with no photographer; a trashed or
    // removal-review photo is no match, as in lookUpNext.
    // `= batch evidence photo` (as batchCounts).
    @Query(value = """
        SELECT count(*) FROM photos p
        JOIN photo_metadata m ON m.photo_id = p.id
        JOIN photos o ON o.id = m.xmp_photo_id
        WHERE p.event_id = :eventId
          AND p.photographer_id = :photographerId
          AND p.deleted_at IS NULL AND p.removal_state = 'NONE' AND p.status IN ('LIVE', 'HIDDEN')
          AND o.photographer_id <> p.photographer_id
          AND o.deleted_at IS NULL AND o.removal_state = 'NONE'
    """, nativeQuery = true)
    fun xmpMatches(@Param("eventId") eventId: UUID, @Param("photographerId") photographerId: UUID): Long

    // Bibbed batch photos, and how many share at least one bib word with another photographer's photo in
    // the same event (plan 5.3 item 4). Goes through photos by event, never through the global bib_number
    // index (the eventsSharingBibs shape), so a word seen only in another event is not an overlap.
    // `= batch evidence photo` (as batchCounts) on the batch side.
    @Query(value = """
        WITH others AS MATERIALIZED (
            SELECT DISTINCT b.bib_number
            FROM photos o
            JOIN photo_bibs b ON b.photo_id = o.id
            WHERE o.event_id = :eventId
              AND o.photographer_id <> :photographerId
              AND o.deleted_at IS NULL AND o.removal_state = 'NONE'
        )
        SELECT count(*) AS "bibbedPhotos", count(*) FILTER (WHERE x.overlaps) AS "bibOverlapPhotos"
        FROM (
            SELECT bool_or(EXISTS (SELECT 1 FROM others WHERE others.bib_number = b.bib_number)) AS overlaps
            FROM photos p
            JOIN photo_bibs b ON b.photo_id = p.id
            WHERE p.event_id = :eventId
              AND p.photographer_id = :photographerId
              AND p.deleted_at IS NULL AND p.removal_state = 'NONE' AND p.status IN ('LIVE', 'HIDDEN')
            GROUP BY p.id
        ) x
    """, nativeQuery = true)
    fun bibOverlap(@Param("eventId") eventId: UUID, @Param("photographerId") photographerId: UUID): BibOverlap

    // PE-2.3: the batches behind one relevance flag (the admin's evidence read).
    fun findByFlagId(flagId: UUID): List<BatchRelevance>

    // PE-2.3: a batch's look-up rows that found a match or an exact copy, nearest first. Over
    // `= batch evidence photo` (as batchCounts), so the samples survive a batch hide.
    @Query(value = """
        SELECT r.photo_id AS "photoId", r.match_photo_id AS "matchPhotoId",
               CAST(r.match_distance AS int) AS "distance", r.match_pair AS "pair", r.exact_photo_id AS "exactPhotoId"
        FROM batch_relevance_photos r
        JOIN photos p ON p.id = r.photo_id
        WHERE r.batch_id = :batchId
          AND (r.match_photo_id IS NOT NULL OR r.exact_photo_id IS NOT NULL)
          AND p.deleted_at IS NULL AND p.removal_state = 'NONE' AND p.status IN ('LIVE', 'HIDDEN')
        ORDER BY r.match_distance NULLS LAST, p.uploaded_at, p.id
        LIMIT :limit
    """, nativeQuery = true)
    fun copySamples(@Param("batchId") batchId: UUID, @Param("limit") limit: Int): List<CopySampleRow>

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

// photoId is for PE-2.3's date samples.
interface BatchMetadataRow {
    val photoId: UUID
    val state: PhotoMetadataState
    val fields: PhotoMetadataFields?
}

interface CopySampleRow {
    val photoId: UUID
    val matchPhotoId: UUID?
    val distance: Int?
    val pair: String?
    val exactPhotoId: UUID?
}

interface BibOverlap {
    val bibbedPhotos: Long
    val bibOverlapPhotos: Long
}

// lastUploadAt keeps microseconds: the due rule compares it with scored_upload_at for equality.
interface DueBatch {
    val eventId: UUID
    val photographerId: UUID
    val lastUploadAt: Instant
    val eventDate: LocalDate
    val sharedRace: Boolean
}
