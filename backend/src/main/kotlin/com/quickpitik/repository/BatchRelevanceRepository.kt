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

    // RV-1.5: the batches due for a version 2 check (plan 5.8), least recently served first. No live claim.
    @Query(value = "SELECT $CHECK_DUE_COLUMNS $CHECK_DUE_FROM" + """
          AND (br.check_claim_until IS NULL OR br.check_claim_until < :now)
        ORDER BY br.check_served_at ASC NULLS FIRST, ep.last_upload_at ASC
        LIMIT :limit
    """, nativeQuery = true)
    fun findCheckDue(
        @Param("quietCutoff") quietCutoff: OffsetDateTime,
        @Param("now") now: OffsetDateTime,
        @Param("countTrigger") countTrigger: Int,
        @Param("limit") limit: Int,
    ): List<CheckDue>

    // RV-1.5: the same rules for one batch the caller holds the claim on, read again after the claim so a
    // due list read before another copy started the check starts nothing (plan 5.8).
    @Query(value = "SELECT $CHECK_DUE_COLUMNS $CHECK_DUE_FROM AND br.id = :id AND br.check_claim = :token",
        nativeQuery = true)
    fun findCheckDueFor(
        @Param("id") id: UUID,
        @Param("token") token: UUID,
        @Param("quietCutoff") quietCutoff: OffsetDateTime,
        @Param("countTrigger") countTrigger: Int,
    ): CheckDue?

    // R5: one changed row means the claim is the caller's. It only stops two server copies working at once.
    @Modifying
    @Query(value = """
        UPDATE batch_relevance SET check_claim = :token, check_claim_until = :until, check_served_at = :now
        WHERE id = :id AND (check_claim_until IS NULL OR check_claim_until < :now)
    """, nativeQuery = true)
    fun claimCheck(
        @Param("id") id: UUID,
        @Param("token") token: UUID,
        @Param("until") until: OffsetDateTime,
        @Param("now") now: OffsetDateTime,
    ): Int

    @Modifying
    @Query(value = """
        UPDATE batch_relevance SET check_claim = NULL, check_claim_until = NULL
        WHERE id = :id AND check_claim = :token
    """, nativeQuery = true)
    fun releaseCheck(@Param("id") id: UUID, @Param("token") token: UUID): Int

    // The upload and photo count a started check covers; the due rules compare the next upload with them.
    @Modifying
    @Query(value = """
        UPDATE batch_relevance SET checked_upload_at = :uploadAt, checked_photos = :photos
        WHERE id = :id AND check_claim = :token
    """, nativeQuery = true)
    fun markChecked(
        @Param("id") id: UUID,
        @Param("token") token: UUID,
        @Param("uploadAt") uploadAt: OffsetDateTime,
        @Param("photos") photos: Int,
    ): Int

    // RV-1.6 (plan 5.8, invariant 12): the four things the due query checks, read again before each later step.
    @Query(value = """
        SELECT EXISTS (SELECT 1 FROM batch_relevance br
                       JOIN events e ON e.id = br.event_id AND e.deleted_at IS NULL AND e.hidden_at IS NULL
                                    AND e.visibility = 'PUBLIC'
                       JOIN users u ON u.id = br.photographer_id AND u.suspended_at IS NULL
                       WHERE br.id = :id)
    """, nativeQuery = true)
    fun stillAllowed(@Param("id") id: UUID): Boolean

    // RV-2.2a (P4): the legal version the batch's photographer accepted, for the consent check at apply.
    @Query(value = "SELECT u.terms_version FROM batch_relevance br JOIN users u ON u.id = br.photographer_id WHERE br.id = :id", nativeQuery = true)
    fun photographerTermsVersion(@Param("id") id: UUID): String?

    // RV-1.6 (S1): the case's sampled photos with no measurement and no error yet, in photo id order. A photo
    // measured by an earlier check has its row already and is skipped. Trashed or removal photos are left out.
    @Query(value = """
        SELECT s.photo_id AS "photoId", p.s3_key AS "s3Key"
        FROM relevance_case_photos s
        JOIN photos p ON p.id = s.photo_id AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
        LEFT JOIN relevance_photo_checks r ON r.photo_id = s.photo_id
        WHERE s.case_id = :caseId AND r.measured_at IS NULL AND r.measure_error IS NULL
        ORDER BY s.photo_id
        LIMIT :limit
    """, nativeQuery = true)
    fun nextToMeasure(@Param("caseId") caseId: UUID, @Param("limit") limit: Int): List<PhotoToMeasure>

    // RV-1.6: one photo's DetectLabels scores, written only while the caller holds the batch's claim (R5). Fills
    // the measurement columns only, so the words mark of RV-1.2 survives. Returns 0 for a lost claim.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_photo_checks
            (photo_id, batch_id, measured_at, fg_sharpness, fg_brightness, sharpness, brightness, person_conf, race_conf)
        SELECT :photoId, b.id, :now, CAST(:fgSharpness AS double precision), CAST(:fgBrightness AS double precision),
               CAST(:sharpness AS double precision), CAST(:brightness AS double precision), :personConf, :raceConf
        FROM batch_relevance b WHERE b.id = :batchId AND b.check_claim = :token
        ON CONFLICT (photo_id) DO UPDATE SET
            measured_at = EXCLUDED.measured_at, fg_sharpness = EXCLUDED.fg_sharpness,
            fg_brightness = EXCLUDED.fg_brightness, sharpness = EXCLUDED.sharpness, brightness = EXCLUDED.brightness,
            person_conf = EXCLUDED.person_conf, race_conf = EXCLUDED.race_conf
    """, nativeQuery = true)
    fun saveMeasurement(
        @Param("photoId") photoId: UUID,
        @Param("batchId") batchId: UUID,
        @Param("token") token: UUID,
        @Param("now") now: OffsetDateTime,
        @Param("fgSharpness") fgSharpness: Double?,
        @Param("fgBrightness") fgBrightness: Double?,
        @Param("sharpness") sharpness: Double?,
        @Param("brightness") brightness: Double?,
        @Param("personConf") personConf: Double,
        @Param("raceConf") raceConf: Double,
    ): Int

    // RV-1.6 (G4): a photo that can never be measured (BAD_IMAGE, NO_FILE). Never junk; left out of the share.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_photo_checks (photo_id, batch_id, measure_error)
        SELECT :photoId, b.id, :error FROM batch_relevance b WHERE b.id = :batchId AND b.check_claim = :token
        ON CONFLICT (photo_id) DO UPDATE SET measure_error = EXCLUDED.measure_error
    """, nativeQuery = true)
    fun saveMeasureError(
        @Param("photoId") photoId: UUID,
        @Param("batchId") batchId: UUID,
        @Param("token") token: UUID,
        @Param("error") error: String,
    ): Int

    // RV-1.6: a throttle, server error or timeout. The photo is tried again next tick; at :maxTries it is given up.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_photo_checks (photo_id, batch_id, measure_tries, measure_error)
        SELECT :photoId, b.id, 1, CASE WHEN :maxTries <= 1 THEN 'GAVE_UP' END
        FROM batch_relevance b WHERE b.id = :batchId AND b.check_claim = :token
        ON CONFLICT (photo_id) DO UPDATE SET
            measure_tries = relevance_photo_checks.measure_tries + 1,
            measure_error = CASE WHEN relevance_photo_checks.measure_tries + 1 >= :maxTries THEN 'GAVE_UP'
                                 ELSE relevance_photo_checks.measure_error END
    """, nativeQuery = true)
    fun saveMeasureRetry(
        @Param("photoId") photoId: UUID,
        @Param("batchId") batchId: UUID,
        @Param("token") token: UUID,
        @Param("maxTries") maxTries: Int,
    ): Int

    // RV-1.6: the stored scores of the case's sample (trashed and removal photos left out); junk is decided from
    // them at read time (RelevanceJunkRules). An unmeasured photo comes back with every score null.
    @Query(value = """
        SELECT r.fg_sharpness AS "fgSharpness", r.fg_brightness AS "fgBrightness", r.sharpness AS "sharpness",
               r.brightness AS "brightness", r.person_conf AS "personConf", r.race_conf AS "raceConf",
               r.measure_error AS "measureError"
        FROM relevance_case_photos s
        JOIN photos p ON p.id = s.photo_id AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
        LEFT JOIN relevance_photo_checks r ON r.photo_id = s.photo_id
        WHERE s.case_id = :caseId
    """, nativeQuery = true)
    fun sampleScores(@Param("caseId") caseId: UUID): List<PhotoScoresRow>

    // RV-1.2 (plan 5.5, R5): marks the photo's words as counted. 1 the first time, 0 on a re-read, so the caller
    // adds a photo's words once. A row the measuring wrote first keeps its scores.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_photo_checks (photo_id, batch_id, words_counted_at)
        VALUES (:photoId, :batchId, :now)
        ON CONFLICT (photo_id) DO UPDATE SET words_counted_at = EXCLUDED.words_counted_at
        WHERE relevance_photo_checks.words_counted_at IS NULL
    """, nativeQuery = true)
    fun markWordsCounted(@Param("photoId") photoId: UUID, @Param("batchId") batchId: UUID, @Param("now") now: OffsetDateTime): Int

    // RV-1.2: one photo's distinct words, sorted by the caller and again here, so two indexing threads lock the
    // count rows in the same order and cannot deadlock.
    @Modifying
    @Query(value = """
        INSERT INTO batch_relevance_words (batch_id, word, photos)
        SELECT :batchId, w, 1 FROM unnest(CAST(:words AS text[])) AS w ORDER BY w
        ON CONFLICT (batch_id, word) DO UPDATE SET photos = batch_relevance_words.photos + 1
    """, nativeQuery = true)
    fun addWords(@Param("batchId") batchId: UUID, @Param("words") words: Array<String>): Int

    // RV-1.2 (R11): N of "words from N of M photos".
    @Query(value = "SELECT count(*) FROM relevance_photo_checks WHERE batch_id = :batchId AND words_counted_at IS NOT NULL",
        nativeQuery = true)
    fun countWordPhotos(@Param("batchId") batchId: UUID): Long

    // RV-1.2 (G14): for up to :limit batches finally scored on or before :settledBefore and not yet purged, deletes
    // the words on under :minSharePercent of the batch's counted photos, then stamps the batch so it is purged once.
    // Returns the batches stamped.
    @Modifying
    @Query(value = """
        WITH due AS (
            SELECT br.id,
                   (SELECT count(*) FROM relevance_photo_checks r
                    WHERE r.batch_id = br.id AND r.words_counted_at IS NOT NULL) AS counted
            FROM batch_relevance br
            WHERE br.final_scored_at <= :settledBefore AND br.words_purged_at IS NULL
            ORDER BY br.final_scored_at, br.id
            LIMIT :limit
        ), purged AS (
            DELETE FROM batch_relevance_words w USING due
            WHERE w.batch_id = due.id AND w.photos * 100 < :minSharePercent * due.counted
        )
        UPDATE batch_relevance br SET words_purged_at = :now FROM due WHERE br.id = due.id
    """, nativeQuery = true)
    fun purgeRareWords(
        @Param("minSharePercent") minSharePercent: Int,
        @Param("settledBefore") settledBefore: OffsetDateTime,
        @Param("limit") limit: Int,
        @Param("now") now: OffsetDateTime,
    ): Int

    // RV-1.2: a completed erasure removes the photo's words mark and measurement.
    @Modifying
    @Query(value = "DELETE FROM relevance_photo_checks WHERE photo_id = :photoId", nativeQuery = true)
    fun deletePhotoCheck(@Param("photoId") photoId: UUID): Int

    // RV-1.8a (plan 5.9, G13): the batch's most common words, each on at least :minSharePercent of its counted
    // photos (the purge's own line, so a word the purge would keep is never left out here).
    @Query(value = """
        SELECT w.word AS "word", w.photos AS "photos"
        FROM batch_relevance_words w
        WHERE w.batch_id = :batchId
          AND w.photos * 100 >= :minSharePercent * (SELECT count(*) FROM relevance_photo_checks r
                                                    WHERE r.batch_id = :batchId AND r.words_counted_at IS NOT NULL)
        ORDER BY w.photos DESC, w.word
        LIMIT :limit
    """, nativeQuery = true)
    fun topWords(@Param("batchId") batchId: UUID, @Param("minSharePercent") minSharePercent: Int, @Param("limit") limit: Int): List<WordCount>

    // RV-1.8a (P13): for each of :words, how many other batches of the event have it on at least :minSharePercent of
    // their counted photos. One batch is one photographer, so a photographer who prints signs counts once.
    @Query(value = """
        SELECT w.word AS "word", CAST(count(*) AS int) AS "photographers"
        FROM batch_relevance br
        JOIN batch_relevance_words w ON w.batch_id = br.id
        WHERE br.event_id = :eventId AND br.id <> :exceptBatchId
          AND w.word = ANY (CAST(:words AS text[]))
          AND w.photos * 100 >= :minSharePercent * (SELECT count(*) FROM relevance_photo_checks r
                                                    WHERE r.batch_id = br.id AND r.words_counted_at IS NOT NULL)
        GROUP BY w.word
    """, nativeQuery = true)
    fun eventWords(
        @Param("eventId") eventId: UUID,
        @Param("exceptBatchId") exceptBatchId: UUID,
        @Param("words") words: Array<String>,
        @Param("minSharePercent") minSharePercent: Int,
    ): List<WordPhotographers>

    // RV-1.8a: the event's other photographers that have a batch (batch_relevance is one row per photographer).
    @Query(value = "SELECT CAST(count(*) AS int) FROM batch_relevance WHERE event_id = :eventId AND id <> :exceptBatchId",
        nativeQuery = true)
    fun otherPhotographers(@Param("eventId") eventId: UUID, @Param("exceptBatchId") exceptBatchId: UUID): Int
}

interface WordCount {
    val word: String
    val photos: Int
}

interface WordPhotographers {
    val word: String
    val photographers: Int
}

// RV-1.6: one sampled photo to measure; s3Key is the original, never the watermarked preview.
interface PhotoToMeasure {
    val photoId: UUID
    val s3Key: String
}

interface PhotoScoresRow {
    val fgSharpness: Double?
    val fgBrightness: Double?
    val sharpness: Double?
    val brightness: Double?
    val personConf: Double?
    val raceConf: Double?
    val measureError: String?
}

// RV-1.5 (plan 5.8). A batch is due when its event is public, not hidden and not deleted, its photographer is
// not suspended, no case of it is in flight, and either it went quiet with a newest upload not yet checked
// (QUIET) or it is still uploading and grew by :countTrigger photos since its last check (COUNT).
// event_photographer.photo_count is what the check records, so the COUNT rule compares like with like.
// RV-1.8b (P9): or it is quiet, version 1 has scored the newest upload, and its copy or date result is no longer the one the
// current case was decided with (RULES). A new upload wins: it is checked with a new sample.
private const val QUIET_DUE = "(ep.last_upload_at <= :quietCutoff AND br.checked_upload_at IS DISTINCT FROM ep.last_upload_at)"
private const val COUNT_DUE = "(ep.last_upload_at > :quietCutoff AND ep.photo_count - COALESCE(br.checked_photos, 0) >= :countTrigger)"
private const val RULES_DUE = """(ep.last_upload_at <= :quietCutoff AND br.scored_upload_at = ep.last_upload_at
     AND EXISTS (SELECT 1 FROM relevance_cases a WHERE a.batch_id = br.id AND a.stage = 'APPLIED'
                 AND (a.copy_low IS DISTINCT FROM br.copy_low OR a.date_low IS DISTINCT FROM br.date_low)))"""

private const val CHECK_DUE_COLUMNS = """
    br.id AS "batchId", ep.event_id AS "eventId", ep.photographer_id AS "photographerId",
    ep.last_upload_at AS "lastUploadAt", ep.photo_count AS "photoCount",
    CASE WHEN $QUIET_DUE THEN 'QUIET' WHEN $COUNT_DUE THEN 'COUNT' ELSE 'RULES' END AS "trigger"
"""

private const val CHECK_DUE_FROM = """
    FROM event_photographer ep
    JOIN events e ON e.id = ep.event_id AND e.deleted_at IS NULL AND e.hidden_at IS NULL AND e.visibility = 'PUBLIC'
    JOIN users u ON u.id = ep.photographer_id AND u.suspended_at IS NULL
    LEFT JOIN batch_relevance br ON br.event_id = ep.event_id AND br.photographer_id = ep.photographer_id
    WHERE NOT EXISTS (SELECT 1 FROM relevance_cases c
                      WHERE c.batch_id = br.id AND c.stage IN ('SAMPLING', 'DECIDING', 'DECIDED'))
      AND ($QUIET_DUE OR $COUNT_DUE OR $RULES_DUE)
"""

// lastUploadAt keeps microseconds: the QUIET rule compares it with checked_upload_at for equality.
interface CheckDue {
    val batchId: UUID?
    val eventId: UUID
    val photographerId: UUID
    val lastUploadAt: Instant
    val photoCount: Int
    val trigger: String
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
