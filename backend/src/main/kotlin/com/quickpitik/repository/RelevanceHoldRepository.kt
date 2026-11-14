package com.quickpitik.repository

import com.quickpitik.entity.BatchRelevance
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.Repository
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

// RV-2.1 (part 2, A.2): batch holds in flag_hidden_photos (V106), one kind (held_by) at a time. Written only by
// service/relevance; flag rows stay service/admin's (FlagRepository). flag_hidden_photos has no entity, so this
// repository hangs off BatchRelevance and holds native queries only. Callers run each chunk in its own short write
// transaction: hold rows first, then PhotoRepository.lockIdsForUpdate on the returned ids, then hidePhotos or
// unhideReleased. The id lists must not be empty (IN () is not SQL).
interface RelevanceHoldRepository : Repository<BatchRelevance, UUID> {

    // Hide step for target HIDE_ALL: hold up to :chunk photos of the batch that this kind does not hold yet. A photo a
    // person restored under this case is never held again by the same case; one restored automatically is.
    @Query(value = """
        INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at)
        SELECT :batchId, p.id, :heldBy, :caseId, :now FROM photos p
        WHERE p.event_id = :eventId AND p.photographer_id = :photographerId AND p.deleted_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM flag_hidden_photos h
              WHERE h.batch_id = :batchId AND h.photo_id = p.id AND h.held_by = :heldBy
                AND (h.restored_at IS NULL OR (h.case_id = :caseId AND h.restored_by IS DISTINCT FROM 'AUTO')))
        ORDER BY p.id LIMIT :chunk
        ON CONFLICT (batch_id, photo_id, held_by) DO UPDATE
            SET restored_at = NULL, restored_by = NULL, hidden_at = EXCLUDED.hidden_at, case_id = EXCLUDED.case_id
        RETURNING photo_id
    """, nativeQuery = true)
    fun hideChunk(
        @Param("batchId") batchId: UUID,
        @Param("caseId") caseId: UUID,
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("heldBy") heldBy: String,
        @Param("now") now: OffsetDateTime,
        @Param("chunk") chunk: Int,
    ): List<UUID>

    // RV-2.3b: the hide step for target HIDE_JUNK. hideChunk limited to photos measured as junk (JUNK_PREDICATE).
    @Query(value = """
        INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at)
        SELECT :batchId, p.id, 'AUTO', :caseId, :now FROM photos p
        JOIN relevance_photo_checks k ON k.photo_id = p.id
        WHERE p.event_id = :eventId AND p.photographer_id = :photographerId AND p.deleted_at IS NULL
          AND $JUNK_PREDICATE
          AND NOT EXISTS (
              SELECT 1 FROM flag_hidden_photos h
              WHERE h.batch_id = :batchId AND h.photo_id = p.id AND h.held_by = 'AUTO'
                AND (h.restored_at IS NULL OR (h.case_id = :caseId AND h.restored_by IS DISTINCT FROM 'AUTO')))
        ORDER BY p.id LIMIT :chunk
        ON CONFLICT (batch_id, photo_id, held_by) DO UPDATE
            SET restored_at = NULL, restored_by = NULL, hidden_at = EXCLUDED.hidden_at, case_id = EXCLUDED.case_id
        RETURNING photo_id
    """, nativeQuery = true)
    fun hideJunkChunk(
        @Param("batchId") batchId: UUID,
        @Param("caseId") caseId: UUID,
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("now") now: OffsetDateTime,
        @Param("chunk") chunk: Int,
        @Param("blurCutoff") blurCutoff: Double,
        @Param("darkCutoff") darkCutoff: Double,
        @Param("labelMin") labelMin: Double,
    ): List<UUID>

    // RV-2.3b: the restore step for target HIDE_JUNK. Up to :chunk live automatic holds of the batch, whichever case they
    // name, whose photo was measured and is not junk; an unmeasured photo stays held until the scan measures it (A.2).
    @Query(value = """
        UPDATE flag_hidden_photos SET restored_at = :now, restored_by = 'AUTO'
        WHERE id IN (SELECT h.id FROM flag_hidden_photos h
                     JOIN relevance_photo_checks k ON k.photo_id = h.photo_id
                     WHERE h.batch_id = :batchId AND h.held_by = 'AUTO' AND h.restored_at IS NULL
                       AND k.measured_at IS NOT NULL AND k.measure_error IS NULL AND NOT ($JUNK_PREDICATE)
                     ORDER BY h.photo_id LIMIT :chunk)
        RETURNING photo_id
    """, nativeQuery = true)
    fun restoreNotJunkChunk(
        @Param("batchId") batchId: UUID,
        @Param("now") now: OffsetDateTime,
        @Param("chunk") chunk: Int,
        @Param("blurCutoff") blurCutoff: Double,
        @Param("darkCutoff") darkCutoff: Double,
        @Param("labelMin") labelMin: Double,
    ): List<UUID>

    // Restore step for target NONE: stamp up to :chunk live holds of this kind as restored by :restoredBy.
    @Query(value = """
        UPDATE flag_hidden_photos SET restored_at = :now, restored_by = :restoredBy
        WHERE id IN (SELECT id FROM flag_hidden_photos
                     WHERE batch_id = :batchId AND held_by = :heldBy AND restored_at IS NULL
                     ORDER BY photo_id LIMIT :chunk)
        RETURNING photo_id
    """, nativeQuery = true)
    fun restoreChunk(
        @Param("batchId") batchId: UUID,
        @Param("heldBy") heldBy: String,
        @Param("restoredBy") restoredBy: String,
        @Param("now") now: OffsetDateTime,
        @Param("chunk") chunk: Int,
    ): List<UUID>

    // Conditional, like PhotoRepository.hideRecordedForFlag, so a row the pipeline is writing is not clobbered.
    @Modifying
    @Query(value = "UPDATE photos SET status = 'HIDDEN' WHERE id IN (:ids) AND status IN ('LIVE', 'PROCESSING')", nativeQuery = true)
    fun hidePhotos(@Param("ids") ids: Collection<UUID>): Int

    // The one restore rule (part 1, 5.10), after restoreChunk stamped the caller's holds: a photo comes back only when no
    // un-restored hold row and no hidden photo flag still holds it. PROCESSING when it never got a watermark, as
    // AdminFlagService.unhidePhoto and PhotoRepository.unhideRecordedForFlag do.
    @Modifying
    @Query(value = """
        UPDATE photos p
        SET status = CASE WHEN p.watermark_s3_key IS NULL THEN 'PROCESSING' ELSE 'LIVE' END, processing_attempts = 0
        WHERE p.id IN (:ids) AND p.status = 'HIDDEN'
          AND NOT EXISTS (SELECT 1 FROM flags f WHERE f.target_kind = 'photo' AND f.target_id = p.id AND f.status = 'hidden')
          AND NOT EXISTS (SELECT 1 FROM flag_hidden_photos o WHERE o.photo_id = p.id AND o.restored_at IS NULL)
    """, nativeQuery = true)
    fun unhideReleased(@Param("ids") ids: Collection<UUID>): Int

    @Query(value = "SELECT count(*) FROM flag_hidden_photos WHERE batch_id = :batchId AND held_by = :heldBy AND restored_at IS NULL",
        nativeQuery = true)
    fun countLiveHolds(@Param("batchId") batchId: UUID, @Param("heldBy") heldBy: String): Long

    // RV-2.3a (A.4): the finished full scan's counts, from one aggregate over the batch's photos that are not trashed and
    // not removed. Measured leaves out photos that could not be measured; blurry uses the predicate's blur line.
    @Query(value = """
        SELECT count(*) FILTER (WHERE k.measured_at IS NOT NULL AND k.measure_error IS NULL) AS "measured",
               count(*) FILTER (WHERE $JUNK_PREDICATE) AS "junk",
               count(*) FILTER (WHERE k.measured_at IS NOT NULL AND k.measure_error IS NULL
                                  AND COALESCE(k.fg_sharpness, k.sharpness) < :blurCutoff) AS "blurry"
        FROM relevance_photo_checks k
        JOIN photos p ON p.id = k.photo_id AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
        WHERE k.batch_id = :batchId
    """, nativeQuery = true)
    fun scanCounts(
        @Param("batchId") batchId: UUID,
        @Param("blurCutoff") blurCutoff: Double,
        @Param("darkCutoff") darkCutoff: Double,
        @Param("labelMin") labelMin: Double,
    ): ScanCounts
}

// RV-2.3a (part 2, A.2): RelevanceJunkRules.isJunk in SQL, over relevance_photo_checks aliased k, for every query that
// needs it (RelevanceScanIntegrationTest proves they agree). Scores are bound as doubles; a measured photo stores label
// confidence 0, never NULL; never run it with a cut-off unset (the caller returns first).
const val JUNK_PREDICATE = """k.measured_at IS NOT NULL AND k.measure_error IS NULL AND (
    COALESCE(k.fg_sharpness, k.sharpness) < :blurCutoff
    OR k.brightness < :darkCutoff
    OR (COALESCE(k.person_conf, 0) < :labelMin AND COALESCE(k.race_conf, 0) < :labelMin))"""

interface ScanCounts {
    val measured: Long
    val junk: Long
    val blurry: Long
}
