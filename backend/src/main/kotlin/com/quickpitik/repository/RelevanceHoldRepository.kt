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
}
