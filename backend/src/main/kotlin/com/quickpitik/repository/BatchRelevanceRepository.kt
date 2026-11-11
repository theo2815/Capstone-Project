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
}

// lastUploadAt keeps microseconds: the due rule compares it with scored_upload_at for equality.
interface DueBatch {
    val eventId: UUID
    val photographerId: UUID
    val lastUploadAt: Instant
    val eventDate: LocalDate
    val sharedRace: Boolean
}
