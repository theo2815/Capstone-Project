package com.quickpitik.repository

import com.quickpitik.entity.RelevanceCase
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

// RV-1.1: relevance_cases, written only by service/relevance. Queries come with the tasks that need them.
interface RelevanceCaseRepository : JpaRepository<RelevanceCase, UUID> {

    // RV-1.5 (plan 5.8): up to :sampleSize random photos of the batch not yet in this case's sample, from
    // every photo not trashed and not under removal, whatever its status: a hidden junk batch topped up
    // with good photos must not look clean. Returns the photos added.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_case_photos (case_id, photo_id, round)
        SELECT :caseId, p.id, :round FROM photos p
        WHERE p.event_id = :eventId AND p.photographer_id = :photographerId
          AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
          AND NOT EXISTS (SELECT 1 FROM relevance_case_photos s WHERE s.case_id = :caseId AND s.photo_id = p.id)
        ORDER BY random() LIMIT :sampleSize
    """, nativeQuery = true)
    fun pickSample(
        @Param("caseId") caseId: UUID,
        @Param("eventId") eventId: UUID,
        @Param("photographerId") photographerId: UUID,
        @Param("round") round: Int,
        @Param("sampleSize") sampleSize: Int,
    ): Int

    @Query(value = "SELECT CAST(count(*) AS int) FROM relevance_case_photos WHERE case_id = :caseId", nativeQuery = true)
    fun countSample(@Param("caseId") caseId: UUID): Int

    // RV-1.6: cases still sampling whose batch has no live claim, least recently served first (S3).
    @Query(value = """
        SELECT c.id AS "caseId", c.batch_id AS "batchId", br.event_id AS "eventId", br.photographer_id AS "photographerId"
        FROM relevance_cases c
        JOIN batch_relevance br ON br.id = c.batch_id
        WHERE c.stage = 'SAMPLING' AND (br.check_claim_until IS NULL OR br.check_claim_until < :now)
        ORDER BY br.check_served_at ASC NULLS FIRST, c.created_at
        LIMIT :limit
    """, nativeQuery = true)
    fun findSampling(@Param("now") now: OffsetDateTime, @Param("limit") limit: Int): List<SamplingCase>

    // RV-1.6: the sampling-stage writes. Each needs the stage SAMPLING and the caller's claim on the batch (R5).
    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET stage = 'CLOSED', closed_at = :now, close_reason = 'SKIPPED'
        WHERE c.id = :id AND c.stage = 'SAMPLING' $HOLDS_CLAIM
    """, nativeQuery = true)
    fun closeSkipped(@Param("id") id: UUID, @Param("token") token: UUID, @Param("now") now: OffsetDateTime): Int

    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET sample_round = 2
        WHERE c.id = :id AND c.stage = 'SAMPLING' AND c.sample_round = 1 $HOLDS_CLAIM
    """, nativeQuery = true)
    fun startRoundTwo(@Param("id") id: UUID, @Param("token") token: UUID): Int

    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET stage = 'DECIDING', sample_size = :sampleSize, measured = :measured,
            blurry = :blurry, dark = :dark, not_race = :notRace, junk = :junk, band = :band
        WHERE c.id = :id AND c.stage = 'SAMPLING' $HOLDS_CLAIM
    """, nativeQuery = true)
    fun finishSampling(
        @Param("id") id: UUID,
        @Param("token") token: UUID,
        @Param("sampleSize") sampleSize: Int,
        @Param("measured") measured: Int,
        @Param("blurry") blurry: Int,
        @Param("dark") dark: Int,
        @Param("notRace") notRace: Int,
        @Param("junk") junk: Int,
        @Param("band") band: String,
    ): Int
}

private const val HOLDS_CLAIM =
    "AND EXISTS (SELECT 1 FROM batch_relevance b WHERE b.id = c.batch_id AND b.check_claim = :token)"

interface SamplingCase {
    val caseId: UUID
    val batchId: UUID
    val eventId: UUID
    val photographerId: UUID
}
