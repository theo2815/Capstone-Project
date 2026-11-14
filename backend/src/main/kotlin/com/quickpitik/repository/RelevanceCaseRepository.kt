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

    // RV-1.8b: cases waiting for a verdict whose batch has no live claim, least recently served first (S3). A case
    // is listed once version 1 has scored the same upload, or once it is older than :waitCutoff (plan 5.9).
    @Query(value = """
        SELECT c.id AS "caseId", c.batch_id AS "batchId", br.event_id AS "eventId", br.photographer_id AS "photographerId"
        FROM relevance_cases c
        JOIN batch_relevance br ON br.id = c.batch_id
        WHERE c.stage = 'DECIDING' AND (br.check_claim_until IS NULL OR br.check_claim_until < :now)
          AND (br.scored_upload_at = c.upload_at OR c.created_at <= :waitCutoff)
        ORDER BY br.check_served_at ASC NULLS FIRST, c.created_at
        LIMIT :limit
    """, nativeQuery = true)
    fun findDeciding(@Param("now") now: OffsetDateTime, @Param("waitCutoff") waitCutoff: OffsetDateTime, @Param("limit") limit: Int): List<SamplingCase>

    // RV-1.8b: the deciding-stage writes. Each needs the stage DECIDING and the caller's claim on the batch (R5).
    // The raise is its own short write before each model call, so a call is paid for at most once per raise.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET decide_attempts = decide_attempts + 1
        WHERE c.id = :id AND c.stage = 'DECIDING' $HOLDS_CLAIM
    """, nativeQuery = true)
    fun raiseDecideAttempts(@Param("id") id: UUID, @Param("token") token: UUID): Int

    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET stage = 'CLOSED', closed_at = :now, close_reason = 'SKIPPED'
        WHERE c.id = :id AND c.stage = 'DECIDING' $HOLDS_CLAIM
    """, nativeQuery = true)
    fun closeSkippedDeciding(@Param("id") id: UUID, @Param("token") token: UUID, @Param("now") now: OffsetDateTime): Int

    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET stage = 'DECIDED', decided_at = :now, verdict = :verdict, verdict_note = :note,
            model_verdict = :modelVerdict, admin_reason = :adminReason, evidence_used = :evidenceUsed,
            summary_sent = :summary, model = :model, prompt_version = :promptVersion,
            copy_low = :copyLow, date_low = :dateLow, words_photos = :wordsPhotos
        WHERE c.id = :id AND c.stage = 'DECIDING' $HOLDS_CLAIM
    """, nativeQuery = true)
    fun saveVerdict(
        @Param("id") id: UUID,
        @Param("token") token: UUID,
        @Param("now") now: OffsetDateTime,
        @Param("verdict") verdict: String,
        @Param("note") note: String?,
        @Param("modelVerdict") modelVerdict: String?,
        @Param("adminReason") adminReason: String?,
        @Param("evidenceUsed") evidenceUsed: String?,
        @Param("summary") summary: String,
        @Param("model") model: String,
        @Param("promptVersion") promptVersion: String,
        @Param("copyLow") copyLow: Boolean?,
        @Param("dateLow") dateLow: Boolean?,
        @Param("wordsPhotos") wordsPhotos: Int,
    ): Int

    // The batch's current case (plan 5.1), if any.
    @Query(value = "SELECT * FROM relevance_cases WHERE batch_id = :batchId AND stage = 'APPLIED'", nativeQuery = true)
    fun currentFor(@Param("batchId") batchId: UUID): RelevanceCase?

    // RV-1.8b (P9): a RULES re-check takes the current case's sample as it is, so it measures nothing new.
    @Modifying
    @Query(value = """
        INSERT INTO relevance_case_photos (case_id, photo_id, round)
        SELECT :caseId, s.photo_id, s.round FROM relevance_case_photos s WHERE s.case_id = :fromCaseId
    """, nativeQuery = true)
    fun copySample(@Param("caseId") caseId: UUID, @Param("fromCaseId") fromCaseId: UUID): Int

    // RV-1.9 (plan 5.11): decided cases waiting to be applied, oldest verdict first. Applying reads and writes the
    // database only, so it needs no batch claim: every write below is guarded by the stage it expects.
    @Query(value = "SELECT * FROM relevance_cases WHERE stage = 'DECIDED' ORDER BY decided_at, created_at LIMIT :limit", nativeQuery = true)
    fun findDecided(@Param("limit") limit: Int): List<RelevanceCase>

    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET stage = 'CLOSED', closed_at = :now, close_reason = :reason
        WHERE id = :id AND stage = :fromStage
    """, nativeQuery = true)
    fun closeCase(
        @Param("id") id: UUID,
        @Param("fromStage") fromStage: String,
        @Param("reason") reason: String,
        @Param("now") now: OffsetDateTime,
    ): Int

    // Runs after closeCase of the old current case: the unique index on the current case allows no other order.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET stage = 'APPLIED', applied_mode = :mode, action = :action, applied_at = :now
        WHERE id = :id AND stage = 'DECIDED'
    """, nativeQuery = true)
    fun applyCase(@Param("id") id: UUID, @Param("mode") mode: String, @Param("action") action: String, @Param("now") now: OffsetDateTime): Int

    // The current case takes a newer check's copy and date results, so the RULES trigger does not fire again for them.
    @Modifying
    @Query(value = "UPDATE relevance_cases SET copy_low = :copyLow, date_low = :dateLow WHERE id = :id AND stage = 'APPLIED'", nativeQuery = true)
    fun copyRuleResults(@Param("id") id: UUID, @Param("copyLow") copyLow: Boolean?, @Param("dateLow") dateLow: Boolean?): Int
}

private const val HOLDS_CLAIM =
    "AND EXISTS (SELECT 1 FROM batch_relevance b WHERE b.id = c.batch_id AND b.check_claim = :token)"

interface SamplingCase {
    val caseId: UUID
    val batchId: UUID
    val eventId: UUID
    val photographerId: UUID
}
