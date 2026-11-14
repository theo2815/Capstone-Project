package com.quickpitik.repository

import com.quickpitik.entity.RelevanceCase
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.Instant
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

    // RV-1.10 (plan 5.11): one tab of the admin page, current cases only, newest first. :tab is 'hidden' (an action
    // other than NONE), 'cant_tell' or 'all'; the service checks it.
    @Query(value = "SELECT $CASE_ROW $CASE_FROM WHERE c.stage = 'APPLIED' AND $CASE_TAB ORDER BY c.applied_at DESC, c.id",
        countQuery = "SELECT count(*) $CASE_FROM WHERE c.stage = 'APPLIED' AND $CASE_TAB", nativeQuery = true)
    fun pageCurrentCases(@Param("tab") tab: String, pageable: Pageable): Page<RelevanceCaseRow>

    @Query(value = "SELECT $CASE_ROW $CASE_FROM WHERE c.id = :id", nativeQuery = true)
    fun caseRow(@Param("id") id: UUID): RelevanceCaseRow?

    // RV-1.10: the case's measured sample photos (trashed, removal and failed photos left out) with their stored scores
    // and preview keys, never the original's key (G19). Junk or clean is decided from the scores (RelevanceJunkRules).
    @Query(value = """
        SELECT s.photo_id AS "photoId", p.watermark_s3_key AS "watermarkKey", p.thumbnail_s3_key AS "thumbnailKey",
               r.fg_sharpness AS "fgSharpness", r.fg_brightness AS "fgBrightness", r.sharpness AS "sharpness",
               r.brightness AS "brightness", r.person_conf AS "personConf", r.race_conf AS "raceConf"
        FROM relevance_case_photos s
        JOIN photos p ON p.id = s.photo_id AND p.deleted_at IS NULL AND p.removal_state = 'NONE'
        JOIN relevance_photo_checks r ON r.photo_id = s.photo_id AND r.measured_at IS NOT NULL AND r.measure_error IS NULL
        WHERE s.case_id = :caseId
        ORDER BY s.round, s.photo_id
    """, nativeQuery = true)
    fun measuredSample(@Param("caseId") caseId: UUID): List<SamplePhotoRow>
}

private const val HOLDS_CLAIM =
    "AND EXISTS (SELECT 1 FROM batch_relevance b WHERE b.id = c.batch_id AND b.check_claim = :token)"

private const val CASE_ROW = """c.id AS "caseId", c.batch_id AS "batchId", br.event_id AS "eventId", e.name AS "eventName",
    ps.handle AS "photographerHandle", u.name AS "photographerName", c.verdict AS "verdict", c.verdict_note AS "verdictNote",
    c.action AS "action", c.applied_mode AS "appliedMode", c.batch_photos AS "batchPhotos", c.measured AS "measured",
    c.junk AS "junk", c.decided_at AS "decidedAt""""

private const val CASE_FROM = """FROM relevance_cases c
    JOIN batch_relevance br ON br.id = c.batch_id
    JOIN events e ON e.id = br.event_id
    JOIN users u ON u.id = br.photographer_id
    LEFT JOIN photographer_settings ps ON ps.user_id = br.photographer_id"""

private const val CASE_TAB =
    "(:tab = 'all' OR (:tab = 'hidden' AND c.action <> 'NONE') OR (:tab = 'cant_tell' AND c.verdict = 'CANT_TELL'))"

interface RelevanceCaseRow {
    val caseId: UUID
    val batchId: UUID
    val eventId: UUID
    val eventName: String
    val photographerHandle: String?
    val photographerName: String
    val verdict: String?
    val verdictNote: String?
    val action: String?
    val appliedMode: String?
    val batchPhotos: Int
    val measured: Int?
    val junk: Int?
    val decidedAt: Instant?
}

interface SamplePhotoRow {
    val photoId: UUID
    val watermarkKey: String?
    val thumbnailKey: String?
    val fgSharpness: Double?
    val fgBrightness: Double?
    val sharpness: Double?
    val brightness: Double?
    val personConf: Double?
    val raceConf: Double?
}

interface SamplingCase {
    val caseId: UUID
    val batchId: UUID
    val eventId: UUID
    val photographerId: UUID
}
