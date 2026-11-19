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

    // RVF-4: the batch's latest case, whatever its stage; a RETRY check starts from it (the due rule reads the same one).
    @Query(value = "SELECT * FROM relevance_cases WHERE batch_id = :batchId ORDER BY created_at DESC, id DESC LIMIT 1", nativeQuery = true)
    fun latestFor(@Param("batchId") batchId: UUID): RelevanceCase?

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

    // RV-2.2a (A.3, P21): with no current case, the batch's last case an admin closed is the comparator, so an
    // admin's decision is not undone by the next upload.
    @Query(value = """
        SELECT * FROM relevance_cases
        WHERE batch_id = :batchId AND stage = 'CLOSED' AND close_reason IN ('RESTORED_BY_ADMIN', 'APPEAL_WON', 'LOOKS_FINE')
        ORDER BY closed_at DESC, created_at DESC LIMIT 1
    """, nativeQuery = true)
    fun lastClosedByAdmin(@Param("batchId") batchId: UUID): RelevanceCase?

    // RV-2.2b, RV-2.3b (A.2, P18): the current auto-hide cases whose batch holds must follow their target: every HIDE_ALL
    // and HIDE_JUNK case, and a NONE case only while its batch has a live automatic hold left to restore. RV-L7 part a:
    // withdrawn marks a photographer who withdrew the consent to automatic actions: the hold step hides nothing more for
    // them (isHideTarget) but still restores, so a won appeal or a lowered target still puts photos back.
    @Query(value = """
        SELECT c.id AS "caseId", c.batch_id AS "batchId", br.event_id AS "eventId", br.photographer_id AS "photographerId",
               (c.hide_finished_at IS NOT NULL) AS "hideFinished", c.action AS "action",
               (c.scan_state = 'FINISHED') AS "scanFinished",
               CASE WHEN c.verdict_note = 'ADMIN_HIDE' THEN 'ADMIN' ELSE 'AUTO' END AS "heldBy",
               EXISTS (SELECT 1 FROM relevance_appeals a WHERE a.case_id = c.id AND a.status = 'WON') AS "appealWon",
               (u.relevance_consent_withdrawn_at IS NOT NULL) AS "withdrawn"
        FROM relevance_cases c JOIN batch_relevance br ON br.id = c.batch_id
        JOIN users u ON u.id = br.photographer_id
        WHERE c.stage = 'APPLIED' AND c.applied_mode = 'AUTO_HIDE'
          AND (c.action IN ('HIDE_ALL', 'HIDE_JUNK') OR EXISTS (
              SELECT 1 FROM flag_hidden_photos h WHERE h.batch_id = c.batch_id AND h.held_by = 'AUTO' AND h.restored_at IS NULL))
        ORDER BY c.applied_at, c.id
    """, nativeQuery = true)
    fun findHoldTargets(): List<HoldTarget>

    // RV-2.3a (A.4, S3, P14, P18): current auto-hide cases whose full scan is queued or running, least recently served first.
    // Only batches still allowed (invariant 12, the conditions of BatchRelevanceRepository.stillAllowed), so a batch that
    // is not allowed takes no turn slot; its case stays as it is (G9) and the scan goes on once it is allowed again. RV-L7
    // part a: a photographer who withdrew the consent to automatic actions is scanned only for a HIDE_JUNK case (Mixed, or
    // a whole-batch case whose appeal was won), whose scan can only put photos back: nothing more is hidden for them.
    @Query(value = """
        SELECT c.id AS "caseId", c.batch_id AS "batchId", br.event_id AS "eventId", br.photographer_id AS "photographerId"
        FROM relevance_cases c JOIN batch_relevance br ON br.id = c.batch_id
        JOIN events e ON e.id = br.event_id AND e.deleted_at IS NULL AND e.hidden_at IS NULL AND e.visibility = 'PUBLIC'
        JOIN users u ON u.id = br.photographer_id AND u.suspended_at IS NULL
            AND (u.relevance_consent_withdrawn_at IS NULL OR c.action = 'HIDE_JUNK')
        WHERE c.stage = 'APPLIED' AND c.applied_mode = 'AUTO_HIDE' AND c.scan_state IN ('QUEUED', 'RUNNING')
        ORDER BY c.scan_served_at ASC NULLS FIRST, c.applied_at, c.id
        LIMIT :limit
    """, nativeQuery = true)
    fun findScans(@Param("limit") limit: Int): List<SamplingCase>

    // RV-2.3a: one turn starts. Read again under the batch claim: still the current auto-hide case with its scan queued
    // or running. A queued scan (also a re-queued finished one) becomes RUNNING; the turn is stamped for the rotation.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET scan_state = 'RUNNING', scan_served_at = :now
        WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE' AND scan_state IN ('QUEUED', 'RUNNING')
    """, nativeQuery = true)
    fun startScanTurn(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    // RV-2.3a: nothing left to measure. Only from RUNNING, so a scan queued again meanwhile keeps QUEUED and runs again;
    // the counts are the batch's whole aggregate and replace the old ones, so a second finish never double counts.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET scan_state = 'FINISHED', scan_finished_at = :now,
            scan_measured = :measured, scan_junk = :junk, scan_blurry = :blurry
        WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE' AND scan_state = 'RUNNING'
    """, nativeQuery = true)
    fun finishScan(
        @Param("id") id: UUID,
        @Param("measured") measured: Int,
        @Param("junk") junk: Int,
        @Param("blurry") blurry: Int,
        @Param("now") now: OffsetDateTime,
    ): Int

    // RV-2.2b, RV-2.3b: still the current auto-hide case with this target? Read before each chunk; a plain read, no lock.
    @Query(value = """
        SELECT EXISTS (SELECT 1 FROM relevance_cases
                       WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE' AND action = :action)
    """, nativeQuery = true)
    fun isHoldTarget(@Param("id") id: UUID, @Param("action") action: String): Boolean

    // RV-L7 part a: isHoldTarget for a chunk that hides: also no automatic hide for a photographer who withdrew the consent
    // to automatic actions, read before each chunk (a withdrawal after the targets were read stops the next chunk). An
    // admin's own hide (ADMIN_HIDE) still goes ahead.
    @Query(value = """
        SELECT EXISTS (SELECT 1 FROM relevance_cases c JOIN batch_relevance br ON br.id = c.batch_id
                       JOIN users u ON u.id = br.photographer_id
                       WHERE c.id = :id AND c.stage = 'APPLIED' AND c.applied_mode = 'AUTO_HIDE' AND c.action = :action
                         AND (c.verdict_note = 'ADMIN_HIDE' OR u.relevance_consent_withdrawn_at IS NULL))
    """, nativeQuery = true)
    fun isHideTarget(@Param("id") id: UUID, @Param("action") action: String): Boolean

    // RV-2.4 (A.4, G5): the full scan has the last word on the 10% line, once. Right after finishScan, with the counts in
    // hand: a Fits case at or over the line becomes Mixed (HIDE_JUNK); a Mixed case under it becomes Fits (NONE), and the
    // hold step then restores its automatic holds. Never with an open appeal; a Mixed case over 70% stays Mixed.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET flipped_at = :now, verdict_note = 'SCAN_FLIP',
            verdict = CASE WHEN c.verdict = 'FITS' THEN 'MIXED' ELSE 'FITS' END,
            action = CASE WHEN c.verdict = 'FITS' THEN 'HIDE_JUNK' ELSE 'NONE' END
        FROM batch_relevance br
        WHERE c.id = :id AND br.id = c.batch_id AND c.stage = 'APPLIED' AND c.applied_mode = 'AUTO_HIDE'
          AND c.flipped_at IS NULL AND $NO_OPEN_APPEAL
          AND ((c.verdict = 'FITS' AND :measured > 0 AND :junk * 100 >= :fitsBelow * :measured)
               OR (c.verdict = 'MIXED' AND :junk * 100 < :fitsBelow * :measured))
    """, nativeQuery = true)
    fun flipAfterScan(
        @Param("id") id: UUID,
        @Param("measured") measured: Int,
        @Param("junk") junk: Int,
        @Param("fitsBelow") fitsBelow: Int,
        @Param("now") now: OffsetDateTime,
    ): Int

    // RV-2.2b: the hide found nothing left, once per case. :photos is the batch's live automatic holds.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET hide_finished_at = :now, hidden_photos = :photos
        WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE' AND hide_finished_at IS NULL
    """, nativeQuery = true)
    fun finishHide(@Param("id") id: UUID, @Param("photos") photos: Int, @Param("now") now: OffsetDateTime): Int

    // RV-3.2 part c (A.6): a won case closes APPEAL_WON once its scan has finished and its good photos are back
    // (reconcileHolds checks that). It never counts; its junk keeps its holds with no current case.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET stage = 'CLOSED', closed_at = :now, close_reason = 'APPEAL_WON'
        WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE' AND action = 'HIDE_JUNK' AND scan_state = 'FINISHED'
          AND EXISTS (SELECT 1 FROM relevance_appeals a WHERE a.case_id = relevance_cases.id AND a.status = 'WON')
    """, nativeQuery = true)
    fun closeWonAppeal(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    // RV-2.5a (G10): an admin closes the current case; closed_by names the admin.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET stage = 'CLOSED', closed_at = :now, close_reason = :reason, closed_by = :adminId
        WHERE id = :id AND stage = 'APPLIED'
    """, nativeQuery = true)
    fun closeByAdmin(
        @Param("id") id: UUID,
        @Param("reason") reason: String,
        @Param("adminId") adminId: UUID,
        @Param("now") now: OffsetDateTime,
    ): Int

    // RV-2.5b (G10): looks fine closes only a current Can't tell case that hides nothing, so a hide-batch that commits
    // after the read is never closed with its admin holds left behind.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET stage = 'CLOSED', closed_at = :now, close_reason = 'LOOKS_FINE', closed_by = :adminId
        WHERE id = :id AND stage = 'APPLIED' AND verdict = 'CANT_TELL' AND action = 'NONE'
    """, nativeQuery = true)
    fun closeLooksFine(@Param("id") id: UUID, @Param("adminId") adminId: UUID, @Param("now") now: OffsetDateTime): Int

    // RV-2.5b: a re-check closes the current case only while its action is still the one read, so an admin hide that
    // commits in between is never superseded (the next tick decides again and meets it).
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET stage = 'CLOSED', closed_at = :now, close_reason = :reason
        WHERE id = :id AND stage = 'APPLIED' AND action IS NOT DISTINCT FROM CAST(:action AS VARCHAR)
    """, nativeQuery = true)
    fun closeIfAction(
        @Param("id") id: UUID,
        @Param("action") action: String?,
        @Param("reason") reason: String,
        @Param("now") now: OffsetDateTime,
    ): Int

    // RV-3.1 (A.5): notes owed by auto-hide cases (P18), each with no message of its notice key yet. 'verdict': a current
    // whole-batch hide (an admin's hide of a Can't tell batch included) or a current Mixed case (a Fits case flipped to
    // Mixed too); 'blurry': a current Fits case that hides nothing (not one a re-check lifted to HIDE_JUNK: its blurry
    // photos are off sale) whose scan finished with blurry photos; 'restored': a current case flipped
    // from Mixed to Fits, or a case a re-check lifted.
    @Query(value = """
        SELECT c.id AS "caseId", br.photographer_id AS "photographerId", n.note AS "note", e.name AS "eventName",
               CAST(e.date AS VARCHAR) AS "eventDate", c.verdict AS "verdict", c.close_reason AS "closeReason",
               c.batch_photos AS "batchPhotos", c.measured AS "measured", c.junk AS "junk",
               c.scan_measured AS "scanMeasured", c.scan_blurry AS "scanBlurry", c.verdict_note AS "verdictNote"
        FROM relevance_cases c
        JOIN batch_relevance br ON br.id = c.batch_id
        JOIN events e ON e.id = br.event_id
        CROSS JOIN (VALUES ('verdict'), ('blurry'), ('restored')) AS n (note)
        WHERE c.applied_mode = 'AUTO_HIDE' AND (
              (n.note = 'verdict' AND c.stage = 'APPLIED' AND (c.action = 'HIDE_ALL' OR c.verdict = 'MIXED'))
           OR (n.note = 'blurry' AND c.stage = 'APPLIED' AND c.verdict = 'FITS' AND c.action = 'NONE' AND c.scan_state = 'FINISHED' AND c.scan_blurry > 0)
           OR (n.note = 'restored' AND ((c.stage = 'APPLIED' AND c.verdict = 'FITS' AND c.verdict_note = 'SCAN_FLIP')
                                        OR (c.stage = 'CLOSED' AND c.close_reason = 'RECHECK_LIFTED'))))
          AND NOT EXISTS (SELECT 1 FROM photographer_messages m WHERE m.notice_key = 'relevance:' || c.id || ':' || n.note)
        ORDER BY c.applied_at, c.id, n.note
        LIMIT :limit
    """, nativeQuery = true)
    fun findOwedNotes(@Param("limit") limit: Int): List<OwedNote>

    // RV-3.1 (A.5): a verdict note starts the 14-day appeal clock, once. A second server copy that read the same case
    // waits on the row, then changes nothing.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET notice_at = :now
        WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE' AND notice_at IS NULL
    """, nativeQuery = true)
    fun stampNotice(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    // RV-3.2a (A.6): one case of the photographer's own batch, with what the photographer may see of it: never the
    // model's reason, the summary sent or the words read from photos.
    @Query(value = """
        SELECT c.id AS "caseId", c.batch_id AS "batchId", br.event_id AS "eventId", e.name AS "eventName",
               c.verdict AS "verdict", c.action AS "action", c.stage AS "stage", c.applied_mode AS "appliedMode",
               c.batch_photos AS "batchPhotos", c.measured AS "measured", c.junk AS "junk", c.hidden_photos AS "hiddenPhotos",
               c.scan_measured AS "scanMeasured", c.scan_blurry AS "scanBlurry", c.notice_at AS "noticeAt",
               CAST(e.date AS VARCHAR) AS "eventDate", c.close_reason AS "closeReason", c.verdict_note AS "verdictNote"
        FROM relevance_cases c
        JOIN batch_relevance br ON br.id = c.batch_id
        JOIN events e ON e.id = br.event_id
        WHERE c.id = :caseId AND br.photographer_id = :photographerId
    """, nativeQuery = true)
    fun ownCase(@Param("caseId") caseId: UUID, @Param("photographerId") photographerId: UUID): OwnCaseRow?

    // RV-3.2 part b (A.6): a won appeal lowers the current auto-hide case's target. A whole-batch hide (an admin's hide of a
    // Can't tell batch included) becomes HIDE_JUNK with its full scan queued, so the good photos come back as they are
    // measured and the junk stays hidden; a junk-only hide (Mixed) becomes NONE.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET action = CASE WHEN action = 'HIDE_ALL' THEN 'HIDE_JUNK' ELSE 'NONE' END,
            scan_state = CASE WHEN action = 'HIDE_ALL' THEN 'QUEUED' ELSE scan_state END
        WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE' AND action IN ('HIDE_ALL', 'HIDE_JUNK')
    """, nativeQuery = true)
    fun winAppeal(@Param("id") id: UUID): Int

    // RV-3.2 part b (A.7): confirming stamps the case once. Only a current auto-hide whole-batch finding (junk dump, wrong
    // event, copied) is ever confirmed; an admin's hide of a Can't tell batch never is, nor a case whose appeal was won
    // or is open (RV-4.1: one filed after the confirm step listed the case).
    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET confirmed_at = :now
        WHERE c.id = :id AND c.stage = 'APPLIED' AND c.applied_mode = 'AUTO_HIDE' AND c.confirmed_at IS NULL
          AND c.verdict IN ('JUNK_DUMP', 'WRONG_EVENT', 'COPIED') AND c.verdict_note IS DISTINCT FROM 'ADMIN_HIDE'
          AND NOT EXISTS (SELECT 1 FROM relevance_appeals a WHERE a.case_id = c.id AND a.status IN ('WON', 'OPEN'))
    """, nativeQuery = true)
    fun stampConfirmed(@Param("id") id: UUID, @Param("now") now: OffsetDateTime): Int

    // RV-4.1 (A.7): current auto-hide whole-batch findings due to be confirmed: the appeal was lost, or no appeal came and
    // the last day to appeal has passed (notice_at before :cutoff). The event's visibility is not read (G9).
    @Query(value = """
        SELECT c.id FROM relevance_cases c
        WHERE c.stage = 'APPLIED' AND c.applied_mode = 'AUTO_HIDE' AND c.confirmed_at IS NULL AND c.notice_at IS NOT NULL
          AND c.verdict IN ('JUNK_DUMP', 'WRONG_EVENT', 'COPIED') AND c.verdict_note IS DISTINCT FROM 'ADMIN_HIDE'
          AND (EXISTS (SELECT 1 FROM relevance_appeals a WHERE a.case_id = c.id AND a.status = 'LOST')
               OR (c.notice_at < :cutoff AND NOT EXISTS (SELECT 1 FROM relevance_appeals a WHERE a.case_id = c.id)))
        ORDER BY c.notice_at, c.id LIMIT :limit
    """, nativeQuery = true)
    fun findConfirmDue(@Param("cutoff") cutoff: OffsetDateTime, @Param("limit") limit: Int): List<UUID>

    // RV-4.2 (A.8): the case names the flag that holds its payouts, once.
    @Modifying
    @Query(value = "UPDATE relevance_cases SET hold_flag_id = :flagId WHERE id = :id AND hold_flag_id IS NULL", nativeQuery = true)
    fun setHoldFlag(@Param("id") id: UUID, @Param("flagId") flagId: UUID): Int

    // RV-4.2: current auto-hide Copied cases whose hold was never placed (hold_flag_id is set once and never cleared, so an
    // ended hold is never placed again), oldest first. RV-L7 part a: never for a photographer who withdrew the consent.
    // RV-4.5a: never for an admin's hide (a no-consent case): only the admin's hold-payouts holds those payouts.
    @Query(value = """
        SELECT c.id FROM relevance_cases c JOIN batch_relevance br ON br.id = c.batch_id
        JOIN users u ON u.id = br.photographer_id AND u.relevance_consent_withdrawn_at IS NULL
        WHERE c.stage = 'APPLIED' AND c.applied_mode = 'AUTO_HIDE' AND c.verdict = 'COPIED' AND c.hold_flag_id IS NULL
          AND c.verdict_note IS DISTINCT FROM 'ADMIN_HIDE'
        ORDER BY c.applied_at, c.id LIMIT :limit
    """, nativeQuery = true)
    fun findHoldDue(@Param("limit") limit: Int): List<UUID>

    // RV-4.5a review: the replacing case takes the replaced case's hold flag while it still holds payouts, once.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases n SET hold_flag_id = o.hold_flag_id
        FROM relevance_cases o JOIN flags f ON f.id = o.hold_flag_id
        WHERE n.id = :toId AND o.id = :fromId AND n.hold_flag_id IS NULL
          AND f.status IN ('open', 'escalated') AND f.payout_hold_at IS NOT NULL
    """, nativeQuery = true)
    fun carryLiveHold(@Param("fromId") fromId: UUID, @Param("toId") toId: UUID): Int

    // RV-4.2: the case's hold flag while it still holds payouts (open or escalated, held), and when the hold started.
    @Query(value = "SELECT f.id $LIVE_HOLD", nativeQuery = true)
    fun liveHoldFlag(@Param("id") id: UUID): UUID?

    @Query(value = "SELECT f.payout_hold_at $LIVE_HOLD", nativeQuery = true)
    fun liveHoldAt(@Param("id") id: UUID): Instant?

    // RV-3.2a: filing an appeal locks its case first, so two filings of the same case run one after the other.
    @Query(value = "SELECT id FROM relevance_cases WHERE id = :id FOR UPDATE", nativeQuery = true)
    fun lockCase(@Param("id") id: UUID): UUID?

    // RV-2.5b (G10): an admin hides a current auto-hide Can't tell case's whole batch. The verdict stays Can't tell.
    // RV-4.5a: also a current case applied as record-only for want of consent (NO_CONSENT) whose verdict would have hidden
    // the whole batch; it becomes an auto-hide case, so the hold step, the note and the appeal act on it as on any admin
    // hide, and ADMIN_HIDE keeps it out of the ladder and the automatic payout hold.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases SET action = 'HIDE_ALL', verdict_note = 'ADMIN_HIDE', applied_mode = 'AUTO_HIDE'
        WHERE id = :id AND stage = 'APPLIED' AND (
              (applied_mode = 'AUTO_HIDE' AND verdict = 'CANT_TELL' AND action = 'NONE')
           OR (applied_mode = 'RECORD_ONLY' AND (no_consent OR verdict_note = 'NO_CONSENT')
               AND verdict IN ('JUNK_DUMP', 'WRONG_EVENT', 'COPIED')))
    """, nativeQuery = true)
    fun markAdminHide(@Param("id") id: UUID): Int

    // RV-2.5a: batches an admin restored since :since that have no current case and still a live automatic or admin hold
    // (a hide chunk that outran the restore, or a restore request that broke off). Read from the cases, proved per batch
    // by EXISTS.
    @Query(value = """
        SELECT DISTINCT c.batch_id FROM relevance_cases c
        WHERE c.stage = 'CLOSED' AND c.close_reason = 'RESTORED_BY_ADMIN' AND c.closed_at > :since
          AND NOT EXISTS (SELECT 1 FROM relevance_cases n WHERE n.batch_id = c.batch_id AND n.stage = 'APPLIED')
          AND EXISTS (SELECT 1 FROM flag_hidden_photos h
                      WHERE h.batch_id = c.batch_id AND h.held_by IN ('AUTO', 'ADMIN') AND h.restored_at IS NULL)
    """, nativeQuery = true)
    fun findAdminRestoresLeft(@Param("since") since: OffsetDateTime): List<UUID>

    // RV-2.2a (P4): a case auto-hide could not act on because the photographer has not accepted the minimum Terms.
    // RVF-9: its own mark, so the case keeps its note.
    @Modifying
    @Query(value = "UPDATE relevance_cases SET no_consent = TRUE WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'RECORD_ONLY'", nativeQuery = true)
    fun markNoConsent(@Param("id") id: UUID): Int

    // RV-2.2a (A.3 step 5, P8): queue the full scan. Only an auto-hide case is ever scanned (P14, P18).
    @Modifying
    @Query(value = "UPDATE relevance_cases SET scan_state = 'QUEUED' WHERE id = :id AND stage = 'APPLIED' AND applied_mode = 'AUTO_HIDE'", nativeQuery = true)
    fun queueScan(@Param("id") id: UUID): Int

    // The current case takes a newer check's copy and date results, so the RULES trigger does not fire again for them.
    @Modifying
    @Query(value = "UPDATE relevance_cases SET copy_low = :copyLow, date_low = :dateLow WHERE id = :id AND stage = 'APPLIED'", nativeQuery = true)
    fun copyRuleResults(@Param("id") id: UUID, @Param("copyLow") copyLow: Boolean?, @Param("dateLow") dateLow: Boolean?): Int

    // RVF-5 (RVR-1): a current case left with the band UNKNOWN takes a same-verdict retry's band and counts, so the
    // admin sees them and the batch is not due again.
    @Modifying
    @Query(value = """
        UPDATE relevance_cases c SET sample_size = r.sample_size, measured = r.measured, blurry = r.blurry, dark = r.dark,
            not_race = r.not_race, junk = r.junk, band = r.band
        FROM relevance_cases r
        WHERE c.id = :id AND c.stage = 'APPLIED' AND c.band = 'UNKNOWN' AND r.id = :fromCaseId
    """, nativeQuery = true)
    fun takeBand(@Param("id") id: UUID, @Param("fromCaseId") fromCaseId: UUID): Int

    // RV-1.10 (plan 5.11): one tab of the admin page, current cases only, newest first. :tab is 'hidden' (an action
    // other than NONE), 'cant_tell' or 'all'; the service checks it. RV-4.1: :repeat keeps only "Repeat cases", the
    // confirmed cases of a photographer with two or more strikes confirmed after :since.
    @Query(value = "SELECT $CASE_ROW $CASE_FROM WHERE c.stage = 'APPLIED' AND $CASE_TAB AND $CASE_REPEAT ORDER BY c.applied_at DESC, c.id",
        countQuery = "SELECT count(*) $CASE_FROM WHERE c.stage = 'APPLIED' AND $CASE_TAB AND $CASE_REPEAT", nativeQuery = true)
    fun pageCurrentCases(
        @Param("tab") tab: String,
        @Param("repeat") repeat: Boolean,
        @Param("since") since: OffsetDateTime,
        pageable: Pageable,
    ): Page<RelevanceCaseRow>

    // AX-8: the admin badge. Current cases a person must look at, each case once: Can't tell with no admin hide yet
    // (an admin hide keeps the verdict, markAdminHide), or an open appeal.
    @Query(value = "SELECT count(*) $CASE_FROM WHERE c.stage = 'APPLIED' AND (($CANT_TELL AND c.action = 'NONE') OR $OPEN_APPEAL)",
        nativeQuery = true)
    fun countNeedingPerson(): Long

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
    c.junk AS "junk", c.decided_at AS "decidedAt", c.no_consent AS "noConsent""""

private const val CASE_FROM = """FROM relevance_cases c
    JOIN batch_relevance br ON br.id = c.batch_id
    JOIN events e ON e.id = br.event_id
    JOIN users u ON u.id = br.photographer_id
    LEFT JOIN photographer_settings ps ON ps.user_id = br.photographer_id"""

private const val CANT_TELL = "c.verdict = 'CANT_TELL'"

// RV-3.2 part b: current cases with an open appeal.
private const val OPEN_APPEAL = "EXISTS (SELECT 1 FROM relevance_appeals a WHERE a.case_id = c.id AND a.status = 'OPEN')"

private const val CASE_TAB =
    "(:tab = 'all' OR (:tab = 'hidden' AND c.action <> 'NONE') OR (:tab = 'cant_tell' AND $CANT_TELL) " +
        "OR (:tab = 'appeals' AND $OPEN_APPEAL))"

private const val LIVE_HOLD = "FROM relevance_cases c JOIN flags f ON f.id = c.hold_flag_id " +
    "WHERE c.id = :id AND f.status IN ('open', 'escalated') AND f.payout_hold_at IS NOT NULL"

private const val CASE_REPEAT = "(:repeat = false OR (c.confirmed_at IS NOT NULL AND (SELECT count(*) FROM relevance_strikes s " +
    "WHERE s.photographer_id = br.photographer_id AND s.confirmed_at > :since AND s.voided_at IS NULL) >= 2))"

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
    val noConsent: Boolean
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

// RV-3.1: one note a case owes (findOwedNotes). note is 'verdict', 'blurry' or 'restored'; eventDate is ISO.
interface OwedNote {
    val caseId: UUID
    val photographerId: UUID
    val note: String
    val eventName: String
    val eventDate: String
    val verdict: String?
    val closeReason: String?
    val batchPhotos: Int
    val measured: Int?
    val junk: Int?
    val scanMeasured: Int?
    val scanBlurry: Int?
    // RV-4.5a: ADMIN_HIDE on a Junk dump, Wrong event or Copied case is an admin's hide of a no-consent case (V2-T24).
    val verdictNote: String?
}

// RV-3.2a: a photographer's own case (ownCase).
interface OwnCaseRow {
    val caseId: UUID
    val batchId: UUID
    val eventId: UUID
    val eventName: String
    val verdict: String?
    val action: String?
    val stage: String
    val appliedMode: String?
    val batchPhotos: Int
    val measured: Int?
    val junk: Int?
    val hiddenPhotos: Int
    val scanMeasured: Int?
    val scanBlurry: Int?
    val noticeAt: Instant?
    val eventDate: String?
    val closeReason: String?
    // RV-4.5a: ADMIN_HIDE marks an admin's hide, which never counts.
    val verdictNote: String?
}

interface SamplingCase {
    val caseId: UUID
    val batchId: UUID
    val eventId: UUID
    val photographerId: UUID
}

// RV-2.2b: one current auto-hide case whose holds follow its target (findHoldTargets).
interface HoldTarget {
    val caseId: UUID
    val batchId: UUID
    val eventId: UUID
    val photographerId: UUID
    val hideFinished: Boolean
    val action: String
    val scanFinished: Boolean
    // RV-2.5b: the hold kind the hide places: ADMIN for an admin's hide of a Can't tell batch, else AUTO.
    val heldBy: String
    // RV-3.2 part c: the case's appeal was won, so it closes APPEAL_WON once the scan and the restore are done.
    val appealWon: Boolean
    // RV-L7 part a: the photographer withdrew the consent to automatic actions.
    val withdrawn: Boolean
}

// RV-2.4 (P8, A.4): the batch aliased br has no open appeal. One predicate for the due query and the scan flip.
const val NO_OPEN_APPEAL = "NOT EXISTS (SELECT 1 FROM relevance_appeals a WHERE a.batch_id = br.id AND a.status = 'OPEN')"
