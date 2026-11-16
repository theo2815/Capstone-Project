package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoToMeasure
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RVF-4 (RVR-11): a batch whose latest check gave no usable result (a failed model call, or a skip while its event
// was private or its photographer suspended) is checked again by itself, at least the wait apart and at most
// max-retries times for one upload. The container is shared, so fixtures sit in 2013 and assertions look only at
// this class's own rows.
class RelevanceRetryIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `a batch whose check failed is due again after the wait, as a retry`() {
        val failedFirst = batch()
        case(failedFirst, stage = "APPLIED", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT)
        assertEquals("RETRY", dueRow(failedFirst)?.trigger, "a first check whose call failed became the current case")

        val failedRecheck = batch()
        case(failedRecheck, stage = "APPLIED", verdict = "FITS", endedAt = PAST_WAIT.minusHours(1))
        case(failedRecheck, stage = "CLOSED", verdict = "CANT_TELL", note = "BAD_JSON", reason = "NO_RESULT", endedAt = PAST_WAIT)
        assertEquals("RETRY", dueRow(failedRecheck)?.trigger, "a re-check whose call failed was closed with no result")
    }

    @Test
    fun `a batch skipped while its event was private is due again once it is public`() {
        val b = batch()
        case(b, stage = "CLOSED", reason = "SKIPPED", endedAt = PAST_WAIT)
        jdbc.update("UPDATE events SET visibility = 'UNLISTED' WHERE id = ?", b.event)
        assertNull(dueRow(b), "still private")

        jdbc.update("UPDATE events SET visibility = 'PUBLIC' WHERE id = ?", b.event)
        assertEquals("RETRY", dueRow(b)?.trigger)
    }

    @Test
    fun `a skipped batch with no current case is started as a retry`() {
        val b = batch()
        val photos = List(10) { photo(b) }
        val skipped = case(b, stage = "CLOSED", reason = "SKIPPED", endedAt = PAST_WAIT)
        photos.forEach { sampled(skipped, it) }
        assertNull(cases.currentFor(b.id))

        assertTrue(service().start(assertNotNull(dueRow(b)), QUIET_CUTOFF, NOW))

        val check = jdbc.queryForMap("SELECT id, trigger FROM relevance_cases WHERE batch_id = ? AND stage = 'SAMPLING'", b.id)
        assertEquals("RETRY", check["trigger"])
        assertEquals(sample(skipped), sample(check["id"] as UUID), "the skipped check's sample")
    }

    @Test
    fun `a retry whose earlier check has no sample picks a fresh one`() {
        val b = batch(photos = 10)
        val photos = List(10) { photo(b) }
        case(b, stage = "CLOSED", reason = "SKIPPED", endedAt = PAST_WAIT)

        assertTrue(service().start(assertNotNull(dueRow(b)), QUIET_CUTOFF, NOW))

        val retry = jdbc.queryForObject("SELECT id FROM relevance_cases WHERE batch_id = ? AND stage = 'SAMPLING'", UUID::class.java, b.id)!!
        assertEquals(photos.associateWith { 1 }, sample(retry), "the whole small batch, in round one")
    }

    @Test
    fun `a retry copies the earlier sample and measures no photo that was already measured`() {
        val b = batch(photos = 50)
        val photos = List(50) { photo(b) }
        val failed = case(b, stage = "APPLIED", verdict = "CANT_TELL", note = "CUT_OFF", endedAt = PAST_WAIT)
        photos.take(40).forEach { sampled(failed, it) }
        photos.take(30).forEach { measured(b, it, fgSharpness = 80.0) }
        val measurer = Mockito.mock(RelevanceMeasurer::class.java)
        val service = service(measurer)

        assertTrue(service.start(assertNotNull(dueRow(b)), QUIET_CUTOFF, NOW))
        val retry = jdbc.queryForObject("SELECT id FROM relevance_cases WHERE batch_id = ? AND stage = 'SAMPLING'", UUID::class.java, b.id)!!
        assertEquals(sample(failed), sample(retry))
        service.continueCase(cases.findSampling(NOW, 10_000).single { it.batchId == b.id }, NOW)

        @Suppress("UNCHECKED_CAST")
        val asked = ArgumentCaptor.forClass(List::class.java) as ArgumentCaptor<List<PhotoToMeasure>>
        Mockito.verify(measurer).measure(eqArg(b.id), anyArg(), asked.capture() ?: emptyList())
        assertEquals(photos.subList(30, 40).sorted(), asked.value.map { it.photoId }.sorted(), "only the ten not yet measured")
    }

    @Test
    fun `a retry takes no second round`() {
        val b = batch(photos = 50)
        val photos = List(50) { photo(b) }
        // The failed check sampled 40 photos; 28 are blurry, so the share sits on the 70 line.
        val failed = case(b, stage = "APPLIED", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT)
        photos.take(40).forEachIndexed { i, p ->
            sampled(failed, p)
            measured(b, p, fgSharpness = if (i < 28) 1.0 else 80.0)
        }
        val measurer = Mockito.mock(RelevanceMeasurer::class.java)
        val service = service(measurer)

        assertTrue(service.start(assertNotNull(dueRow(b)), QUIET_CUTOFF, NOW))
        val retry = jdbc.queryForObject("SELECT id FROM relevance_cases WHERE batch_id = ? AND stage = 'SAMPLING'", UUID::class.java, b.id)!!
        service.continueCase(cases.findSampling(NOW, 10_000).single { it.batchId == b.id }, NOW)

        Mockito.verifyNoInteractions(measurer)
        val decided = jdbc.queryForMap("SELECT stage, sample_round, measured, junk FROM relevance_cases WHERE id = ?", retry)
        assertEquals(listOf("DECIDING", 1, 40, 28), listOf("stage", "sample_round", "measured", "junk").map { (decided[it] as? Number)?.toInt() ?: decided[it] })
        assertEquals(40, cases.countSample(retry), "no second round on the 70 line")
    }

    @Test
    fun `a retry is not due before the wait has passed`() {
        val b = batch()
        case(b, stage = "CLOSED", reason = "NO_RESULT", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = NOW.minusMinutes(59))
        assertNull(dueRow(b))
    }

    @Test
    fun `a batch is retried at most max-retries times for one upload`() {
        val b = batch()
        case(b, stage = "APPLIED", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT.minusHours(3))
        // A retry of an earlier upload does not count against this one.
        case(b, stage = "CLOSED", trigger = "RETRY", reason = "NO_RESULT", verdict = "CANT_TELL", note = "CALL_FAILED",
            endedAt = PAST_WAIT.minusHours(3), upload = UPLOAD.minusDays(1))
        case(b, stage = "CLOSED", trigger = "RETRY", reason = "NO_RESULT", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT.minusHours(2))
        case(b, stage = "CLOSED", trigger = "RETRY", reason = "NO_RESULT", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT.minusHours(1))
        assertEquals("RETRY", dueRow(b)?.trigger, "two retries so far")

        case(b, stage = "CLOSED", trigger = "RETRY", reason = "NO_RESULT", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT)
        assertNull(dueRow(b), "three retries is the cap")
    }

    @Test
    fun `a newer upload is due as a normal check, not a retry`() {
        val b = batch()
        case(b, stage = "CLOSED", reason = "SKIPPED", endedAt = PAST_WAIT)
        jdbc.update("UPDATE event_photographer SET last_upload_at = ? WHERE event_id = ?", UPLOAD.plusMinutes(1), b.event)
        assertEquals("QUIET", dueRow(b)?.trigger)
    }

    @Test
    fun `a batch due by the rules trigger keeps that label`() {
        val b = batch()
        case(b, stage = "APPLIED", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT)
        jdbc.update("UPDATE batch_relevance SET copy_low = true WHERE id = ?", b.id)
        assertEquals("RULES", dueRow(b)?.trigger)
    }

    @Test
    fun `a seeded case is never retried`() {
        val b = batch()
        case(b, stage = "APPLIED", trigger = "SEEDED", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT)
        assertNull(dueRow(b))
    }

    @Test
    fun `max-retries 0 turns retries off`() {
        val b = batch()
        case(b, stage = "CLOSED", reason = "SKIPPED", endedAt = PAST_WAIT)
        assertEquals("RETRY", dueRow(b)?.trigger)
        assertNull(dueRow(b, maxRetries = 0))
    }

    @Test
    fun `with the mode off the tick starts no retry`() {
        val b = batch()
        case(b, stage = "CLOSED", reason = "SKIPPED", endedAt = PAST_WAIT)
        assertEquals("RETRY", dueRow(b)?.trigger)

        service(mode = RelevanceMode.OFF).tick()

        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM relevance_cases WHERE batch_id = ?", Int::class.java, b.id))
    }

    @Test
    fun `in record-only a retried batch leaves every photo's status, hold row, flag and message as it was`() {
        val b = batch(photos = 600)
        repeat(5) { photo(b) }
        case(b, stage = "APPLIED", verdict = "CANT_TELL", note = "CALL_FAILED", endedAt = PAST_WAIT)
        val retry = case(b, stage = "DECIDED", trigger = "RETRY", verdict = "COPIED", endedAt = null)
        val before = untouched()

        service().applyCase(retry, NOW)

        assertEquals(listOf("APPLIED", "RECORD_ONLY"), jdbc.queryForMap("SELECT stage, applied_mode FROM relevance_cases WHERE id = ?", retry).values.toList())
        assertEquals(before, untouched())
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun service(
        measurer: RelevanceMeasurer = Mockito.mock(RelevanceMeasurer::class.java),
        mode: RelevanceMode = RelevanceMode.RECORD_ONLY,
    ): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        val switch = RelevanceSwitch(props, AdminProperties(), AiApiProperties(enabled = true))
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(), measurer,
            Mockito.mock(AnthropicClient::class.java), events, alerts, Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.5"),
            Mockito.mock(com.quickpitik.repository.RelevanceHoldRepository::class.java),
            Mockito.mock(com.quickpitik.repository.PhotoRepository::class.java),
            Mockito.mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java),
            org.springframework.core.task.SyncTaskExecutor(),
            Mockito.mock(com.quickpitik.service.relevance.RelevanceNotifier::class.java),
            Mockito.mock(com.quickpitik.service.relevance.RelevanceConfirmer::class.java),
            Mockito.mock(com.quickpitik.service.relevance.RelevanceHoldService::class.java))
    }

    // A public shared race's batch, quiet, already checked and scored by version 1 for its newest upload.
    private fun batch(photos: Int = 40): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "relretry-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2013-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relretry-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?, ?)",
            event, photographer, photos, UPLOAD, UPLOAD,
        )
        val batchId = id(
            "INSERT INTO batch_relevance (event_id, photographer_id, scored_at, scored_upload_at, copy_low, checked_upload_at, checked_photos) " +
                "VALUES (?, ?, ?, ?, false, ?, ?)",
            event, photographer, NOW, UPLOAD, UPLOAD, photos,
        )
        return Batch(batchId, event, photographer)
    }

    // One case of the batch. An APPLIED case ended at its applied_at, a CLOSED one at its closed_at; each later call
    // makes a later case (created_at follows the end), so the last one written is the batch's latest.
    private fun case(
        b: Batch,
        stage: String,
        verdict: String? = null,
        note: String? = null,
        reason: String? = null,
        trigger: String = "QUIET",
        endedAt: OffsetDateTime?,
        upload: OffsetDateTime = UPLOAD,
    ): UUID {
        val applied = stage == "APPLIED"
        return id(
            "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, verdict_note, copy_low, " +
                "decided_at, applied_mode, action, applied_at, closed_at, close_reason, created_at) " +
                "VALUES (?, ?, ?, 40, ?, ?, ?, false, ?, ?, ?, ?, ?, ?, ?)",
            b.id, trigger, upload, stage, verdict, note, verdict?.let { endedAt ?: NOW },
            if (applied) "RECORD_ONLY" else null, if (applied) (if (verdict == "FITS" || verdict == "CANT_TELL") "NONE" else "HIDE_ALL") else null,
            if (applied) endedAt else null, if (stage == "CLOSED") endedAt else null, reason,
            (endedAt ?: NOW).minusMinutes(5),
        )
    }

    private fun photo(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, 125, 'LIVE')",
        b.event, b.photographer, "relretry/${UUID.randomUUID()}",
    )

    private fun sampled(caseId: UUID, photo: UUID) =
        jdbc.update("INSERT INTO relevance_case_photos (case_id, photo_id, round) VALUES (?, ?, 1)", caseId, photo)

    private fun measured(b: Batch, photo: UUID, fgSharpness: Double) = jdbc.update(
        "INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, fg_brightness, sharpness, brightness, " +
            "person_conf, race_conf) VALUES (?, ?, ?, ?, 60, 70, 50, 0.99, 0.9)",
        photo, b.id, NOW, fgSharpness,
    )

    private fun sample(caseId: UUID): Map<UUID, Int> =
        jdbc.queryForList("SELECT photo_id, round FROM relevance_case_photos WHERE case_id = ?", caseId)
            .associate { it["photo_id"] as UUID to (it["round"] as Number).toInt() }

    private fun dueRow(b: Batch, maxRetries: Int = 3): CheckDue? =
        batches.findCheckDue(QUIET_CUTOFF, NOW, 500, RETRY_CUTOFF, maxRetries, 10_000)
            .singleOrNull { it.eventId == b.event && it.photographerId == b.photographer }

    // Every table record-only must leave alone, counted across the whole database.
    private fun untouched(): List<Any?> = listOf(
        jdbc.queryForList("SELECT status, count(*) AS n FROM photos GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM flag_hidden_photos", Long::class.java),
        jdbc.queryForList("SELECT status, count(*) AS n FROM flags GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages", Long::class.java),
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private fun <T> eqArg(value: T): T = Mockito.eq(value)
    private fun <T> anyArg(): T = Mockito.any()

    private companion object {
        // Non-zero microseconds: the rules compare upload times for equality.
        val NOW: OffsetDateTime = OffsetDateTime.parse("2013-03-04T05:06:07.123456Z")
        val QUIET_CUTOFF: OffsetDateTime = NOW.minusMinutes(10)
        val RETRY_CUTOFF: OffsetDateTime = NOW.minusMinutes(60)
        val PAST_WAIT: OffsetDateTime = NOW.minusMinutes(61)
        val UPLOAD: OffsetDateTime = NOW.minusHours(6)
    }
}
