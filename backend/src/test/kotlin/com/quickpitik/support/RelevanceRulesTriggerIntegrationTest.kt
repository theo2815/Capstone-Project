package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.CheckDue
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-1.8b (plan 5.8, P9): a batch is due again when version 1's copy or date result moved away from the one its
// current case was decided with, and that re-check reuses the current case's sample. The container is shared, so
// fixtures sit in 2004 and assertions look only at this class's own rows.
class RelevanceRulesTriggerIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `a changed copy result makes the batch due`() {
        val b = batch()
        current(b, copyLow = false)
        assertNull(dueRow(b), "checked, scored, and the current case has the same results")

        jdbc.update("UPDATE batch_relevance SET copy_low = true WHERE id = ?", b.id)
        assertEquals("RULES", dueRow(b)?.trigger)

        jdbc.update("UPDATE batch_relevance SET copy_low = false, date_low = true WHERE id = ?", b.id)
        assertEquals("RULES", dueRow(b)?.trigger, "a changed date result too")

        jdbc.update("UPDATE batch_relevance SET scored_upload_at = ? WHERE id = ?", UPLOAD.minusMinutes(5), b.id)
        assertNull(dueRow(b), "version 1 has not scored the newest upload yet")

        jdbc.update("UPDATE batch_relevance SET scored_upload_at = ? WHERE id = ?", UPLOAD, b.id)
        jdbc.update("UPDATE relevance_cases SET date_low = true WHERE batch_id = ?", b.id)
        assertNull(dueRow(b), "the same results again")

        jdbc.update("UPDATE batch_relevance SET copy_low = true WHERE id = ?", b.id)
        jdbc.update("UPDATE event_photographer SET last_upload_at = ? WHERE event_id = ?", UPLOAD.plusMinutes(1), b.event)
        jdbc.update("UPDATE batch_relevance SET scored_upload_at = ? WHERE id = ?", UPLOAD.plusMinutes(1), b.id)
        assertEquals("QUIET", dueRow(b)?.trigger, "a new upload is checked with a new sample")
    }

    @Test
    fun `a rules re-check copies the old sample and measures nothing`() {
        val b = batch(photos = 50)
        val photos = List(50) { photo(b) }
        // The current case sampled 40 photos; 28 are blurry, so the share sits on the 70 line.
        val old = current(b, copyLow = false)
        photos.take(40).forEachIndexed { i, p ->
            jdbc.update("INSERT INTO relevance_case_photos (case_id, photo_id, round) VALUES (?, ?, 1)", old, p)
            measured(b, p, fgSharpness = if (i < 28) 1.0 else 80.0)
        }
        jdbc.update("UPDATE batch_relevance SET copy_low = true WHERE id = ?", b.id)
        val measurer = Mockito.mock(RelevanceMeasurer::class.java)
        val service = service(measurer)

        assertTrue(service.start(assertNotNull(dueRow(b)), QUIET_CUTOFF, NOW))

        val check = jdbc.queryForMap("SELECT id, trigger, sample_round FROM relevance_cases WHERE batch_id = ? AND stage = 'SAMPLING'", b.id)
        assertEquals("RULES", check["trigger"])
        val caseId = check["id"] as UUID
        assertEquals(sample(old), sample(caseId), "the current case's sample, rounds included")

        val row = cases.findSampling(NOW, 10_000).single { it.batchId == b.id }
        service.continueCase(row, NOW)

        Mockito.verifyNoInteractions(measurer)
        val decided = jdbc.queryForMap("SELECT stage, sample_round, measured, junk FROM relevance_cases WHERE id = ?", caseId)
        assertEquals(listOf("DECIDING", 1, 40, 28), listOf("stage", "sample_round", "measured", "junk").map { (decided[it] as? Number)?.toInt() ?: decided[it] })
        assertEquals(40, cases.countSample(caseId), "no second round on the 70 line")
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun service(measurer: RelevanceMeasurer): RelevanceCheckService {
        val props = RelevanceProperties(mode = RelevanceMode.RECORD_ONLY, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        val switch = RelevanceSwitch(props, AdminProperties(), AiApiProperties(enabled = true))
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(), measurer,
            Mockito.mock(AnthropicClient::class.java), events)
    }

    // A shared race's batch, quiet, already checked for its newest upload and scored by version 1 for it.
    private fun batch(photos: Int = 40): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "relrules-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2004-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relrules-${UUID.randomUUID()}",
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

    private fun current(b: Batch, copyLow: Boolean): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, copy_low, verdict, applied_mode, action) " +
            "VALUES (?, 'QUIET', ?, 40, 'APPLIED', ?, 'FITS', 'RECORD_ONLY', 'NONE')",
        b.id, UPLOAD, copyLow,
    )

    private fun photo(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, 125, 'LIVE')",
        b.event, b.photographer, "relrules/${UUID.randomUUID()}",
    )

    private fun measured(b: Batch, photo: UUID, fgSharpness: Double) = jdbc.update(
        "INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, fg_brightness, sharpness, brightness, " +
            "person_conf, race_conf) VALUES (?, ?, ?, ?, 60, 70, 50, 0.99, 0.9)",
        photo, b.id, NOW, fgSharpness,
    )

    private fun sample(caseId: UUID): Map<UUID, Int> =
        jdbc.queryForList("SELECT photo_id, round FROM relevance_case_photos WHERE case_id = ?", caseId)
            .associate { it["photo_id"] as UUID to (it["round"] as Number).toInt() }

    private fun dueRow(b: Batch): CheckDue? =
        batches.findCheckDue(QUIET_CUTOFF, NOW, 500, 10_000).singleOrNull { it.eventId == b.event && it.photographerId == b.photographer }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        // Non-zero microseconds: the rules compare scored_upload_at and checked_upload_at with last_upload_at for equality.
        val NOW: OffsetDateTime = OffsetDateTime.parse("2004-03-04T05:06:07.123456Z")
        val QUIET_CUTOFF: OffsetDateTime = NOW.minusMinutes(10)
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
    }
}
