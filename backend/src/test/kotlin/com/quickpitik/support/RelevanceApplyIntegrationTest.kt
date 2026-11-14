package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

// RV-1.9 (plan 5.11): a decided case is applied in record-only. It becomes the batch's current case and stores the
// alerts auto-hide would have raised, and nothing else changes. The container is shared, so fixtures sit in 2005
// and assertions look only at this class's own rows.
class RelevanceApplyIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `a decided case becomes the current case and the old one is superseded`() {
        val b = batch(photos = 40)
        val old = case(b, "APPLIED", "FITS", applied = true)
        val new = case(b, "DECIDED", "JUNK_DUMP")

        service().applyCase(new, NOW)

        assertEquals(listOf("CLOSED", "SUPERSEDED"), row(old, "stage", "close_reason"))
        assertEquals(listOf("APPLIED", "RECORD_ONLY", "HIDE_ALL"), row(new, "stage", "applied_mode", "action"))
        assertEquals(new, cases.currentFor(b.id)?.id)
    }

    @Test
    fun `the same verdict again closes the new case and raises nothing`() {
        val b = batch(photos = 600)
        val old = case(b, "APPLIED", "JUNK_DUMP", applied = true)
        val new = case(b, "DECIDED", "JUNK_DUMP")

        service().applyCase(new, NOW)

        assertEquals(listOf("CLOSED", "SAME_VERDICT"), row(new, "stage", "close_reason"))
        assertEquals(old, cases.currentFor(b.id)?.id)
        assertEquals(emptyList(), alertsOf(b))
    }

    @Test
    fun `a whole-batch verdict over the alert size raises one big-hide alert marked would-have`() {
        val big = batch(photos = 600)
        val dump = case(big, "DECIDED", "JUNK_DUMP")
        // A share of the batch: 120 of 300 sampled photos are junk, so 240 of 600 would be hidden, under the size.
        val mixedSmall = batch(photos = 600)
        val small = case(mixedSmall, "DECIDED", "MIXED", measured = 300, junk = 120)
        // 300 of 1,500 would be hidden: 600, over the size.
        val mixedBig = batch(photos = 1500)
        val share = case(mixedBig, "DECIDED", "MIXED", measured = 300, junk = 120)
        val service = service()

        listOf(dump, small, share).forEach { service.applyCase(it, NOW) }

        assertEquals(listOf(listOf("BIG_HIDE", "big_hide:$dump", true, 600)), alertsOf(big))
        assertEquals(emptyList(), alertsOf(mixedSmall))
        assertEquals(listOf(listOf("BIG_HIDE", "big_hide:$share", true, 600)), alertsOf(mixedBig))
    }

    @Test
    fun `a copied verdict raises a copied alert at any size`() {
        val b = batch(photos = 40)
        val c = case(b, "DECIDED", "COPIED")

        service().applyCase(c, NOW)

        assertEquals(listOf(listOf("COPIED", "copied:$c", true, 40)), alertsOf(b))
    }

    @Test
    fun `applying twice raises each alert once`() {
        val b = batch(photos = 600)
        val c = case(b, "DECIDED", "COPIED")
        val service = service()
        service.applyCase(c, NOW)

        // A second server copy that read the case before the first one committed.
        jdbc.update("UPDATE relevance_cases SET stage = 'DECIDED', applied_mode = NULL, action = NULL, applied_at = NULL WHERE id = ?", c)
        service.applyCase(c, NOW)

        assertEquals(listOf("BIG_HIDE", "COPIED"), alertsOf(b).map { it[0] })
        assertEquals("APPLIED", row(c, "stage").single())
    }

    @Test
    fun `a failed check with a current case is closed with no result and changes nothing`() {
        val b = batch(photos = 600)
        val old = case(b, "APPLIED", "FITS", applied = true, copyLow = false)
        val failed = case(b, "DECIDED", "CANT_TELL", note = "CALL_FAILED", copyLow = true)

        service().applyCase(failed, NOW)

        assertEquals(listOf("CLOSED", "NO_RESULT"), row(failed, "stage", "close_reason"))
        assertEquals(listOf("APPLIED", "FITS", "NONE"), row(old, "stage", "verdict", "action"))
        assertEquals(emptyList(), alertsOf(b))
        // Owner call, 2026-11-14: the rule results move over too, so a failed re-check is not repeated every tick.
        assertEquals(true, row(old, "copy_low").single())
    }

    @Test
    fun `the same verdict again copies the newer rule results onto the current case`() {
        val b = batch(photos = 40)
        val old = case(b, "APPLIED", "FITS", applied = true, copyLow = false, dateLow = null)
        val new = case(b, "DECIDED", "FITS", copyLow = true, dateLow = true)

        service().applyCase(new, NOW)

        assertEquals(listOf(true, true), row(old, "copy_low", "date_low"))
    }

    @Test
    fun `a case whose photographer was suspended is closed skipped`() {
        val b = batch(photos = 600)
        val c = case(b, "DECIDED", "JUNK_DUMP")
        jdbc.update("UPDATE users SET suspended_at = ? WHERE id = ?", NOW, b.photographer)

        service().applyCase(c, NOW)

        assertEquals(listOf("CLOSED", "SKIPPED"), row(c, "stage", "close_reason"))
        assertNull(cases.currentFor(b.id))
        assertEquals(emptyList(), alertsOf(b))
    }

    @Test
    fun `auto-hide with no minimum Terms version set applies as record-only`() {
        val b = batch(photos = 600)
        val c = case(b, "DECIDED", "JUNK_DUMP")
        val service = service(mode = RelevanceMode.AUTO_HIDE, flagsEnabled = true)

        service.applyCase(c, NOW)

        assertEquals(listOf("APPLIED", "RECORD_ONLY", "HIDE_ALL"), row(c, "stage", "applied_mode", "action"))
        assertEquals(listOf(listOf("BIG_HIDE", "big_hide:$c", true, 600)), alertsOf(b))
    }

    @Test
    fun `a changed copy result re-checks once, not every tick`() {
        val b = batch(photos = 40)
        case(b, "APPLIED", "FITS", applied = true, copyLow = false)
        jdbc.update("UPDATE batch_relevance SET copy_low = true WHERE id = ?", b.id)
        assertEquals("RULES", dueTrigger(b))

        // The re-check reaches the same verdict.
        service().applyCase(case(b, "DECIDED", "FITS", trigger = "RULES", copyLow = true), NOW)
        assertNull(dueTrigger(b), "the current case now carries the new copy result")

        // A re-check whose call failed does not start again on the next tick either.
        jdbc.update("UPDATE batch_relevance SET copy_low = false WHERE id = ?", b.id)
        assertEquals("RULES", dueTrigger(b))
        service().applyCase(case(b, "DECIDED", "CANT_TELL", trigger = "RULES", note = "BAD_JSON", copyLow = false), NOW)
        assertNull(dueTrigger(b))
    }

    @Test
    fun `record-only leaves every photo's status, every hold row, every flag and every message as it was`() {
        val b = batch(photos = 600)
        repeat(5) { photo(b) }
        val c = case(b, "DECIDED", "COPIED")
        val before = untouched()

        service().applyCase(c, NOW)

        assertEquals("APPLIED", row(c, "stage").single())
        assertEquals(before, untouched())
    }

    @Test
    fun `it runs with AI disabled`() {
        val b = batch(photos = 600)
        val c = case(b, "DECIDED", "JUNK_DUMP")

        service(aiEnabled = false, maxBatchesPerTick = 10_000).tick()

        assertEquals("APPLIED", row(c, "stage").single())
        assertEquals(listOf(listOf("BIG_HIDE", "big_hide:$c", true, 600)), alertsOf(b))
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun service(
        mode: RelevanceMode = RelevanceMode.RECORD_ONLY,
        flagsEnabled: Boolean = false,
        aiEnabled: Boolean = true,
        maxBatchesPerTick: Int = 3,
    ): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0),
            check = RelevanceProperties.Check(maxBatchesPerTick = maxBatchesPerTick))
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = flagsEnabled), AiApiProperties(enabled = aiEnabled))
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.5"),
            Mockito.mock(com.quickpitik.repository.RelevanceHoldRepository::class.java),
            Mockito.mock(com.quickpitik.repository.PhotoRepository::class.java),
            Mockito.mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java))
    }

    // A public shared race's batch, quiet, checked and scored by version 1 for its newest upload.
    private fun batch(photos: Int): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "relapply-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2005-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relapply-${UUID.randomUUID()}",
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

    private fun case(
        b: Batch,
        stage: String,
        verdict: String,
        applied: Boolean = false,
        trigger: String = "QUIET",
        note: String? = null,
        measured: Int = 40,
        junk: Int = 0,
        copyLow: Boolean? = false,
        dateLow: Boolean? = null,
    ): UUID {
        val photos = jdbc.queryForObject("SELECT photo_count FROM event_photographer WHERE event_id = ?", Int::class.java, b.event)
        return id(
            "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, verdict_note, " +
                "copy_low, date_low, decided_at, applied_mode, action, applied_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            b.id, trigger, UPLOAD, photos, stage, measured, junk, verdict, note, copyLow, dateLow, NOW,
            if (applied) "RECORD_ONLY" else null, if (applied) action(verdict) else null, if (applied) NOW else null,
        )
    }

    private fun action(verdict: String) = when (verdict) {
        "MIXED" -> "HIDE_JUNK"
        "JUNK_DUMP", "WRONG_EVENT", "COPIED" -> "HIDE_ALL"
        else -> "NONE"
    }

    private fun photo(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, 125, 'LIVE')",
        b.event, b.photographer, "relapply/${UUID.randomUUID()}",
    )

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun alertsOf(b: Batch): List<List<Any?>> =
        jdbc.queryForList("SELECT kind, dedupe_key, would_have, photos FROM relevance_alerts WHERE batch_id = ? ORDER BY kind", b.id)
            .map { listOf(it["kind"], it["dedupe_key"], it["would_have"], it["photos"]) }

    private fun dueTrigger(b: Batch): String? =
        batches.findCheckDue(QUIET_CUTOFF, NOW, 500, 10_000).singleOrNull { it.eventId == b.event && it.photographerId == b.photographer }?.trigger

    // Every table record-only must leave alone, counted across the whole database.
    private fun untouched(): List<Any?> = listOf(
        jdbc.queryForList("SELECT status, count(*) AS n FROM photos GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM flag_hidden_photos", Long::class.java),
        jdbc.queryForList("SELECT status, count(*) AS n FROM flags GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages", Long::class.java),
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2005-03-04T05:06:07.123456Z")
        val QUIET_CUTOFF: OffsetDateTime = NOW.minusMinutes(10)
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
    }
}
