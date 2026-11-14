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

// RV-2.2a (plan part 2, A.3): in auto-hide a decided case becomes the batch's current case with its hold target, only
// for a photographer who accepted the minimum Terms version. No photo is touched yet (the hide step is RV-2.2b).
// The container is shared, so fixtures sit in 2006 and assertions look only at this class's own rows.
class RelevanceAutoHideApplyIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `with the mode off nothing runs`() {
        val b = batch(photos = 600)
        val c = case(b, "DECIDED", "JUNK_DUMP")

        service(mode = RelevanceMode.OFF, maxBatchesPerTick = 10_000).tick()

        assertEquals(listOf("DECIDED", null, null), row(c, "stage", "applied_mode", "action"))
        assertEquals(emptyList(), alertsOf(b))
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val b = batch(photos = 600)
        repeat(3) { photo(b) }
        val c = case(b, "DECIDED", "COPIED")
        val before = untouched()

        service(mode = RelevanceMode.RECORD_ONLY).applyCase(c, NOW)

        assertEquals(listOf("APPLIED", "RECORD_ONLY", "NONE"), row(c, "stage", "applied_mode", "scan_state"))
        assertEquals(listOf(true, true), alertsOf(b).map { it[2] })
        assertEquals(before, untouched())
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch(photos = 600)
        val old = case(b, "APPLIED", "FITS", applied = "RECORD_ONLY")
        // Auto-hide is on, but this photographer has not consented, so the same verdict closes the new case.
        jdbc.update("UPDATE users SET terms_version = '1.0' WHERE id = ?", b.photographer)
        val new = case(b, "DECIDED", "FITS")

        service().applyCase(new, NOW)

        assertEquals(listOf("CLOSED", "SAME_VERDICT"), row(new, "stage", "close_reason"))
        assertEquals(listOf("APPLIED", "RECORD_ONLY", "NONE"), row(old, "stage", "applied_mode", "scan_state"))
    }

    @Test
    fun `a decided case in auto-hide becomes current with its target`() {
        val mixed = batch(photos = 600)
        val m = case(mixed, "DECIDED", "MIXED", measured = 300, junk = 120)
        val dump = batch(photos = 600)
        val d = case(dump, "DECIDED", "JUNK_DUMP")
        val fits = batch(photos = 40)
        val f = case(fits, "DECIDED", "FITS")
        repeat(3) { photo(dump) }
        val before = untouched()
        val service = service()

        listOf(m, d, f).forEach { service.applyCase(it, NOW) }

        assertEquals(listOf("APPLIED", "AUTO_HIDE", "HIDE_JUNK", "QUEUED"), row(m, "stage", "applied_mode", "action", "scan_state"))
        assertEquals(listOf("APPLIED", "AUTO_HIDE", "HIDE_ALL", "NONE"), row(d, "stage", "applied_mode", "action", "scan_state"))
        assertEquals(listOf("APPLIED", "AUTO_HIDE", "NONE", "QUEUED"), row(f, "stage", "applied_mode", "action", "scan_state"))
        // Only a copied verdict raises an alert at apply; the big-hide alert waits for the real hide (RV-2.2b).
        assertEquals(emptyList(), alertsOf(dump))
        assertEquals(before, untouched(), "no photo is touched in this task")
    }

    @Test
    fun `a photographer below the minimum Terms version is applied as record-only with NO_CONSENT`() {
        val below = batch(photos = 600, terms = "1.6")
        val c = case(below, "DECIDED", "JUNK_DUMP")
        val none = batch(photos = 600, terms = null)
        val n = case(none, "DECIDED", "JUNK_DUMP")
        val service = service()

        listOf(c, n).forEach { service.applyCase(it, NOW) }

        for (id in listOf(c, n)) {
            assertEquals(listOf("APPLIED", "RECORD_ONLY", "HIDE_ALL", "NO_CONSENT"), row(id, "stage", "applied_mode", "action", "verdict_note"))
        }
        assertEquals(listOf(listOf("BIG_HIDE", "big_hide:$c", true, 600)), alertsOf(below))
    }

    @Test
    fun `auto-hide with the minimum unset or blank treats nobody as consenting`() {
        for (min in listOf(null, "", "  ")) {
            val b = batch(photos = 40, terms = "9.9")
            val c = case(b, "DECIDED", "MIXED", measured = 40, junk = 10)

            service(minVersion = min).applyCase(c, NOW)

            assertEquals(listOf("RECORD_ONLY", "NO_CONSENT", "NONE"), row(c, "applied_mode", "verdict_note", "scan_state"), "min=$min")
        }
    }

    @Test
    fun `auto-hide with a cut-off unset applies as record-only`() {
        val b = batch(photos = 40)
        val c = case(b, "DECIDED", "MIXED", measured = 40, junk = 10)

        service(darkCutoff = null).applyCase(c, NOW)

        assertEquals(listOf("APPLIED", "RECORD_ONLY", "NONE"), row(c, "stage", "applied_mode", "scan_state"))
    }

    @Test
    fun `a failed check never changes the current case`() {
        val b = batch(photos = 600)
        val old = case(b, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE")
        val failed = case(b, "DECIDED", "CANT_TELL", note = "BAD_JSON")
        // With no current case, an admin's restore is the comparator: a failed check does not undo it either.
        val restored = batch(photos = 600)
        case(restored, "CLOSED", "JUNK_DUMP", applied = "AUTO_HIDE", closeReason = "RESTORED_BY_ADMIN")
        val failed2 = case(restored, "DECIDED", "CANT_TELL", note = "CALL_FAILED")
        val service = service()

        listOf(failed, failed2).forEach { service.applyCase(it, NOW) }

        assertEquals(listOf("CLOSED", "NO_RESULT"), row(failed, "stage", "close_reason"))
        assertEquals(listOf("APPLIED", "HIDE_ALL"), row(old, "stage", "action"))
        assertEquals(listOf("CLOSED", "NO_RESULT"), row(failed2, "stage", "close_reason"))
        assertNull(cases.currentFor(restored.id))
    }

    @Test
    fun `a can't tell never lifts a hide`() {
        val b = batch(photos = 600)
        val old = case(b, "APPLIED", "MIXED", applied = "AUTO_HIDE")
        val unsure = case(b, "DECIDED", "CANT_TELL", note = "SMALL_BATCH")

        service().applyCase(unsure, NOW)

        assertEquals(listOf("CLOSED", "HELD_BY_CASE"), row(unsure, "stage", "close_reason"))
        assertEquals(listOf("APPLIED", "HIDE_JUNK"), row(old, "stage", "action"))
    }

    @Test
    fun `the same verdict after a record-only case replaces it and acts`() {
        val b = batch(photos = 600)
        val old = case(b, "APPLIED", "MIXED", applied = "RECORD_ONLY")
        val new = case(b, "DECIDED", "MIXED", measured = 300, junk = 120)

        service().applyCase(new, NOW)

        assertEquals(listOf("CLOSED", "SUPERSEDED", "NONE"), row(old, "stage", "close_reason", "scan_state"))
        assertEquals(listOf("APPLIED", "AUTO_HIDE", "HIDE_JUNK", "QUEUED"), row(new, "stage", "applied_mode", "action", "scan_state"))
    }

    @Test
    fun `the same verdict on an auto-hide case queues its scan again`() {
        val b = batch(photos = 600)
        val old = case(b, "APPLIED", "FITS", applied = "AUTO_HIDE", scan = "FINISHED")
        val new = case(b, "DECIDED", "FITS", copyLow = true)

        service().applyCase(new, NOW)

        assertEquals(listOf("CLOSED", "SAME_VERDICT"), row(new, "stage", "close_reason"))
        assertEquals(listOf("APPLIED", "QUEUED", true), row(old, "stage", "scan_state", "copy_low"))
    }

    @Test
    fun `a fits or mixed re-check does not lift a whole-batch hide before the G7 rows exist`() {
        for (oldVerdict in listOf("JUNK_DUMP", "WRONG_EVENT", "COPIED")) {
            for (newVerdict in listOf("FITS", "MIXED")) {
                val b = batch(photos = 600)
                val old = case(b, "APPLIED", oldVerdict, applied = "AUTO_HIDE")
                val new = case(b, "DECIDED", newVerdict, measured = 300, junk = 120)

                service().applyCase(new, NOW)

                assertEquals(listOf("CLOSED", "HELD_BY_CASE"), row(new, "stage", "close_reason"), "$oldVerdict -> $newVerdict")
                assertEquals(listOf("APPLIED", "HIDE_ALL"), row(old, "stage", "action"), "$oldVerdict -> $newVerdict")
            }
        }
        // A whole-batch hide that becomes another whole-batch hide lifts nothing, so it is applied as usual.
        val b = batch(photos = 600)
        val old = case(b, "APPLIED", "JUNK_DUMP", applied = "AUTO_HIDE")
        val new = case(b, "DECIDED", "WRONG_EVENT")
        service().applyCase(new, NOW)
        assertEquals(listOf("CLOSED", "SUPERSEDED"), row(old, "stage", "close_reason"))
    }

    @Test
    fun `a record-only case never closes an auto-hide case with the same verdict`() {
        for ((label, service, terms) in recordOnlySetups()) {
            val b = batch(photos = 600, terms = terms)
            val old = case(b, "APPLIED", "MIXED", applied = "AUTO_HIDE", copyLow = false)
            val new = case(b, "DECIDED", "MIXED", measured = 300, junk = 120, copyLow = true)

            service.applyCase(new, NOW)

            assertEquals(listOf("CLOSED", "SAME_VERDICT"), row(new, "stage", "close_reason"), label)
            assertEquals(listOf("APPLIED", "AUTO_HIDE", true), row(old, "stage", "applied_mode", "copy_low"), label)
        }
    }

    @Test
    fun `a record-only case never replaces an auto-hide case with a different verdict`() {
        for ((label, service, terms) in recordOnlySetups()) {
            val b = batch(photos = 600, terms = terms)
            val old = case(b, "APPLIED", "MIXED", applied = "AUTO_HIDE", copyLow = false)
            val new = case(b, "DECIDED", "FITS", copyLow = true)

            service.applyCase(new, NOW)

            assertEquals(listOf("CLOSED", "HELD_BY_CASE"), row(new, "stage", "close_reason"), label)
            assertEquals(listOf("APPLIED", "AUTO_HIDE", true), row(old, "stage", "applied_mode", "copy_low"), label)
        }
    }

    // The two ways a new case is applied in record-only while an auto-hide case is current.
    private fun recordOnlySetups() = listOf(
        Triple("record-only mode", service(mode = RelevanceMode.RECORD_ONLY), "1.7"),
        Triple("no consent", service(), "1.0"),
    )

    @Test
    fun `the same verdict after an admin restore hides nothing again`() {
        val b = batch(photos = 600)
        case(b, "CLOSED", "JUNK_DUMP", applied = "AUTO_HIDE", closeReason = "RESTORED_BY_ADMIN")
        val new = case(b, "DECIDED", "JUNK_DUMP")
        // A different verdict after an admin's decision is applied as usual.
        val other = batch(photos = 600)
        case(other, "CLOSED", "JUNK_DUMP", applied = "AUTO_HIDE", closeReason = "LOOKS_FINE")
        val newer = case(other, "DECIDED", "MIXED", measured = 300, junk = 120)

        service().applyCase(new, NOW)
        service().applyCase(newer, NOW)

        assertEquals(listOf("CLOSED", "SAME_VERDICT"), row(new, "stage", "close_reason"))
        assertNull(cases.currentFor(b.id))
        assertEquals(listOf("APPLIED", "AUTO_HIDE", "HIDE_JUNK"), row(newer, "stage", "applied_mode", "action"))
    }

    @Test
    fun `a case whose event turned private is closed skipped`() {
        val b = batch(photos = 600)
        val c = case(b, "DECIDED", "JUNK_DUMP")
        jdbc.update("UPDATE events SET visibility = 'UNLISTED' WHERE id = ?", b.event)

        service().applyCase(c, NOW)

        assertEquals(listOf("CLOSED", "SKIPPED"), row(c, "stage", "close_reason"))
        assertNull(cases.currentFor(b.id))
        assertEquals(emptyList(), alertsOf(b))
    }

    @Test
    fun `a copied verdict raises a real alert even when a would-have alert exists for an older case`() {
        val b = batch(photos = 40)
        val old = case(b, "DECIDED", "COPIED")
        service(mode = RelevanceMode.RECORD_ONLY).applyCase(old, NOW)
        val new = case(b, "DECIDED", "COPIED")

        service().applyCase(new, NOW)

        assertEquals(listOf("CLOSED", "SUPERSEDED"), row(old, "stage", "close_reason"))
        assertEquals(
            listOf(listOf("COPIED", "copied:$old", true, 40), listOf("COPIED", "copied:$new", false, 40)),
            alertsOf(b).sortedBy { it[2] as Boolean }.reversed(),
        )
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun service(
        mode: RelevanceMode = RelevanceMode.AUTO_HIDE,
        darkCutoff: Double? = 5.0,
        minVersion: String? = MIN_VERSION,
        maxBatchesPerTick: Int = 3,
    ): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = darkCutoff),
            check = RelevanceProperties.Check(maxBatchesPerTick = maxBatchesPerTick))
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true))
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java),
            LegalProperties(currentVersion = "1.7", photographerMinVersion = minVersion),
            Mockito.mock(com.quickpitik.repository.RelevanceHoldRepository::class.java),
            Mockito.mock(com.quickpitik.repository.PhotoRepository::class.java),
            Mockito.mock(com.quickpitik.service.admin.AdminDecisionLogService::class.java),
            org.springframework.core.task.SyncTaskExecutor())
    }

    // A public shared race's batch, quiet, checked and scored by version 1, whose photographer accepted `terms`.
    private fun batch(photos: Int, terms: String? = "1.7"): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', ?)",
            "relauto-${UUID.randomUUID()}@test.local", terms,
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2006-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relauto-${UUID.randomUUID()}",
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
        applied: String? = null,
        closeReason: String? = null,
        scan: String = "NONE",
        note: String? = null,
        measured: Int = 40,
        junk: Int = 0,
        copyLow: Boolean? = false,
    ): UUID {
        val photos = jdbc.queryForObject("SELECT photo_count FROM event_photographer WHERE event_id = ?", Int::class.java, b.event)
        return id(
            "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, verdict_note, " +
                "copy_low, decided_at, applied_mode, action, applied_at, closed_at, close_reason, scan_state) " +
                "VALUES (?, 'QUIET', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            b.id, UPLOAD, photos, stage, measured, junk, verdict, note, copyLow, NOW,
            applied, applied?.let { action(verdict) }, applied?.let { NOW }, closeReason?.let { NOW }, closeReason, scan,
        )
    }

    private fun action(verdict: String) = when (verdict) {
        "MIXED" -> "HIDE_JUNK"
        "JUNK_DUMP", "WRONG_EVENT", "COPIED" -> "HIDE_ALL"
        else -> "NONE"
    }

    private fun photo(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, 125, 'LIVE')",
        b.event, b.photographer, "relauto/${UUID.randomUUID()}",
    )

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun alertsOf(b: Batch): List<List<Any?>> =
        jdbc.queryForList("SELECT kind, dedupe_key, would_have, photos FROM relevance_alerts WHERE batch_id = ? ORDER BY kind", b.id)
            .map { listOf(it["kind"], it["dedupe_key"], it["would_have"], it["photos"]) }

    // Every table applying must leave alone in this task, counted across the whole database.
    private fun untouched(): List<Any?> = listOf(
        jdbc.queryForList("SELECT status, count(*) AS n FROM photos GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM flag_hidden_photos", Long::class.java),
        jdbc.queryForList("SELECT status, count(*) AS n FROM flags GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages", Long::class.java),
        jdbc.queryForObject("SELECT count(*) FROM admin_decision_log", Long::class.java),
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        const val MIN_VERSION = "1.7"
        val NOW: OffsetDateTime = OffsetDateTime.parse("2006-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
    }
}
