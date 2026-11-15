package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.exception.ApiException
import com.quickpitik.config.LegalProperties
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceAppealRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.relevance.RelevanceAppealService
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceNote
import com.quickpitik.service.relevance.RelevanceNotices
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-3.2 part b (plan part 2, A.6, A.7): an admin opens an appeal's evidence and decides it. Won: a whole-batch case
// gets target HIDE_JUNK and a queued scan, a Mixed case target NONE. Lost: a whole-batch case is confirmed at once.
// Closing a won case APPEAL_WON is part c. Fixtures sit in 2011; assertions look only at this class's own rows.
class RelevanceAppealDecideIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var appeals: RelevanceAppealRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var strikes: RelevanceStrikeRepository
    @Autowired private lateinit var messages: PhotographerMessageRepository
    @Autowired private lateinit var decisionRows: AdminDecisionLogRepository
    @Autowired private lateinit var mapper: ObjectMapper
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    @Autowired private lateinit var batchRepo: BatchRelevanceRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photoRepo: PhotoRepository
    @Autowired private lateinit var decisionLog: AdminDecisionLogService
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)

    @Test
    fun `an admin who did not open the evidence cannot decide`() {
        val (b, c, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val admin = admin()
        val other = admin()

        assertEquals("EVIDENCE_NOT_OPENED", refused { decide(admin, appeal, "LOST") })
        tx.execute { service().evidenceOpened(other, appeal) }
        assertEquals("EVIDENCE_NOT_OPENED", refused { decide(admin, appeal, "LOST") }, "another admin's look does not count")
        assertEquals("OPEN", appeals.findById(appeal).get().status)
        assertNull(row(c, "confirmed_at").single())

        tx.execute { service().evidenceOpened(admin, appeal) }
        assertEquals("LOST", decide(admin, appeal, "LOST").status)
        assertEquals(1, notes(b).count { it["notice_key"] == "relevance:$appeal:lost" })
    }

    @Test
    fun `a lost appeal confirms the case at once`() {
        val (b, c, appeal) = appealed(verdict = "WRONG_EVENT", action = "HIDE_ALL")
        val admin = admin()

        opened(admin, appeal)
        val decided = decide(admin, appeal, "LOST", note = "Dates far from the race")

        assertEquals(listOf("LOST", admin, "Dates far from the race"), listOf(decided.status, decided.decidedBy, decided.decisionNote))
        assertNotNull(row(c, "confirmed_at").single())
        val strike = jdbc.queryForMap("SELECT photographer_id, event_id, verdict FROM relevance_strikes WHERE case_id = ?", c)
        assertEquals(listOf(b.photographer, b.event, "WRONG_EVENT"), listOf(strike["photographer_id"], strike["event_id"], strike["verdict"]))
        assertEquals(listOf("APPLIED", "HIDE_ALL"), row(c, "stage", "action"), "the hide stays")
        assertEquals(listOf(RelevanceNotices.appealLost("Rel Run", copied = false)), notesOf(b))
    }

    @Test
    fun `a lost copied appeal keeps the payouts sentence`() {
        val (b, _, appeal) = appealed(verdict = "COPIED", action = "HIDE_ALL")
        val admin = admin()
        opened(admin, appeal)

        decide(admin, appeal, "LOST")

        assertEquals(listOf(RelevanceNotices.appealLost("Rel Run", copied = true)), notesOf(b))
    }

    @Test
    fun `confirming a case twice writes one strike`() {
        val (_, c, _) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val confirmer = RelevanceConfirmer(cases, strikes)

        val first = tx.execute { confirmer.confirm(c, NOW) }
        val second = tx.execute { confirmer.confirm(c, NOW.plusDays(1)) }

        assertEquals(listOf(true, false), listOf(first, second))
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE case_id = ?", Int::class.java, c))
    }

    @Test
    fun `a won case never counts`() {
        val (b, c, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val admin = admin()
        opened(admin, appeal)

        assertEquals("WON", decide(admin, appeal, "WON").status)

        assertEquals(listOf("APPLIED", "HIDE_JUNK", "QUEUED", null), row(c, "stage", "action", "scan_state", "confirmed_at"))
        assertFalse(tx.execute { RelevanceConfirmer(cases, strikes).confirm(c, NOW) }!!, "a won case is never confirmed later")
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE case_id = ?", Int::class.java, c))
        assertEquals(listOf(RelevanceNotices.appealWon("Rel Run", wholeBatch = true)), notesOf(b))
    }

    // RV-3.2 part c (A.6): once for an automatic junk-dump hide (AUTO holds), once for an admin's hide (ADMIN holds).
    @Test
    fun `a won appeal on a whole-batch hide queues the scan and the good photos come back while the junk stays hidden`() {
        val admin = admin()
        for ((verdict, note, heldBy) in listOf(Triple("JUNK_DUMP", null, "AUTO"), Triple("CANT_TELL", "ADMIN_HIDE", "ADMIN"))) {
            val (b, c, appeal) = appealed(verdict = verdict, action = "HIDE_ALL", note = note)
            val clean = List(2) { heldPhoto(b, c, heldBy, sharpness = 50.0) }
            val junk = heldPhoto(b, c, heldBy, sharpness = 10.0)
            opened(admin, appeal)

            decide(admin, appeal, "WON")
            assertEquals(listOf("HIDE_JUNK", "QUEUED"), row(c, "action", "scan_state"), verdict)
            tickUntilClosed(c)

            assertEquals(listOf("LIVE", "LIVE"), clean.map(::status), verdict)
            assertEquals("HIDDEN", status(junk), verdict)
            assertEquals(listOf("CLOSED", "APPEAL_WON", null), row(c, "stage", "close_reason", "confirmed_at"), verdict)
            assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE case_id = ?", Int::class.java, c))
            assertFalse(tx.execute { RelevanceConfirmer(cases, strikes).confirm(c, NOW) }!!, verdict)
        }
    }

    // RV-3.3 (G8): a won appeal never brings back a photo the photographer hid themselves.
    @Test
    fun `a won admin's hide never lifts the photographer's own hold`() {
        val admin = admin()
        val (b, c, appeal) = appealed(verdict = "CANT_TELL", action = "HIDE_ALL", note = "ADMIN_HIDE")
        val clean = heldPhoto(b, c, "ADMIN", sharpness = 50.0)
        val own = heldPhoto(b, c, "ADMIN", sharpness = 50.0).also {
            jdbc.update("INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at) VALUES (?, ?, 'PHOTOGRAPHER', ?, ?)",
                b.id, it, c, NOW)
        }
        opened(admin, appeal)

        decide(admin, appeal, "WON")
        tickUntilClosed(c)

        assertEquals(listOf("LIVE", "HIDDEN"), listOf(clean, own).map(::status))
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM flag_hidden_photos WHERE photo_id = ? AND held_by = 'PHOTOGRAPHER' " +
            "AND restored_at IS NULL AND restored_by IS NULL", Int::class.java, own), "the photographer's hold is neither lifted nor stamped")
    }

    @Test
    fun `a mixed case that wins gets target none`() {
        val (b, c, appeal) = appealed(verdict = "MIXED", action = "HIDE_JUNK")
        val admin = admin()
        opened(admin, appeal)

        decide(admin, appeal, "WON")

        assertEquals(listOf("APPLIED", "NONE", "NONE"), row(c, "stage", "action", "scan_state"))
        assertEquals(listOf(RelevanceNotices.appealWon("Rel Run", wholeBatch = false)), notesOf(b))
    }

    @Test
    fun `an admin's hide is never confirmed, won or lost`() {
        val admin = admin()
        val (_, lost, lostAppeal) = appealed(verdict = "CANT_TELL", action = "HIDE_ALL", note = "ADMIN_HIDE")
        opened(admin, lostAppeal)
        decide(admin, lostAppeal, "LOST")
        val (_, won, wonAppeal) = appealed(verdict = "CANT_TELL", action = "HIDE_ALL", note = "ADMIN_HIDE")
        opened(admin, wonAppeal)
        decide(admin, wonAppeal, "WON")

        assertEquals(listOf("HIDE_ALL", null), row(lost, "action", "confirmed_at"))
        // V2-T23 promised the scan: the good photos come back once part c lifts the admin's holds.
        assertEquals(listOf("HIDE_JUNK", "QUEUED", null), row(won, "action", "scan_state", "confirmed_at"))
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE case_id IN (?, ?)", Int::class.java, lost, won))
    }

    @Test
    fun `a decided appeal cannot be opened or decided again`() {
        val (_, _, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val admin = admin()
        opened(admin, appeal)
        decide(admin, appeal, "WON")

        assertEquals("APPEAL_ALREADY_DECIDED", refused { decide(admin, appeal, "LOST") })
        assertEquals("APPEAL_ALREADY_DECIDED", refused { tx.execute { service().evidenceOpened(admin, appeal) } })
        assertEquals("WON", appeals.findById(appeal).get().status)
    }

    @Test
    fun `an unknown outcome is refused`() {
        val (_, _, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val admin = admin()
        opened(admin, appeal)

        assertFailsWith<com.quickpitik.exception.ValidationException> { decide(admin, appeal, "MAYBE") }
        assertEquals("OPEN", appeals.findById(appeal).get().status)
    }

    @Test
    fun `a note with a NUL character is refused`() {
        val (_, _, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val admin = admin()
        opened(admin, appeal)

        // Postgres refuses NUL in TEXT: a 400, not a 500 at the update.
        val ex = assertFailsWith<com.quickpitik.exception.ValidationException> { decide(admin, appeal, "LOST", note = "bad\u0000note") }
        assertEquals("note", ex.field)
        assertEquals("OPEN", appeals.findById(appeal).get().status)
    }

    @Test
    fun `deciding an appeal whose case was closed meanwhile records the outcome and changes nothing else`() {
        val admin = admin()
        // An admin restored each batch while its appeal was open: the case is closed, the appeal still open.
        val decided = listOf("LOST", "WON").map { outcome ->
            val (b, c, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
            jdbc.update("UPDATE relevance_cases SET stage = 'CLOSED', closed_at = ?, close_reason = 'RESTORED_BY_ADMIN' WHERE id = ?", NOW, c)
            opened(admin, appeal)
            assertEquals(outcome, decide(admin, appeal, outcome).status)
            Triple(b, c, appeal)
        }

        for ((b, c, _) in decided) {
            assertEquals(listOf("CLOSED", "HIDE_ALL", "NONE", null), row(c, "stage", "action", "scan_state", "confirmed_at"))
            assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE case_id = ?", Int::class.java, c))
            assertEquals(0, notes(b).size, "no decided note: it would describe a hide that is gone")
        }
    }

    @Test
    fun `with the mode off or in record-only nothing is opened or decided`() {
        val (_, c, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val admin = admin()
        for (mode in listOf(RelevanceMode.OFF, RelevanceMode.RECORD_ONLY)) {
            val s = service(mode)
            assertEquals("RELEVANCE_ACTIONS_OFF", refused { tx.execute { s.evidenceOpened(admin, appeal) } })
            assertEquals("RELEVANCE_ACTIONS_OFF", refused { tx.execute { s.decide(admin, appeal, "LOST", null) } })
        }
        assertEquals(listOf(null, "OPEN"), listOf(appeals.findById(appeal).get().evidenceOpenedAt, appeals.findById(appeal).get().status))
        assertNull(row(c, "confirmed_at").single())
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val (_, c, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL", mode = "RECORD_ONLY")
        val admin = admin()
        opened(admin, appeal)

        decide(admin, appeal, "LOST")

        assertEquals(listOf("HIDE_ALL", "NONE", null), row(c, "action", "scan_state", "confirmed_at"))
    }

    @Test
    fun `a new verdict after a decided appeal can be appealed`() {
        val (b, old, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val admin = admin()
        opened(admin, appeal)
        decide(admin, appeal, "LOST")
        // A later check reaches a new verdict: the old case is superseded and the new one is noticed.
        jdbc.update("UPDATE relevance_cases SET stage = 'CLOSED', closed_at = ?, close_reason = 'SUPERSEDED' WHERE id = ?", NOW, old)
        val next = case(b, verdict = "COPIED", action = "HIDE_ALL")

        val view = tx.execute { service().file(b.photographer, next, "These are mine") }!!

        assertEquals("OPEN", view.appeal?.status)
    }

    @Test
    fun `the admin's note is never shown to the photographer`() {
        val (b, c, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val admin = admin()
        opened(admin, appeal)

        decide(admin, appeal, "LOST", note = "ADMIN-SECRET-NOTE")

        val view = mapper.writeValueAsString(tx.execute { service().caseFor(b.photographer, c) })
        assertFalse(view.contains("ADMIN-SECRET"), view)
        notes(b).forEach { n -> listOf("title", "body").forEach { assertFalse(n[it].toString().contains("ADMIN-SECRET"), it) } }
        assertEquals("ADMIN-SECRET-NOTE", appeals.findById(appeal).get().decisionNote)
    }

    @Test
    fun `the appeals list shows each status, an appeal on a deleted event too, and the appeals tab its open cases`() {
        val (_, open, openAppeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val (gone, _, goneAppeal) = appealed(verdict = "WRONG_EVENT", action = "HIDE_ALL")
        jdbc.update("UPDATE events SET deleted_at = ? WHERE id = ?", NOW, gone.event)
        val (_, decidedCase, decidedAppeal) = appealed(verdict = "MIXED", action = "HIDE_JUNK")
        val admin = admin()
        opened(admin, decidedAppeal)
        decide(admin, decidedAppeal, "WON")

        val openIds = all("OPEN").map { it.appealId }
        assertTrue(openAppeal in openIds && goneAppeal in openIds && decidedAppeal !in openIds)
        assertTrue(decidedAppeal in all("WON").map { it.appealId })
        val everything = all("all")
        assertTrue(listOf(openAppeal, goneAppeal, decidedAppeal).all { id -> id in everything.map { it.appealId } })
        val row = everything.single { it.appealId == openAppeal }
        assertEquals(listOf(open, "Rel Run", "JUNK_DUMP", "OPEN", "These are mine"), listOf(row.caseId, row.eventName, row.verdict, row.status, row.reason))
        assertEquals("Rel Run", everything.single { it.appealId == goneAppeal }.eventName, "a deleted event's appeal keeps its name")

        val tab = cases.pageCurrentCases("appeals", com.quickpitik.common.OffsetLimitPageable(PaginationParams(0, 200))).content.map { it.caseId }
        assertTrue(open in tab)
        assertFalse(decidedCase in tab, "a decided appeal leaves the tab")
    }

    @Test
    fun `the appeal lock reads one appeal`() {
        val (_, _, appeal) = appealed(verdict = "JUNK_DUMP", action = "HIDE_ALL")

        assertEquals(appeal, tx.execute { appeals.lockAppeal(appeal) }?.id)
        assertNull(tx.execute { appeals.lockAppeal(UUID.randomUUID()) })
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    // Pages through every appeal of the status: the container is shared with other classes' rows.
    private fun all(status: String) = tx.execute { service().listAppeals(status, PaginationParams(0, 200)) }!!.items

    private fun decide(admin: UUID, appeal: UUID, outcome: String, note: String? = null) =
        tx.execute { service().decide(admin, appeal, outcome, note) }!!

    private fun opened(admin: UUID, appeal: UUID) = tx.execute { service().evidenceOpened(admin, appeal) }

    private fun refused(block: () -> Unit): String = assertFailsWith<ApiException> { block() }.let {
        assertEquals(HttpStatus.CONFLICT, it.status)
        it.code
    }

    private fun service(mode: RelevanceMode = RelevanceMode.AUTO_HIDE): RelevanceAppealService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        @Suppress("UNCHECKED_CAST")
        val provider = Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false), provider)
        val decisions = AdminDecisionLogService(decisionRows, messages, publisher)
        val notifier = RelevanceNotifier(cases, decisions, switch, props, CLOCK, tx, publisher)
        return RelevanceAppealService(cases, appeals, alerts, notifier, switch, props, CLOCK, publisher,
            RelevanceConfirmer(cases, strikes), decisions)
    }

    // The tick's hold and scan steps (not the whole tick: its other steps would act on other classes' rows), until the
    // case closes; the scan finds every photo measured already, so no image is read.
    private fun tickUntilClosed(caseId: UUID) {
        val check = checks()
        repeat(20) {
            check.reconcileHolds()
            if (row(caseId, "stage").single() == "CLOSED") return
            check.scanTurns()
        }
    }

    private fun checks(): RelevanceCheckService {
        val props = RelevanceProperties(mode = RelevanceMode.AUTO_HIDE, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        @Suppress("UNCHECKED_CAST")
        val provider = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
            .also { Mockito.`when`(it.getIfAvailable()).thenReturn(Mockito.mock(RekognitionAiClient::class.java)) }
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true), provider)
        return RelevanceCheckService(batchRepo, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts, publisher,
            LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"), holds, photoRepo, decisionLog,
            ThreadPoolTaskExecutor().apply { initialize() }, Mockito.mock(RelevanceNotifier::class.java))
    }

    // A measured photo of the batch, hidden and held under the case.
    private fun heldPhoto(b: Batch, caseId: UUID, heldBy: String, sharpness: Double): UUID {
        val photo = id("INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'reldecide/thumb', 'reldecide/wm', 125, 'HIDDEN')", b.event, b.photographer, "reldecide/${UUID.randomUUID()}")
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, sharpness, brightness, person_conf, race_conf) " +
            "VALUES (?, ?, ?, ?, ?, 50, 0.9, 0.9)", photo, b.id, NOW, sharpness, sharpness)
        jdbc.update("INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at) VALUES (?, ?, ?, ?, ?)",
            b.id, photo, heldBy, caseId, NOW)
        return photo
    }

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun admin(): UUID = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Adm', 'ADMIN')",
        "reldecide-admin-${UUID.randomUUID()}@test.local")

    private fun batch(): Batch {
        val photographer = id("INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "reldecide-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2011-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "reldecide-${UUID.randomUUID()}")
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer), event, photographer)
    }

    private fun case(b: Batch, verdict: String, action: String, mode: String = "AUTO_HIDE", note: String? = null): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, verdict_note, decided_at, " +
            "applied_mode, action, applied_at, measured, junk, hidden_photos, notice_at) " +
            "VALUES (?, 'QUIET', ?, 1000, 'APPLIED', ?, ?, ?, ?, ?, ?, 300, 40, 40, ?)",
        b.id, NOW, verdict, note, NOW, mode, action, NOW, NOW,
    )

    // A current case of a new batch with an open appeal (rows written as the filing route writes them).
    private fun appealed(verdict: String, action: String, mode: String = "AUTO_HIDE", note: String? = null): Triple<Batch, UUID, UUID> {
        val b = batch()
        val c = case(b, verdict, action, mode, note)
        val appeal = id("INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason) VALUES (?, ?, ?, 'These are mine')",
            c, b.id, b.photographer)
        return Triple(b, c, appeal)
    }

    private fun notes(b: Batch): List<Map<String, Any?>> = jdbc.queryForList(
        "SELECT * FROM photographer_messages WHERE photographer_id = ? AND kind = 'relevance_notice'", b.photographer)

    private fun notesOf(b: Batch) = notes(b).map { RelevanceNote(it["title"] as String, it["body"] as String) }

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2011-05-07T02:00:00Z")
        val CLOCK: Clock = Clock.fixed(Instant.parse("2011-05-10T02:00:00Z"), ZoneOffset.UTC)
    }
}
