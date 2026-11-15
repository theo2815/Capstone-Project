package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceAppealRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.AdminUserService
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.relevance.RelevanceAppealService
import com.quickpitik.service.relevance.RelevanceNoticeStored
import com.quickpitik.service.relevance.RelevanceNotices
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.websocket.AdminInboxEvent
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-3.2a (plan part 2, A.6): a photographer appeals the current auto-hide case of their own batch, once, within the
// window, with no other appeal open on the batch. The container is shared, so fixtures sit in 2010 and assertions look
// only at this class's own rows.
class RelevanceAppealIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var appeals: RelevanceAppealRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var strikes: com.quickpitik.repository.RelevanceStrikeRepository
    @Autowired private lateinit var events: com.quickpitik.repository.EventRepository
    @Autowired private lateinit var messages: PhotographerMessageRepository
    @Autowired private lateinit var decisionRows: AdminDecisionLogRepository
    @Autowired private lateinit var adminUsers: AdminUserService
    @Autowired private lateinit var mapper: ObjectMapper
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)

    @Test
    fun `filing raises one alert and one note`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL", hidden = 40)

        val view = file(b, c, "  These are my race photos.  ")

        val appeal = appeals.findByCaseId(c)!!
        assertEquals("OPEN", appeal.status)
        assertEquals("These are my race photos.", appeal.reason)
        assertEquals("OPEN", view.appeal?.status)
        val alert = jdbc.queryForList("SELECT kind, dedupe_key, would_have, photos FROM relevance_alerts WHERE case_id = ?", c).single()
        assertEquals(listOf("APPEAL", "appeal:${appeal.id}", false, 40), listOf(alert["kind"], alert["dedupe_key"], alert["would_have"], alert["photos"]))
        val note = notes(b).single()
        assertEquals("relevance:${appeal.id}:filed", note["notice_key"])
        assertEquals(RelevanceNotices.appealReceived("Rel Run", copied = false),
            com.quickpitik.service.relevance.RelevanceNote(note["title"] as String, note["body"] as String))
        assertEquals(c, note["relevance_case_id"])
        val events = Mockito.mockingDetails(publisher).invocations.map { it.arguments[0] }
        assertEquals(1, events.filterIsInstance<AdminInboxEvent>().count { it.payload["type"] == "relevance_alert" && it.payload["actorId"] == c.toString() })
        assertEquals(1, events.filterIsInstance<RelevanceNoticeStored>().count { it.messageId == note["id"] })
    }

    @Test
    fun `a copied case's appeal note says the payouts stay on hold`() {
        val b = batch()
        val c = case(b, verdict = "COPIED", action = "HIDE_ALL")

        file(b, c)

        assertEquals(RelevanceNotices.appealReceived("Rel Run", copied = true).body, notes(b).single()["body"])
    }

    @Test
    fun `an admin's hide of a can't tell batch can be appealed`() {
        val b = batch()
        val c = case(b, verdict = "CANT_TELL", action = "HIDE_ALL", note = "ADMIN_HIDE")

        file(b, c)

        assertEquals("OPEN", appeals.findByCaseId(c)!!.status)
    }

    @Test
    fun `an appeal after the window has closed is refused`() {
        val b = batch()
        // Noticed on 2010-05-06 in Manila: the last day is 2010-05-20; the service clock reads 2010-05-21 there.
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL", noticeAt = OffsetDateTime.parse("2010-05-06T01:00:00Z"))
        val late = service(clock = Clock.fixed(Instant.parse("2010-05-20T16:30:00Z"), ZoneOffset.UTC))

        val closed = assertFailsWith<ConflictException> { tx.execute { late.file(b.photographer, c, "mine") } }
        // V2-T22, word for word: the website shows the server's message.
        assertEquals("APPEAL_WINDOW_CLOSED" to "The time to appeal this decision has ended.", closed.code to closed.message)
        // On the last day itself (late evening in Manila) it is still open.
        val lastDay = service(clock = Clock.fixed(Instant.parse("2010-05-20T15:30:00Z"), ZoneOffset.UTC))
        assertEquals(LocalDate.parse("2010-05-20"), tx.execute { lastDay.file(b.photographer, c, "mine") }!!.appealDeadline)
    }

    @Test
    fun `a second appeal on the same case is refused`() {
        val b = batch()
        val c = case(b, verdict = "MIXED", action = "HIDE_JUNK")
        file(b, c)

        val again = assertFailsWith<ConflictException> { file(b, c) }
        assertEquals("APPEAL_ALREADY_FILED" to "You have already appealed this decision.", again.code to again.message)
        assertEquals(1, notes(b).size)
    }

    @Test
    fun `an appeal while another is open on the batch is refused`() {
        val b = batch()
        val old = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL", stage = "CLOSED")
        jdbc.update("INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason) VALUES (?, ?, ?, 'older')", old, b.id, b.photographer)
        val c = case(b, verdict = "WRONG_EVENT", action = "HIDE_ALL")

        val open = assertFailsWith<ConflictException> { file(b, c) }
        assertEquals("APPEAL_ALREADY_OPEN" to "You already have an open appeal for your photos in this event. " +
            "We will tell you in your inbox when it is decided.", open.code to open.message)
    }

    @Test
    fun `a case that hides nothing, has no clock or is not current cannot be appealed`() {
        val b = batch()
        // A Fits case a re-check lifted: it hides junk but was never noticed, so no appeal clock runs.
        val lifted = case(b, verdict = "FITS", action = "HIDE_JUNK", noticeAt = null)
        assertEquals("NOTHING_TO_APPEAL", refused { file(b, lifted) })
        val fits = batch().let { it to case(it, verdict = "FITS", action = "NONE") }
        assertEquals("NOTHING_TO_APPEAL", refused { file(fits.first, fits.second) })
        val closed = batch().let { it to case(it, verdict = "JUNK_DUMP", action = "HIDE_ALL", stage = "CLOSED") }
        assertEquals("NOTHING_TO_APPEAL", refused { file(closed.first, closed.second) })
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM relevance_appeals WHERE case_id IN (?, ?, ?)", Int::class.java,
            lifted, fits.second, closed.second))
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL", mode = "RECORD_ONLY")

        assertEquals("NOTHING_TO_APPEAL", refused { file(b, c) })
        assertFailsWith<NotFoundException> { tx.execute { service().caseFor(b.photographer, c) } }
        assertEquals(0, notes(b).size)
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val recordOnly = service(mode = RelevanceMode.RECORD_ONLY)

        assertEquals("NOTHING_TO_APPEAL", refused { tx.execute { recordOnly.file(b.photographer, c, "mine") } })
        assertNull(appeals.findByCaseId(c))
        assertEquals(0, notes(b).size)
        Mockito.verifyNoInteractions(publisher)
    }

    @Test
    fun `an empty or over-long reason is refused`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL")

        // A NUL character passes the length check but Postgres refuses it in TEXT: a 400, not a 500.
        for (reason in listOf("", "   ", "x".repeat(1001), "mine\u0000")) {
            val ex = assertFailsWith<ValidationException> { file(b, c, reason) }
            assertEquals("reason", ex.field)
        }
        file(b, c, "x".repeat(1000))
        assertEquals(1000, appeals.findByCaseId(c)!!.reason.length)
    }

    @Test
    fun `another photographer's case is not found`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val stranger = batch().photographer

        assertFailsWith<NotFoundException> { tx.execute { service().caseFor(stranger, c) } }
        assertFailsWith<NotFoundException> { tx.execute { service().file(stranger, c, "mine") } }
        assertFailsWith<NotFoundException> { tx.execute { service().caseFor(b.photographer, UUID.randomUUID()) } }
        assertNull(appeals.findByCaseId(c))
    }

    @Test
    fun `the case shows its numbers and appeal but never Claude's reason or the words`() {
        val b = batch()
        val c = case(b, verdict = "MIXED", action = "HIDE_JUNK", hidden = 12)
        jdbc.update("UPDATE relevance_cases SET admin_reason = 'MODEL-SECRET-REASON', summary_sent = 'MODEL-SECRET-SUMMARY', " +
            "evidence_used = 'MODEL-SECRET-EVIDENCE', scan_measured = 900, scan_blurry = 7 WHERE id = ?", c)
        jdbc.update("INSERT INTO batch_relevance_words (batch_id, word, photos) VALUES (?, 'SECRETWORD', 9)", b.id)

        val before = tx.execute { service().caseFor(b.photographer, c) }!!
        assertNull(before.appeal)
        assertEquals(listOf("MIXED", "HIDE_JUNK", 1000, 300, 40, 12, 900, 7, true), listOf(before.verdict, before.action,
            before.batchPhotos, before.sampleSize, before.junkCount, before.hiddenPhotos, before.scanMeasured, before.scanBlurry, before.current))
        assertEquals(LocalDate.parse("2010-05-21"), before.appealDeadline)
        val after = file(b, c)
        val json = mapper.writeValueAsString(after)
        assertEquals("OPEN", after.appeal?.status)
        assertFalse(json.contains("SECRET"), json)
        assertFalse(json.contains("mine"), "the appeal's own reason is not echoed: $json")
    }

    @Test
    fun `an appeal stays open when the photographer is suspended`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL")
        file(b, c)
        val admin = jdbc.queryForObject("SELECT id FROM users WHERE role = 'ADMIN' ORDER BY created_at LIMIT 1", UUID::class.java)!!

        adminUsers.suspend(admin, b.photographer, "review")

        assertNotNull(jdbc.queryForObject("SELECT suspended_at FROM users WHERE id = ?", OffsetDateTime::class.java, b.photographer))
        assertEquals("OPEN", appeals.findByCaseId(c)!!.status)
    }

    @Test
    fun `the appeal queries find a case's appeal and an open appeal of the batch`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL")
        assertNull(appeals.findByCaseId(c))
        assertFalse(appeals.existsByBatchIdAndStatus(b.id, "OPEN"))
        val id = jdbc.queryForObject("INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason) VALUES (?, ?, ?, 'r') RETURNING id",
            UUID::class.java, c, b.id, b.photographer)
        assertEquals(id, appeals.findByCaseId(c)!!.id)
        assertTrue(appeals.existsByBatchIdAndStatus(b.id, "OPEN"))
        jdbc.update("UPDATE relevance_appeals SET status = 'LOST' WHERE id = ?", id)
        assertFalse(appeals.existsByBatchIdAndStatus(b.id, "OPEN"))
    }

    @Test
    fun `the own-case queries read only the photographer's own case and lock it`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val other = batch().photographer

        val row = cases.ownCase(c, b.photographer)!!
        assertEquals(listOf(c, b.id, b.event, "Rel Run", "JUNK_DUMP", "HIDE_ALL", "APPLIED", "AUTO_HIDE"),
            listOf(row.caseId, row.batchId, row.eventId, row.eventName, row.verdict, row.action, row.stage, row.appliedMode))
        assertNull(cases.ownCase(c, other))
        assertEquals(c, tx.execute { cases.lockCase(c) })
        assertNull(tx.execute { cases.lockCase(UUID.randomUUID()) })
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun file(b: Batch, c: UUID, reason: String = "mine") = tx.execute { service().file(b.photographer, c, reason) }!!

    private fun refused(block: () -> Unit): String = assertFailsWith<ConflictException> { block() }.code

    private fun service(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, clock: Clock = CLOCK): RelevanceAppealService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        @Suppress("UNCHECKED_CAST")
        val provider = Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false), provider)
        val decisions = AdminDecisionLogService(decisionRows, messages, publisher)
        val notifier = RelevanceNotifier(cases, decisions, switch, props, clock, tx, publisher)
        return RelevanceAppealService(cases, appeals, alerts, notifier, switch, props, clock, publisher,
            com.quickpitik.service.relevance.RelevanceConfirmer(cases, strikes, alerts, events, notifier, props, publisher), decisions,
            Mockito.mock(com.quickpitik.service.relevance.RelevanceHoldService::class.java))
    }

    private fun batch(): Batch {
        val photographer = id("INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "relappeal-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2010-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relappeal-${UUID.randomUUID()}")
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer), event, photographer)
    }

    // Noticed 2010-05-07 in Manila by default, so the last day to appeal is 2010-05-21.
    private fun case(
        b: Batch, verdict: String, action: String, mode: String = "AUTO_HIDE", stage: String = "APPLIED", note: String? = null,
        hidden: Int = 0, noticeAt: OffsetDateTime? = NOTICED,
    ): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, verdict_note, decided_at, " +
            "applied_mode, action, applied_at, measured, junk, hidden_photos, notice_at, closed_at, close_reason) " +
            "VALUES (?, 'QUIET', ?, 1000, ?, ?, ?, ?, ?, ?, ?, 300, 40, ?, ?, ?, ?)",
        b.id, NOTICED, stage, verdict, note, NOTICED, mode, action, NOTICED, hidden, noticeAt,
        if (stage == "CLOSED") NOTICED else null, if (stage == "CLOSED") "SUPERSEDED" else null,
    )

    private fun notes(b: Batch): List<Map<String, Any?>> = jdbc.queryForList(
        "SELECT * FROM photographer_messages WHERE photographer_id = ? AND kind = 'relevance_notice'", b.photographer)

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOTICED: OffsetDateTime = OffsetDateTime.parse("2010-05-07T02:00:00Z")
        // 2010-05-10 in Manila: inside the window.
        val CLOCK: Clock = Clock.fixed(Instant.parse("2010-05-10T02:00:00Z"), ZoneOffset.UTC)
    }
}
