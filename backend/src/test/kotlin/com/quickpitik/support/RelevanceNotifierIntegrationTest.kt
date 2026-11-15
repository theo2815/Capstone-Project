package com.quickpitik.support

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.email.EmailService
import com.quickpitik.service.photographer.PhotographerMessageService
import com.quickpitik.service.relevance.RelevanceNoticeEmailListener
import com.quickpitik.service.relevance.RelevanceNotices
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.websocket.PhotographerMessageCreatedEvent
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// RV-3.1 (plan part 2, A.5): the notify step reads current auto-hide cases that owe a note and sends each once, in the
// approved words, with the case id. The container is shared, so fixtures sit in 2009 and assertions look only at this
// class's own rows.
class RelevanceNotifierIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var messages: PhotographerMessageRepository
    @Autowired private lateinit var decisionRows: AdminDecisionLogRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch()
        val c = case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL", mode = "RECORD_ONLY")

        notifier().notifyOwed()

        assertEquals(0, notes(b).size)
        assertNull(row(c, "notice_at").single())
    }

    @Test
    fun `a note is sent once however many ticks or server copies run`() {
        val dump = batch()
        val dumpCase = case(dump, verdict = "JUNK_DUMP", action = "HIDE_ALL")
        val fits = batch()
        val fitsCase = case(fits, verdict = "FITS", scan = "FINISHED", scanBlurry = 3)
        val one = notifier()
        val two = notifier()

        repeat(2) { one.notifyOwed(); two.notifyOwed() }

        assertEquals(listOf("relevance:$fitsCase:blurry"), notes(fits).map { it["notice_key"] })
        assertEquals(listOf("relevance:$dumpCase:verdict"), notes(dump).map { it["notice_key"] })
        assertNotNull(row(dumpCase, "notice_at").single())
        assertNull(row(fitsCase, "notice_at").single(), "a blurry note starts no appeal clock")
    }

    @Test
    fun `the note carries the case id in the list and in the live push`() {
        val b = batch()
        val c = case(b, verdict = "WRONG_EVENT", action = "HIDE_ALL")

        notifier().notifyOwed()

        val listed = PhotographerMessageService(messages).list(b.photographer, PaginationParams(0, 50)).single()
        assertEquals(c, listed.relevanceCaseId)
        assertEquals("relevance_notice", listed.kind)
        val pushed = ArgumentCaptor.forClass(Any::class.java)
        Mockito.verify(publisher, Mockito.atLeastOnce()).publishEvent(pushed.capture())
        val payload = pushed.allValues.filterIsInstance<PhotographerMessageCreatedEvent>().single { it.photographerId == b.photographer }.payload
        assertEquals(c.toString(), (payload["message"] as Map<*, *>)["relevanceCaseId"])
    }

    @Test
    fun `Claude's reason never appears in a note or an e-mail`() {
        val b = batch()
        val c = case(b, verdict = "MIXED", action = "HIDE_JUNK")
        jdbc.update("UPDATE relevance_cases SET admin_reason = 'MODEL-SECRET-REASON', summary_sent = 'MODEL-SECRET-SUMMARY' WHERE id = ?", c)

        notifier().notifyOwed()

        val note = notes(b).single()
        listOf("title", "body").forEach { assertFalse(note[it].toString().contains("MODEL-SECRET"), it) }
        val email = Mockito.mock(EmailService::class.java)
        emailListener(email).sendIfPending(note["id"] as UUID)
        // The title and body handed to the e-mail (Kotlin and argument captors do not mix on non-null parameters).
        val sent = Mockito.mockingDetails(email).invocations.single { it.method.name == "sendRelevanceNotice" }.arguments
        assertEquals(listOf(note["title"], note["body"]), sent.slice(2..3))
        sent.forEach { assertFalse(it.toString().contains("MODEL-SECRET"), it.toString()) }
    }

    @Test
    fun `the copied note never names another photographer`() {
        val b = batch()
        val other = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Bea Otherphotog', 'PHOTOGRAPHER')",
            "relnote-${UUID.randomUUID()}@test.local")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 5, ?, ?)",
            b.event, other, NOW, NOW)
        case(b, verdict = "COPIED", action = "HIDE_ALL")

        notifier().notifyOwed()

        val note = notes(b).single()
        assertEquals("Photos off sale and payouts on hold", note["title"])
        listOf("Bea", "Otherphotog").forEach { assertFalse(note["body"].toString().contains(it), it) }
    }

    @Test
    fun `a fits batch with no blurry photos gets no note`() {
        val clean = batch()
        case(clean, verdict = "FITS", scan = "FINISHED", scanBlurry = 0)
        val scanning = batch()
        case(scanning, verdict = "FITS", scan = "RUNNING", scanBlurry = 4)

        notifier().notifyOwed()

        assertEquals(0, notes(clean).size)
        assertEquals(0, notes(scanning).size, "no note until the scan has finished")
    }

    @Test
    fun `a fits case a re-check lifted to a junk hide gets no blurry note`() {
        // Its blurry photos are off sale (HIDE_JUNK), so "They are still on sale" would be untrue.
        val b = batch()
        case(b, verdict = "FITS", action = "HIDE_JUNK", scan = "FINISHED", scanBlurry = 3)

        notifier().notifyOwed()

        assertEquals(0, notes(b).size)
    }

    @Test
    fun `a note whose event never fired is e-mailed once by the sweep`() {
        val b = batch()
        case(b, verdict = "JUNK_DUMP", action = "HIDE_ALL")
        notifier().notifyOwed()
        val note = notes(b).single()
        // The after-commit event was lost: no claim, and the note is older than the stale-claim cutoff.
        jdbc.update("UPDATE photographer_messages SET created_at = now() - interval '1 hour' WHERE id = ?", note["id"])
        val email = Mockito.mock(EmailService::class.java)
        val listener = emailListener(email)

        repeat(2) { listener.retryUnsentEmails() }

        assertEquals(1, Mockito.mockingDetails(email).invocations.count { it.method.name == "sendRelevanceNotice" })
        assertNotNull(jdbc.queryForMap("SELECT email_delivered_at FROM photographer_messages WHERE id = ?", note["id"])["email_delivered_at"])
    }

    @Test
    fun `each verdict change owes its own note in the approved words`() {
        val mixed = batch()
        val mixedCase = case(mixed, verdict = "MIXED", action = "HIDE_JUNK")
        val blurry = batch()
        case(blurry, verdict = "FITS", scan = "FINISHED", scanBlurry = 1)
        val flipped = batch()
        val flippedCase = case(flipped, verdict = "FITS", note = "SCAN_FLIP", flipped = true)
        val lifted = batch()
        val liftedCase = case(lifted, verdict = "JUNK_DUMP", action = "HIDE_ALL", stage = "CLOSED", closeReason = "RECHECK_LIFTED")
        val adminHid = batch()
        val adminCase = case(adminHid, verdict = "CANT_TELL", action = "HIDE_ALL", note = "ADMIN_HIDE")

        notifier().notifyOwed()

        // 2009-05-06 23:00 UTC is 2009-05-07 in Manila; plus 14 days.
        val by = "MAY 21 · 2009"
        assertEquals(listOf(RelevanceNotices.mixed("Rel Run", 300, 1000, 40, by)), notesOf(mixed))
        assertEquals(listOf(RelevanceNotices.blurry("Rel Run", 900, 1)), notesOf(blurry))
        assertEquals(listOf(RelevanceNotices.restoredAfterScan("Rel Run")), notesOf(flipped))
        assertEquals(listOf(RelevanceNotices.restoredAfterRecheck("Rel Run")), notesOf(lifted))
        assertEquals(listOf(RelevanceNotices.adminHide("Rel Run", by)), notesOf(adminHid))
        assertEquals(listOf("relevance:$flippedCase:restored", "relevance:$liftedCase:restored"),
            (notes(flipped) + notes(lifted)).map { it["notice_key"] })
        assertNotNull(row(mixedCase, "notice_at").single())
        assertNotNull(row(adminCase, "notice_at").single(), "the admin's hide starts the appeal clock")
        assertNull(row(flippedCase, "notice_at").single())
    }

    @Test
    fun `pushMessage with a known notice key returns the stored message and inserts nothing`() {
        val b = batch()
        val c = case(b, verdict = "FITS")
        val service = AdminDecisionLogService(decisionRows, messages, publisher)
        val first = tx.execute { service.pushMessage(b.photographer, com.quickpitik.entity.PhotographerMessageKind.RELEVANCE_NOTICE,
            "one", title = "t", noticeKey = "relevance:$c:test", relevanceCaseId = c) }!!
        val again = tx.execute { service.pushMessage(b.photographer, com.quickpitik.entity.PhotographerMessageKind.RELEVANCE_NOTICE,
            "two", title = "t", noticeKey = "relevance:$c:test", relevanceCaseId = c) }!!

        assertEquals(first.id, again.id)
        assertEquals(listOf("one"), notes(b).map { it["body"] })
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun notifier(): RelevanceNotifier {
        // Every owed note in one pass: other classes' leftover cases share the container.
        val props = RelevanceProperties(mode = RelevanceMode.AUTO_HIDE, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0),
            check = RelevanceProperties.Check(maxBatchesPerTick = 10_000))
        @Suppress("UNCHECKED_CAST")
        val provider = Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false), provider)
        return RelevanceNotifier(cases, AdminDecisionLogService(decisionRows, messages, publisher), switch, props,
            Clock.fixed(NOW.toInstant(), ZoneOffset.UTC), tx, publisher)
    }

    private fun emailListener(email: EmailService): RelevanceNoticeEmailListener {
        @Suppress("UNCHECKED_CAST")
        val provider = Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
        val props = RelevanceProperties(mode = RelevanceMode.AUTO_HIDE, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        return RelevanceNoticeEmailListener(messages, users, email, tx, Clock.systemUTC(), PhotoRemovalProperties(),
            RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false), provider))
    }

    private fun batch(): Batch {
        val photographer = id("INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "relnote-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2009-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relnote-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 1000, ?, ?)",
            event, photographer, NOW, NOW)
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer), event, photographer)
    }

    private fun case(
        b: Batch, verdict: String, action: String = "NONE", mode: String = "AUTO_HIDE", stage: String = "APPLIED",
        note: String? = null, scan: String = "NONE", scanBlurry: Int? = null, flipped: Boolean = false, closeReason: String? = null,
    ): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, verdict_note, decided_at, " +
            "applied_mode, action, applied_at, measured, junk, scan_state, scan_measured, scan_blurry, flipped_at, closed_at, close_reason) " +
            "VALUES (?, 'QUIET', ?, 1000, ?, ?, ?, ?, ?, ?, ?, 300, 40, ?, ?, ?, ?, ?, ?)",
        b.id, NOW, stage, verdict, note, NOW, mode, action, NOW, scan, scanBlurry?.let { 900 }, scanBlurry,
        if (flipped) NOW else null, if (stage == "CLOSED") NOW else null, closeReason,
    )

    private fun notes(b: Batch): List<Map<String, Any?>> = jdbc.queryForList(
        "SELECT * FROM photographer_messages WHERE photographer_id = ? AND kind = 'relevance_notice' ORDER BY created_at, notice_key", b.photographer)

    private fun notesOf(b: Batch) = notes(b).map { com.quickpitik.service.relevance.RelevanceNote(it["title"] as String, it["body"] as String) }

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.ofInstant(Instant.parse("2009-05-06T23:00:00Z"), ZoneOffset.UTC)
    }
}
