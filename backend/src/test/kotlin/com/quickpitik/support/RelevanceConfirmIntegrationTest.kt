package com.quickpitik.support

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceAppealRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceAppealService
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceNote
import com.quickpitik.service.relevance.RelevanceNotices
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.core.task.SyncTaskExecutor
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-4.1 (plan part 2, A.7, G6): the confirm step and the ladder. A current auto-hide whole-batch finding is confirmed
// when its appeal is lost, or when no appeal came by the last day to appeal; each confirmation writes one strike. A
// photographer's second strike in the rolling window sends V2-T20, the third raises one THIRD_CASE alert, and nobody is
// suspended. Fixtures sit in 2000, before every other class's rows, so only this class's cases are ever due here.
class RelevanceConfirmIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var appeals: RelevanceAppealRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var strikes: RelevanceStrikeRepository
    @Autowired private lateinit var messages: PhotographerMessageRepository
    @Autowired private lateinit var decisionRows: AdminDecisionLogRepository
    @Autowired private lateinit var batchRepo: BatchRelevanceRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photoRepo: PhotoRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)

    @Test
    fun `with the mode off nothing runs`() {
        val b = batch()
        val c = case(b)

        checks(RelevanceMode.OFF).tick()
        checks(RelevanceMode.OFF).confirmDue()

        assertNull(confirmedAt(c))
        assertEquals(0, strikeCount(b.photographer))
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val p = photographer()
        strike(p, NOW.minusDays(3))
        val b = batch(p)
        val c = case(b)

        checks(RelevanceMode.RECORD_ONLY).confirmDue()

        assertNull(confirmedAt(c))
        assertEquals(1, strikeCount(p))
        assertEquals(0, messageCount(p))
        Mockito.verifyNoInteractions(publisher)
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch()
        val c = case(b, mode = "RECORD_ONLY")

        checks().confirmDue()

        assertNull(confirmedAt(c))
        assertEquals(0, strikeCount(b.photographer))
    }

    @Test
    fun `a whole-batch hide with no appeal is confirmed after the window and not before`() {
        val b = batch()
        // Philippine date Feb 16; the note's last day to appeal is Mar 1, the clock's own day.
        val c = case(b, verdict = "WRONG_EVENT", noticeAt = OffsetDateTime.parse("2000-02-16T02:00:00Z"))

        checks().confirmDue()
        assertNull(confirmedAt(c), "the last day to appeal is not over")

        checks(clock = NOW.plusDays(1)).confirmDue()
        assertEquals(NOW.plusDays(1).toInstant(), confirmedAt(c)?.toInstant())
        val strike = jdbc.queryForMap("SELECT photographer_id, event_id, verdict FROM relevance_strikes WHERE case_id = ?", c)
        assertEquals(listOf(b.photographer, b.event, "WRONG_EVENT"), listOf(strike["photographer_id"], strike["event_id"], strike["verdict"]))
        assertEquals(listOf("APPLIED", "HIDE_ALL"), row(c, "stage", "action"), "the hide stays")
    }

    @Test
    fun `the last day to appeal is the Philippine date of the note`() {
        // 17:00Z on Feb 15 is 01:00 on Feb 16 in Manila, so the last day to appeal is Mar 1 there (Feb 29 in UTC).
        val c = case(batch(), noticeAt = OffsetDateTime.parse("2000-02-15T17:00:00Z"))

        checks().confirmDue()
        assertNull(confirmedAt(c), "Mar 1 is still the last day in Manila")

        checks(clock = NOW.plusDays(1)).confirmDue()
        assertNotNull(confirmedAt(c))
    }

    @Test
    fun `an appeal filed after the case was listed stops the stamp`() {
        val p = photographer()
        strike(p, NOW.minusDays(1))
        val c = case(batch(p))
        // The step listed the case (no appeal yet); the photographer's filing commits before the stamp.
        assertTrue(c in cases.findConfirmDue(NOW, 10_000))
        appeal(c, "OPEN")

        assertEquals(false, tx.execute { confirmer().confirm(c, NOW) })

        assertNull(confirmedAt(c))
        assertEquals(0, caseStrikes(c))
        assertEquals(emptyList(), strikeNotes(p))
    }

    @Test
    fun `an open appeal stops the clock and a won one never confirms`() {
        val open = case(batch()).also { appeal(it, "OPEN") }
        val won = case(batch()).also { appeal(it, "WON") }
        val lost = case(batch(), verdict = "COPIED", noticeAt = NOW.minusDays(2)).also { appeal(it, "LOST") }

        checks().confirmDue()

        assertNull(confirmedAt(open))
        assertNull(confirmedAt(won))
        assertNotNull(confirmedAt(lost), "a lost appeal confirms, whatever the clock")
        assertEquals(listOf(0, 0, 1), listOf(open, won, lost).map { caseStrikes(it) })
    }

    @Test
    fun `mixed, fits and can't tell cases are never confirmed`() {
        val mixed = case(batch(), verdict = "MIXED", action = "HIDE_JUNK")
        val fits = case(batch(), verdict = "FITS", action = "NONE")
        val adminHide = case(batch(), verdict = "CANT_TELL", action = "HIDE_ALL", note = "ADMIN_HIDE")
        val lifted = case(batch(), stage = "CLOSED", closeReason = "RECHECK_LIFTED")

        checks().confirmDue()

        assertEquals(listOf(null, null, null, null), listOf(mixed, fits, adminHide, lifted).map { confirmedAt(it) })
        assertEquals(listOf(0, 0, 0, 0), listOf(mixed, fits, adminHide, lifted).map { caseStrikes(it) })
    }

    @Test
    fun `the event going private does not stop the clock`() {
        val b = batch()
        val c = case(b)
        jdbc.update("UPDATE events SET visibility = 'UNLISTED' WHERE id = ?", b.event)

        checks().confirmDue()

        assertNotNull(confirmedAt(c))
        assertEquals(1, caseStrikes(c))
    }

    @Test
    fun `cases are counted across events for one photographer`() {
        val p = photographer()
        case(batch(p, "Conf North"), noticeAt = OLD)
        case(batch(p, "Conf South"), verdict = "COPIED", noticeAt = OLD.plusHours(1))

        checks().confirmDue()

        assertEquals(2, strikeCount(p))
        assertEquals(listOf(RelevanceNotices.secondCase("Conf South", 6)), strikeNotes(p))
    }

    @Test
    fun `the second case in six months sends one warning`() {
        val p = photographer()
        strike(p, NOW.minusMonths(5))
        val b = batch(p, "Conf Second")
        val c = case(b)

        checks().confirmDue()
        checks(clock = NOW.plusDays(1)).confirmDue()

        val notes = jdbc.queryForList("SELECT notice_key, relevance_case_id, title, body FROM photographer_messages " +
            "WHERE photographer_id = ? AND kind = 'relevance_notice'", p)
        assertEquals(1, notes.size)
        assertEquals("relevance:strike2:$c", notes.single()["notice_key"])
        assertEquals(c, notes.single()["relevance_case_id"])
        // V2-T20, character for character.
        assertEquals("Second confirmed case", notes.single()["title"])
        assertEquals("Your photos in Conf Second were taken off sale, and that decision is now final. This is the second " +
            "confirmed case on your account in 6 months. If there is a third within 6 months, an admin will decide whether " +
            "to suspend your account. Nothing is suspended automatically. To avoid this, upload only your own photos, to " +
            "the event they were taken at, and leave out blurry or very dark shots.", notes.single()["body"])
        assertEquals(0, thirdCaseAlerts(p))
    }

    @Test
    fun `a case older than six months does not count toward the second or the third`() {
        val p = photographer()
        strike(p, NOW.minusMonths(7))
        case(batch(p, "Conf One"), noticeAt = OLD)
        case(batch(p, "Conf Two"), noticeAt = OLD.plusHours(1))

        checks().confirmDue()

        assertEquals(listOf(RelevanceNotices.secondCase("Conf Two", 6)), strikeNotes(p))
        assertEquals(0, thirdCaseAlerts(p))
    }

    @Test
    fun `the third case raises one admin alert and suspends nobody`() {
        val p = photographer()
        case(batch(p, "Conf A"), noticeAt = OLD)
        case(batch(p, "Conf B"), noticeAt = OLD.plusHours(1))
        val third = case(batch(p, "Conf C"), noticeAt = OLD.plusHours(2))

        checks().confirmDue()
        checks(clock = NOW.plusDays(1)).confirmDue()

        val alert = jdbc.queryForList("SELECT a.case_id, a.dedupe_key, a.would_have FROM relevance_alerts a " +
            "JOIN batch_relevance br ON br.id = a.batch_id WHERE br.photographer_id = ? AND a.kind = 'THIRD_CASE'", p)
        assertEquals(listOf(listOf<Any?>(third, "third_case:$third", false)), alert.map { listOf(it["case_id"], it["dedupe_key"], it["would_have"]) })
        assertEquals(listOf(RelevanceNotices.secondCase("Conf B", 6)), strikeNotes(p), "the third case sends no second warning")
        assertNull(jdbc.queryForObject("SELECT suspended_at FROM users WHERE id = ?", OffsetDateTime::class.java, p))
    }

    @Test
    fun `each later case in the window raises one alert of its own`() {
        val p = photographer()
        strike(p, NOW.minusDays(3))
        strike(p, NOW.minusDays(2))
        val third = case(batch(p, "Conf C"), noticeAt = OLD)
        val fourth = case(batch(p, "Conf D"), noticeAt = OLD.plusHours(1))

        checks().confirmDue()
        checks(clock = NOW.plusDays(1)).confirmDue()
        tx.execute { confirmer().confirm(fourth, NOW.plusDays(2)) }

        val keys = jdbc.queryForList("SELECT a.dedupe_key FROM relevance_alerts a JOIN batch_relevance br ON br.id = a.batch_id " +
            "WHERE br.photographer_id = ? AND a.kind = 'THIRD_CASE' ORDER BY a.dedupe_key", String::class.java, p)
        assertEquals(listOf("third_case:$third", "third_case:$fourth").sorted(), keys)
        assertEquals(emptyList(), strikeNotes(p))
    }

    @Test
    fun `confirming twice writes one strike`() {
        val p = photographer()
        strike(p, NOW.minusDays(1))
        val c = case(batch(p))

        checks().confirmDue()
        checks(clock = NOW.plusDays(1)).confirmDue()
        tx.execute { confirmer().confirm(c, NOW.plusDays(2)) }

        assertEquals(1, caseStrikes(c))
        assertEquals(NOW.toInstant(), confirmedAt(c)?.toInstant())
        assertEquals(1, strikeNotes(p).size)
    }

    @Test
    fun `a lost appeal climbs the same ladder`() {
        val p = photographer()
        strike(p, NOW.minusDays(1))
        val b = batch(p, "Conf Lost")
        val c = case(b, noticeAt = NOW.minusDays(2))
        val admin = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Adm', 'ADMIN')",
            "relconfirm-admin-${UUID.randomUUID()}@test.local")
        val appeal = appeal(c, "OPEN", openedBy = admin)

        tx.execute { appealService().decide(admin, appeal, "LOST", null) }

        assertNotNull(confirmedAt(c))
        assertEquals(listOf(RelevanceNotices.secondCase("Conf Lost", 6)), strikeNotes(p))
    }

    @Test
    fun `the case page shows the confirmation and the photographer's strikes`() {
        val p = photographer()
        strike(p, NOW.minusMonths(7))
        val c = case(batch(p))
        checks().confirmDue()

        val detail = adminService().case(c)

        assertEquals(NOW.toInstant().toString(), detail.confirmedAt)
        assertEquals(1, detail.photographerStrikes, "the strike older than the window is left out")
        assertEquals(listOf<String?>(null, "0"), adminService().case(case(batch())).let { listOf(it.confirmedAt, it.photographerStrikes.toString()) })
    }

    @Test
    fun `the all tab can list only repeat cases`() {
        val repeat = photographer()
        val first = case(batch(repeat), noticeAt = OLD)
        val second = case(batch(repeat), noticeAt = OLD.plusHours(1))
        val once = case(batch(), noticeAt = OLD)
        val unconfirmed = case(batch(repeat), noticeAt = NOW)
        checks().confirmDue()

        fun listed(repeatOnly: Boolean) = cases.pageCurrentCases("all", repeatOnly, NOW.minusMonths(6), OffsetLimitPageable(0, 1000))
            .content.map { it.caseId }.filter { it in setOf(first, second, once, unconfirmed) }.toSet()

        assertEquals(setOf(first, second), listed(true))
        assertEquals(setOf(first, second, once, unconfirmed), listed(false))
        assertEquals(setOf(first, second), adminService().listCases("all", com.quickpitik.common.PaginationParams(0, 200), repeat = true)
            .items.map { it.caseId }.filter { it in setOf(first, second, once, unconfirmed) }.toSet())
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun props(mode: RelevanceMode) = RelevanceProperties(mode = mode,
        junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0),
        check = RelevanceProperties.Check(maxBatchesPerTick = 10_000))

    private fun switch(props: RelevanceProperties): RelevanceSwitch {
        @Suppress("UNCHECKED_CAST")
        val provider = Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>
        return RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false), provider)
    }

    private fun decisions() = AdminDecisionLogService(decisionRows, messages, publisher)

    private fun confirmer(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, clock: OffsetDateTime = NOW): RelevanceConfirmer {
        val props = props(mode)
        val notifier = RelevanceNotifier(cases, decisions(), switch(props), props, fixed(clock), tx, publisher)
        return RelevanceConfirmer(cases, strikes, alerts, events, notifier, props, publisher)
    }

    private fun checks(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, clock: OffsetDateTime = NOW): RelevanceCheckService {
        val props = props(mode)
        return RelevanceCheckService(batchRepo, cases, props, switch(props), fixed(clock), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts, publisher,
            LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"), holds, photoRepo, decisions(),
            SyncTaskExecutor(), Mockito.mock(RelevanceNotifier::class.java), confirmer(mode, clock))
    }

    private fun appealService(): RelevanceAppealService {
        val props = props(RelevanceMode.AUTO_HIDE)
        val notifier = RelevanceNotifier(cases, decisions(), switch(props), props, fixed(NOW), tx, publisher)
        return RelevanceAppealService(cases, appeals, alerts, notifier, switch(props), props, fixed(NOW), publisher,
            confirmer(), decisions())
    }

    private fun adminService(): AdminRelevanceService {
        val props = props(RelevanceMode.AUTO_HIDE)
        return AdminRelevanceService(cases, alerts, batchRepo, props, Mockito.mock(StorageService::class.java), StorageProperties(),
            fixed(NOW), holds, photoRepo, decisions(), tx, switch(props), strikes)
    }

    private fun fixed(at: OffsetDateTime): Clock = Clock.fixed(at.toInstant(), ZoneOffset.UTC)

    private fun photographer(): UUID = id("INSERT INTO users (email, password_hash, name, role, terms_version) " +
        "VALUES (?, 'unused', 'Conf', 'PHOTOGRAPHER', '1.7')", "relconfirm-${UUID.randomUUID()}@test.local")

    private fun batch(photographer: UUID = photographer(), eventName: String = "Conf Run"): Batch {
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) " +
            "VALUES (?, ?, DATE '2000-01-15', 'Cebu', 'ACTIVE', 'PUBLIC')", "relconfirm-${UUID.randomUUID()}", eventName)
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer), event, photographer)
    }

    private fun case(
        b: Batch,
        verdict: String = "JUNK_DUMP",
        action: String = "HIDE_ALL",
        noticeAt: OffsetDateTime? = OLD,
        mode: String = "AUTO_HIDE",
        note: String? = null,
        stage: String = "APPLIED",
        closeReason: String? = null,
    ): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, verdict_note, decided_at, " +
            "applied_mode, action, applied_at, measured, junk, hidden_photos, notice_at, close_reason, closed_at) " +
            "VALUES (?, 'QUIET', ?, 1000, ?, ?, ?, ?, ?, ?, ?, 300, 240, 1000, ?, ?, ?)",
        b.id, OLD, stage, verdict, note, OLD, mode, action, OLD, noticeAt, closeReason, closeReason?.let { OLD },
    )

    private fun appeal(caseId: UUID, status: String, openedBy: UUID? = null): UUID {
        val (batchId, photographer) = jdbc.queryForMap("SELECT br.id, br.photographer_id FROM relevance_cases c " +
            "JOIN batch_relevance br ON br.id = c.batch_id WHERE c.id = ?", caseId).let { it["id"] as UUID to it["photographer_id"] as UUID }
        return id("INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason, status, decided_at, " +
            "evidence_opened_at, evidence_opened_by) VALUES (?, ?, ?, 'These are mine', ?, ?, ?, ?)",
            caseId, batchId, photographer, status, if (status == "OPEN") null else OLD, openedBy?.let { OLD }, openedBy)
    }

    // A strike confirmed earlier, on another (deleted) case and event: no foreign key, as P6 allows.
    private fun strike(photographer: UUID, at: OffsetDateTime) {
        jdbc.update("INSERT INTO relevance_strikes (photographer_id, case_id, event_id, verdict, confirmed_at) VALUES (?, ?, ?, 'JUNK_DUMP', ?)",
            photographer, UUID.randomUUID(), UUID.randomUUID(), at)
    }

    private fun confirmedAt(caseId: UUID): OffsetDateTime? =
        jdbc.queryForObject("SELECT confirmed_at FROM relevance_cases WHERE id = ?", OffsetDateTime::class.java, caseId)

    private fun strikeCount(photographer: UUID): Int =
        jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE photographer_id = ?", Int::class.java, photographer)!!

    private fun caseStrikes(caseId: UUID): Int =
        jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE case_id = ?", Int::class.java, caseId)!!

    private fun messageCount(photographer: UUID): Int =
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages WHERE photographer_id = ?", Int::class.java, photographer)!!

    private fun strikeNotes(photographer: UUID): List<RelevanceNote> = jdbc.queryForList(
        "SELECT title, body FROM photographer_messages WHERE photographer_id = ? AND notice_key LIKE 'relevance:strike2:%' ORDER BY created_at",
        photographer).map { RelevanceNote(it["title"] as String, it["body"] as String) }

    private fun thirdCaseAlerts(photographer: UUID): Int = jdbc.queryForObject("SELECT count(*) FROM relevance_alerts a " +
        "JOIN batch_relevance br ON br.id = a.batch_id WHERE br.photographer_id = ? AND a.kind = 'THIRD_CASE'", Int::class.java, photographer)!!

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2000-03-01T02:00:00Z")
        // Philippine date Feb 1: the last day to appeal (Feb 15) is long past at NOW.
        val OLD: OffsetDateTime = OffsetDateTime.parse("2000-02-01T02:00:00Z")
    }
}
