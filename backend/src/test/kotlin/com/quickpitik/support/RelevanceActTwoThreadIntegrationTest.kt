package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
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
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.AdminFlagService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceAppealService
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceHoldService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceNoticeStored
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// RVF-23 (RVR-7): each keyed step that acts (hide, note, confirm, payout hold, appeal, scan, restore, decide), run by two
// threads at once on a real database; one wins and the other changes nothing. Same shape as
// RelevanceCheckTwoThreadIntegrationTest: the first call's transaction is held open until Postgres shows the second waiting
// on it. The scan's guard is the batch claim, taken and committed before the measuring, so there the second copy runs
// while the first is inside the measurer. Fixtures sit in 2012; assertions look only at this class's own rows.
class RelevanceActTwoThreadIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var appeals: RelevanceAppealRepository
    @Autowired private lateinit var strikes: RelevanceStrikeRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var messages: PhotographerMessageRepository
    @Autowired private lateinit var decisionRows: AdminDecisionLogRepository
    @Autowired private lateinit var decisionLog: AdminDecisionLogService
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var systemFlags: SystemFlagService
    @Autowired private lateinit var adminFlags: AdminFlagService
    @Autowired private lateinit var adminRelevance: AdminRelevanceService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `two copies hiding one chunk make one hold per photo`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP", "HIDE_ALL")
        val all = List(4) { photo(b) }

        val hidden = race({ hideChunk(b, c, NOW) }, { tx.execute { hideChunk(b, c, LATER) }!! })

        // The loser meets the winner's rows on the unique key and arms them again: same photos, nothing more hidden.
        assertEquals(4 to 4, hidden.first)
        assertEquals(4 to 0, hidden.second)
        val rows = jdbc.queryForList("SELECT photo_id FROM flag_hidden_photos WHERE batch_id = ?", UUID::class.java, b.id)
        assertEquals(4, rows.size)
        assertEquals(all.toSet(), rows.toSet())
        assertEquals(listOf("HIDDEN"), all.map(::status).distinct())
    }

    @Test
    fun `two copies sending one note send it once`() {
        // Other classes' owed notes first, so the race below acts on this case alone.
        notifier(NOW, Mockito.mock(ApplicationEventPublisher::class.java)).notifyOwed()
        val b = batch()
        val c = case(b, "JUNK_DUMP", "HIDE_ALL")
        val firstPush = Mockito.mock(ApplicationEventPublisher::class.java)
        val secondPush = Mockito.mock(ApplicationEventPublisher::class.java)

        race({ notifier(NOW, firstPush).notifyOwed() }, { notifier(LATER, secondPush).notifyOwed() })

        val sent = jdbc.queryForList("SELECT id FROM photographer_messages WHERE notice_key = ?", UUID::class.java, "relevance:$c:verdict")
        assertEquals(1, sent.size)
        // The appeal clock starts once, at the winner's time; only the winner hands the note to the e-mail.
        assertEquals(listOf(NOW.toInstant()), row("relevance_cases", c, "notice_at"))
        assertEquals(listOf(RelevanceNoticeStored(sent.single())), stored(firstPush, sent.single()))
        assertEquals(emptyList(), stored(secondPush, sent.single()))
    }

    @Test
    fun `two copies confirming one case write one strike`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP", "HIDE_ALL")
        val confirmer = confirmer()

        val confirmed = race({ confirmer.confirm(c, NOW) }, { tx.execute { confirmer.confirm(c, LATER) }!! })

        assertEquals(true to false, confirmed)
        assertEquals(listOf(NOW.toInstant()), row("relevance_cases", c, "confirmed_at"))
        assertEquals(1, count("SELECT count(*) FROM relevance_strikes WHERE case_id = ?", c))
    }

    @Test
    fun `two cases of one photographer confirmed at once count as two`() {
        val one = batch()
        val two = batch(one.photographer)
        val first = case(one, "JUNK_DUMP", "HIDE_ALL")
        val second = case(two, "COPIED", "HIDE_ALL")
        val confirmer = confirmer()

        val confirmed = race({ confirmer.confirm(first, NOW) }, { tx.execute { confirmer.confirm(second, LATER) }!! })

        assertEquals(true to true, confirmed)
        assertEquals(2, count("SELECT count(*) FROM relevance_strikes WHERE photographer_id = ?", one.photographer))
        // The second count waited for the first strike, so it saw two: the second-case note goes out once, for that case.
        assertEquals(listOf("relevance:strike2:$second"), jdbc.queryForList(
            "SELECT notice_key FROM photographer_messages WHERE photographer_id = ? AND notice_key LIKE 'relevance:strike2:%'",
            String::class.java, one.photographer))
    }

    @Test
    fun `two copies placing one hold make one flag`() {
        val b = batch()
        val c = case(b, "COPIED", "HIDE_ALL")
        val hold = RelevanceHoldService(cases, strikes, systemFlags, adminFlags, decisionLog, switch(), tx, AdminProperties(flagsEnabled = true), users)

        race({ hold.place(c) }, { hold.place(c) })

        val flags = jdbc.queryForList("SELECT id FROM flags WHERE target_id = ? AND payout_hold_at IS NOT NULL", UUID::class.java, b.photographer)
        assertEquals(1, flags.size)
        assertEquals(flags, row("relevance_cases", c, "hold_flag_id"))
    }

    @Test
    fun `two appeals on one batch at once - one is filed`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP", "HIDE_ALL", noticeAt = NOW)
        val service = appealService()

        val (filed, second) = race({ service.file(b.photographer, c, "These are mine") },
            { runCatching { tx.execute { service.file(b.photographer, c, "Mine as well") }!! } })

        assertEquals(c, filed.caseId)
        assertEquals(ErrorCodes.APPEAL_ALREADY_FILED, (second.exceptionOrNull() as ConflictException).code)
        assertEquals(listOf("These are mine"), jdbc.queryForList("SELECT reason FROM relevance_appeals WHERE batch_id = ?", String::class.java, b.id))
        assertEquals(1, count("SELECT count(*) FROM relevance_alerts WHERE case_id = ? AND kind = 'APPEAL'", c))
    }

    @Test
    fun `two copies taking one scan turn - one takes it`() {
        val b = batch()
        val c = case(b, "FITS", "NONE", scan = "QUEUED")
        repeat(3) { photo(b) }
        val measuring = CountDownLatch(1)
        val release = CountDownLatch(1)
        val turns = AtomicInteger()
        // Other classes' queued scans may be served too; only this batch's measuring waits.
        val measurer = Mockito.mock(RelevanceMeasurer::class.java) { call ->
            if (call.method.name != "measure" || call.arguments[0] != b.id) return@mock 0
            turns.incrementAndGet()
            measuring.countDown()
            check(release.await(10, TimeUnit.SECONDS))
            (call.arguments[2] as List<*>).size
        }
        val pool = Executors.newSingleThreadExecutor()
        try {
            val first = pool.submit { checks(NOW, measurer).scanTurns() }
            assertTrue(measuring.await(10, TimeUnit.SECONDS), "the first copy must reach the measuring")
            // A second copy's whole scan step, while the first measures with the claim held and no transaction open.
            checks(LATER, measurer).scanTurns()
            release.countDown()
            first.get(10, TimeUnit.SECONDS)
        } finally {
            release.countDown()
            pool.shutdownNow()
        }

        assertEquals(1, turns.get())
        assertEquals(listOf("RUNNING", NOW.toInstant(), null), row("relevance_cases", c, "scan_state", "scan_served_at", "scan_finished_at"))
        assertEquals(listOf(null), row("batch_relevance", b.id, "check_claim"))
    }

    @Test
    fun `an admin restore and the tick's hide of one batch both finish`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP", "HIDE_ALL")
        // Four photos an admin hid, whose automatic hold the tick arms again, and two the tick hides for the first time.
        val armed = List(4) { photo(b, "HIDDEN").also { p -> hold(b, c, p, "ADMIN"); hold(b, c, p, "AUTO", restoredBy = "AUTO") } }
        val fresh = List(2) { photo(b) }

        val (hidden, restored) = race({ hideChunk(b, c, NOW) }, { adminRelevance.restore(admin(), c, null).restoredPhotos })

        assertEquals(6 to 2, hidden)
        // The admin's four, then the six automatic holds the hide armed meanwhile: nothing stays hidden after the restore.
        assertEquals(10, restored)
        assertEquals(0, count("SELECT count(*) FROM flag_hidden_photos WHERE batch_id = ? AND restored_at IS NULL", b.id))
        assertEquals(listOf("LIVE"), (armed + fresh).map(::status).distinct())
        assertEquals(listOf("CLOSED", "RESTORED_BY_ADMIN"), row("relevance_cases", c, "stage", "close_reason"))
    }

    @Test
    fun `two admins deciding one appeal - one decision stands`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP", "HIDE_ALL", noticeAt = NOW)
        val appeal = id("INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason) VALUES (?, ?, ?, 'These are mine')",
            c, b.id, b.photographer)
        val (winner, other) = admin() to admin()
        jdbc.update("UPDATE relevance_appeals SET evidence_opened_at = ?, evidence_opened_by = ? WHERE id = ?", NOW, winner, appeal)
        val service = appealService()

        // The other admin does what the page does: open the evidence, then decide.
        val (_, second) = race({ service.decide(winner, appeal, "WON", null) }, {
            runCatching { tx.execute { service.evidenceOpened(other, appeal) }; service.decide(other, appeal, "LOST", null) }
        })

        assertEquals(ErrorCodes.APPEAL_ALREADY_DECIDED, (second.exceptionOrNull() as ApiException).code)
        assertEquals(listOf("WON", winner, winner), row("relevance_appeals", appeal, "status", "decided_by", "evidence_opened_by"))
        assertEquals(0, count("SELECT count(*) FROM relevance_strikes WHERE case_id = ?", c))
        assertEquals(listOf("relevance:$appeal:won"), jdbc.queryForList(
            "SELECT notice_key FROM photographer_messages WHERE notice_key LIKE ?", String::class.java, "relevance:$appeal:%"))
    }

    // The first call runs inside a transaction that stays open until the second is seen waiting on it; then it commits.
    private fun <A : Any, B> race(first: () -> A, second: () -> B): Pair<A, B> {
        val pool = Executors.newFixedThreadPool(2)
        val holding = CountDownLatch(1)
        val release = CountDownLatch(1)
        val holderPid = CompletableFuture<Int>()
        try {
            val winner = pool.submit<A> {
                tx.execute {
                    holderPid.complete(pid())
                    first().also {
                        holding.countDown()
                        check(release.await(10, TimeUnit.SECONDS))
                    }
                }!!
            }
            // A first call that threw shows its own error, not a timeout.
            if (!holding.await(10, TimeUnit.SECONDS)) winner.get(1, TimeUnit.SECONDS)
            val loser = pool.submit<B> { second() }
            val holder = holderPid.get(10, TimeUnit.SECONDS)
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
            var blocked = false
            while (!blocked && System.nanoTime() < deadline) {
                blocked = jdbc.queryForObject("SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE ? = ANY(pg_blocking_pids(pid)))",
                    Boolean::class.java, holder)!!
                if (!blocked) Thread.sleep(10)
            }
            assertTrue(blocked, "the second call must wait for the first")
            release.countDown()
            return winner.get(10, TimeUnit.SECONDS) to loser.get(10, TimeUnit.SECONDS)
        } finally {
            release.countDown()
            pool.shutdownNow()
        }
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun props() = RelevanceProperties(mode = RelevanceMode.AUTO_HIDE, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0),
        check = RelevanceProperties.Check(maxBatchesPerTick = 10_000), scan = RelevanceProperties.Scan(maxBatchesPerTick = 10_000))

    private fun switch(): RelevanceSwitch {
        @Suppress("UNCHECKED_CAST")
        val provider = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
            .also { Mockito.`when`(it.getIfAvailable()).thenReturn(Mockito.mock(RekognitionAiClient::class.java)) }
        return RelevanceSwitch(props(), AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true), provider)
    }

    private fun notifier(at: OffsetDateTime, publisher: ApplicationEventPublisher) = RelevanceNotifier(cases,
        AdminDecisionLogService(decisionRows, messages, publisher), switch(), props(), Clock.fixed(at.toInstant(), at.offset), tx, publisher)

    private fun confirmer(): RelevanceConfirmer {
        val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        return RelevanceConfirmer(cases, strikes, alerts, events, notifier(NOW, publisher), props(), publisher)
    }

    // Auto-hide, so a won appeal lowers the case (the tick restores) and no restore runs in the request.
    private fun appealService(): RelevanceAppealService {
        val publisher = Mockito.mock(ApplicationEventPublisher::class.java)
        val notifier = notifier(LATER, publisher)
        return RelevanceAppealService(cases, appeals, alerts, notifier, switch(), props(), Clock.fixed(LATER.toInstant(), LATER.offset),
            publisher, RelevanceConfirmer(cases, strikes, alerts, events, notifier, props(), publisher), decisionLog,
            Mockito.mock(RelevanceHoldService::class.java), tx, adminRelevance)
    }

    private fun checks(at: OffsetDateTime, measurer: RelevanceMeasurer) = RelevanceCheckService(batches, cases, props(), switch(),
        Clock.fixed(at.toInstant(), at.offset), tx, SimpleMeterRegistry(), measurer, Mockito.mock(AnthropicClient::class.java), events, alerts,
        Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"),
        holds, photos, decisionLog, IDLE_POOL, Mockito.mock(RelevanceNotifier::class.java), Mockito.mock(RelevanceConfirmer::class.java),
        Mockito.mock(RelevanceHoldService::class.java))

    // The tick's hide chunk (RelevanceCheckService.hide): hold rows, then the photos locked in id order and hidden.
    private fun hideChunk(b: Batch, c: UUID, at: OffsetDateTime): Pair<Int, Int> {
        val ids = holds.hideChunk(b.id, c, b.event, b.photographer, "AUTO", at, 1000)
        if (ids.isEmpty()) return 0 to 0
        photos.lockIdsForUpdate(ids)
        return ids.size to holds.hidePhotos(ids)
    }

    private fun stored(publisher: ApplicationEventPublisher, messageId: UUID) = Mockito.mockingDetails(publisher).invocations
        .map { it.arguments.single() }.filterIsInstance<RelevanceNoticeStored>().filter { it.messageId == messageId }

    // A public race's batch; a second batch of the same photographer when `photographer` is given.
    private fun batch(photographer: UUID? = null): Batch {
        val owner = photographer ?: id(
            "INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "relact-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Act Run', DATE '2012-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relact-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 40, ?, ?)",
            event, owner, UPLOAD, UPLOAD,
        )
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, owner), event, owner)
    }

    // A current case applied in auto-hide.
    private fun case(b: Batch, verdict: String, action: String, scan: String = "NONE", noticeAt: OffsetDateTime? = null): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, band, verdict, decided_at, " +
            "started_mode, applied_mode, action, applied_at, scan_state, notice_at) " +
            "VALUES (?, 'QUIET', ?, 40, 'APPLIED', 40, 30, 'MIXED', ?, ?, 'AUTO_HIDE', 'AUTO_HIDE', ?, ?, ?, ?)",
        b.id, UPLOAD, verdict, UPLOAD, action, UPLOAD, scan, noticeAt,
    )

    private fun photo(b: Batch, status: String = "LIVE"): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'relact/thumb', 'relact/wm', 125, ?)",
        b.event, b.photographer, "relact/${UUID.randomUUID()}", status,
    )

    private fun hold(b: Batch, c: UUID, photo: UUID, heldBy: String, restoredBy: String? = null) = jdbc.update(
        "INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at, restored_at, restored_by) VALUES (?, ?, ?, ?, ?, ?, ?)",
        b.id, photo, heldBy, c, UPLOAD, restoredBy?.let { UPLOAD }, restoredBy,
    )

    private fun admin(): UUID = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Adm', 'ADMIN')",
        "relact-admin-${UUID.randomUUID()}@test.local")

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    // Timestamps come back as Instants, so they compare with the fixtures' times.
    private fun row(table: String, id: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM $table WHERE id = ?", id)
            .let { r -> columns.map { r[it].let { v -> if (v is java.sql.Timestamp) v.toInstant() else v } } }

    private fun pid(): Int = jdbc.queryForObject("SELECT pg_backend_pid()", Int::class.java)!!

    private fun count(sql: String, vararg params: Any): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2012-03-04T05:06:07.123456Z")
        val LATER: OffsetDateTime = NOW.plusSeconds(1)
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(30)
        val IDLE_POOL = ThreadPoolTaskExecutor().apply { initialize() }
    }
}
