package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
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
import com.quickpitik.service.ai.ImageCheck
import com.quickpitik.service.ai.LabelHit
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.LlmResult
import com.quickpitik.service.relevance.RelevanceAppealService
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceHoldService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor
import org.springframework.transaction.support.TransactionSynchronizationManager
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.Executor
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

// RVF-24 (RVR-7): the whole check on a real database, tick by tick: the real RelevanceCheckService, RelevanceMeasurer,
// notifier, confirmer and payout hold, with fakes for the label client, the storage and the model. A mode change is a new
// service over the same database, because the mode is fixed per process. Each fake records whether a database transaction
// was open when it was called (an assertion thrown there would be swallowed by the measurer's own catch).
// The container is shared: fixtures sit in 1985, before every other class's uploads, and the count trigger is out of reach,
// so no other class's batch starts a check here. Leftovers still get served (this class's earlier tests, other classes'
// cases in flight), so each test counts only its own provider calls (callsOf) and reads only its own rows.
class RelevanceChainIntegrationTest : PostgresIntegrationTest() {
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
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var systemFlags: SystemFlagService
    @Autowired private lateinit var adminFlags: AdminFlagService
    @Autowired private lateinit var adminRelevance: AdminRelevanceService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    // Every provider call with what it was for (a photo's key, which names its batch, or the model's prompt, which names its
    // case), and the ones made with a transaction open. Other tests' batches may be served by a tick too, so each test
    // counts only its own (callsOf).
    private val calls = mutableListOf<Pair<String, String>>()
    private val inTransaction = mutableListOf<String>()
    private var answer: LlmResult = verdict("junk_dump")
    private val publisher: ApplicationEventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)

    private val storage: StorageService = Mockito.mock(StorageService::class.java) { call ->
        when (call.method.name) {
            "getBytes" -> (call.arguments[0] as String).also { seen("storage", it) }.toByteArray()
            "exists" -> true
            else -> Mockito.RETURNS_DEFAULTS.answer(call)
        }
    }

    // A key under relchain/junk/ reads blurry (the blur cut-off is 20); every other photo is a sharp, bright race photo.
    private val labels: RekognitionAiClient = Mockito.mock(RekognitionAiClient::class.java) { call ->
        if (call.method.name != "imageCheck") return@mock Mockito.RETURNS_DEFAULTS.answer(call)
        val key = String(call.arguments[0] as ByteArray)
        seen("labels", key)
        val sharp = if ("/junk/" in key) 10.0 else 50.0
        ImageCheck(sharp, 50.0, sharp, 50.0, listOf(LabelHit("Person", 0.9)))
    }

    private val model: AnthropicClient = Mockito.mock(AnthropicClient::class.java) { call ->
        if (call.method.name != "ask") return@mock Mockito.RETURNS_DEFAULTS.answer(call)
        seen("model", call.arguments[1] as String)
        answer
    }

    private fun seen(provider: String, what: String) {
        synchronized(calls) {
            calls += provider to what
            if (TransactionSynchronizationManager.isActualTransactionActive()) inTransaction += provider
        }
    }

    @BeforeEach fun quiet() {
        // Scans other classes left queued would take this class's turns: the container is shared.
        jdbc.update("UPDATE relevance_cases SET scan_state = 'FINISHED' WHERE scan_state IN ('QUEUED', 'RUNNING')")
    }

    @Test
    fun `a quiet batch goes from due to applied in record-only`() {
        val b = batch()
        val shots = photos(b, junk = 40)
        val service = checks(RelevanceMode.RECORD_ONLY, NOW)

        repeat(5) { service.tick() }

        val c = only(b)
        assertEquals(listOf("QUIET", "RECORD_ONLY", "APPLIED", "RECORD_ONLY", "JUNK_DUMP", "HIDE_ALL", "OVER", 40, 40, 40, 40),
            row(c, "trigger", "started_mode", "stage", "applied_mode", "verdict", "action", "band", "batch_photos", "sample_size", "measured", "junk"))
        assertEquals(40, count("SELECT count(*) FROM relevance_photo_checks WHERE batch_id = ? AND measured_at IS NOT NULL", b.id))
        // The record-only promise: the verdict would hide the batch, and nothing changed for the photographer.
        assertUntouched(b, shots)
        assertEquals(emptyList(), inTransaction)
    }

    @Test
    fun `the same chain in auto-hide hides a junk dump for a photographer who consented and nothing for one who did not`() {
        val yes = batch(terms = "1.7")
        val yesShots = photos(yes, junk = 40)
        val no = batch(terms = "1.0")
        val noShots = photos(no, junk = 40)
        val service = checks(RelevanceMode.AUTO_HIDE, NOW)

        repeat(5) { service.tick() }

        assertEquals(listOf("AUTO_HIDE", "APPLIED", "AUTO_HIDE", "JUNK_DUMP", "HIDE_ALL", false),
            row(only(yes), "started_mode", "stage", "applied_mode", "verdict", "action", "no_consent"))
        assertEquals(listOf("HIDDEN"), yesShots.map(::status).distinct())
        assertEquals(40, count("SELECT count(*) FROM flag_hidden_photos WHERE batch_id = ? AND restored_at IS NULL", yes.id))
        assertNotNull(row(only(yes), "notice_at").single(), "the photographer who consented is told")
        // The consent promise: the same verdict, recorded only.
        assertEquals(listOf("AUTO_HIDE", "APPLIED", "RECORD_ONLY", "JUNK_DUMP", "HIDE_ALL", true),
            row(only(no), "started_mode", "stage", "applied_mode", "verdict", "action", "no_consent"))
        assertUntouched(no, noShots)
    }

    @Test
    fun `no provider is called inside a database transaction`() {
        val b = batch()
        photos(b, junk = 10, good = 30)
        answer = verdict("mixed")
        val service = checks(RelevanceMode.AUTO_HIDE, NOW)

        repeat(5) { service.tick() }

        assertEquals(listOf("APPLIED", "MIXED", "HIDE_JUNK"), row(only(b), "stage", "verdict", "action"))
        val mine = callsOf(b)
        assertTrue(mine.count { it == "storage" } >= 40 && mine.count { it == "labels" } >= 40 && "model" in mine, mine.toString())
        assertEquals(emptyList(), inTransaction)
    }

    @Test
    fun `with a cut-off unset the mode auto-hide runs as record-only, the chain stores raw scores and the band unknown, the verdict can only be fits or can't tell, and nothing hides`() {
        val (b, shots, c) = unsetCutoffChain()

        // Blurry is known (its cut-off is set), so the junk is counted; with the dark cut-off unset the band is not.
        assertEquals(listOf("RECORD_ONLY", "APPLIED", "RECORD_ONLY", "UNKNOWN", "CANT_TELL", "CONTRADICTS", "JUNK_DUMP", 40, 40),
            row(c,"started_mode", "stage", "applied_mode", "band", "verdict", "verdict_note", "model_verdict", "measured", "junk"))
        // The raw scores are kept, so the band can be worked out again later with no AWS call.
        assertEquals(listOf(10.0), jdbc.queryForList("SELECT DISTINCT fg_sharpness FROM relevance_photo_checks WHERE batch_id = ?",
            Double::class.java, b.id))
        assertUntouched(b, shots)

        // Fits is the one other verdict an unknown band allows.
        val fits = batch()
        val fitsShots = photos(fits, junk = 40)
        answer = verdict("fits")
        repeat(5) { checks(RelevanceMode.AUTO_HIDE, NOW, cutoffs = false).tick() }
        assertEquals(listOf("RECORD_ONLY", "UNKNOWN", "FITS", null), row(only(fits), "applied_mode", "band", "verdict", "verdict_note"))
        assertUntouched(fits, fitsShots)
    }

    @Test
    fun `with both cut-offs set the batch is tried again from its stored scores`() {
        val (b, shots, first) = unsetCutoffChain()
        calls.clear()
        val later = checks(RelevanceMode.AUTO_HIDE, NOW.plusMinutes(61))

        repeat(5) { later.tick() }

        val retry = cases.currentFor(b.id)!!.id
        assertEquals(listOf("RETRY", "AUTO_HIDE", "APPLIED", "AUTO_HIDE", "OVER", "JUNK_DUMP", 40, 40),
            row(retry, "trigger", "started_mode", "stage", "applied_mode", "band", "verdict", "measured", "junk"))
        assertEquals(listOf("CLOSED", "SUPERSEDED"), row(first, "stage", "close_reason"))
        // No new label call and no storage read: the band came from the stored scores. One model call.
        assertEquals(listOf("model"), callsOf(b))
        assertEquals(listOf("HIDDEN"), shots.map(::status).distinct())
    }

    @Test
    fun `a model failure ends as can't tell, hides nothing and is tried again after the wait`() {
        val b = batch()
        val shots = photos(b, junk = 40)
        answer = LlmResult.Failed("overloaded")
        val service = checks(RelevanceMode.AUTO_HIDE, NOW)

        repeat(6) { service.tick() }

        val failed = only(b)
        assertEquals(listOf("APPLIED", "AUTO_HIDE", "CANT_TELL", "CALL_FAILED", "NONE", 3),
            row(failed, "stage", "applied_mode", "verdict", "verdict_note", "action", "decide_attempts"))
        assertEquals(3, callsOf(b).count { it == "model" })
        assertUntouched(b, shots)

        // Before the wait nothing starts; after it the batch is checked again on the sample it has.
        answer = verdict("junk_dump")
        repeat(3) { checks(RelevanceMode.AUTO_HIDE, NOW.plusMinutes(30)).tick() }
        assertEquals(listOf(failed), caseIds(b))
        calls.clear()
        repeat(5) { checks(RelevanceMode.AUTO_HIDE, NOW.plusMinutes(61)).tick() }

        val retry = cases.currentFor(b.id)!!.id
        assertEquals(listOf("RETRY", "AUTO_HIDE", "JUNK_DUMP", "HIDE_ALL"), row(retry, "trigger", "applied_mode", "verdict", "action"))
        assertEquals(listOf("CLOSED", "SUPERSEDED"), row(failed, "stage", "close_reason"))
        assertEquals(listOf("model"), callsOf(b))
        assertEquals(listOf("HIDDEN"), shots.map(::status).distinct())
    }

    @Test
    fun `a mode switch with work in flight`() {
        // 1. A check starts in record-only and stops once decided (the next tick would apply it).
        val one = batch()
        val oneShots = photos(one, junk = 40).toMutableList()
        val recordOnly = checks(RelevanceMode.RECORD_ONLY, NOW)
        repeat(3) { recordOnly.tick() }
        val inFlight = only(one)
        assertEquals(listOf("DECIDED", "RECORD_ONLY", "JUNK_DUMP"), row(inFlight, "stage", "started_mode", "verdict"))

        // 2. The next server runs auto-hide: the check applies as it started, record-only (RVF-8).
        checks(RelevanceMode.AUTO_HIDE, NOW).tick()
        assertEquals(listOf("APPLIED", "RECORD_ONLY"), row(inFlight, "stage", "applied_mode"))
        assertUntouched(one, oneShots)

        // 3. A new upload under auto-hide is checked again and hides the batch.
        val t1 = NOW.plusHours(2)
        oneShots += upload(one, t1.minusMinutes(20), junk = 2)
        repeat(5) { checks(RelevanceMode.AUTO_HIDE, t1).tick() }
        val hiding = cases.currentFor(one.id)!!.id
        assertEquals(listOf("AUTO_HIDE", "AUTO_HIDE", "HIDE_ALL", 42), row(hiding, "started_mode", "applied_mode", "action", "batch_photos"))
        assertEquals(listOf("CLOSED", "SUPERSEDED"), row(inFlight, "stage", "close_reason"))
        assertEquals(listOf("HIDDEN"), oneShots.map(::status).distinct())
        assertNotNull(row(hiding, "notice_at").single())

        // 4. The next server runs record-only: the case page and the appeal still work (RVF-14), and the tick hides nothing
        // more, not even a photo uploaded now. A second batch starts its check here and is left decided.
        val t2 = t1.plusHours(1)
        val page = tx.execute { appealService(RelevanceMode.RECORD_ONLY, t2).caseFor(one.photographer, hiding) }!!
        assertEquals(hiding, page.caseId)
        tx.execute { appealService(RelevanceMode.RECORD_ONLY, t2).file(one.photographer, hiding, "These are all mine") }
        assertEquals(listOf("OPEN"), jdbc.queryForList("SELECT status FROM relevance_appeals WHERE case_id = ?", String::class.java, hiding))
        val late = upload(one, t2.minusMinutes(20), junk = 1).single()
        val two = batch(upload = t2.minusMinutes(20))
        val twoShots = photos(two, junk = 40)
        val recordOnlyAgain = checks(RelevanceMode.RECORD_ONLY, t2)
        repeat(3) { recordOnlyAgain.tick() }
        assertEquals("LIVE", status(late))
        assertEquals(42, count("SELECT count(*) FROM flag_hidden_photos WHERE batch_id = ? AND restored_at IS NULL", one.id))
        val twoCase = only(two)
        assertEquals(listOf("DECIDED", "RECORD_ONLY"), row(twoCase, "stage", "started_mode"))

        // 5. The mode off: the tick does nothing and calls no provider.
        val t3 = t2.plusHours(1)
        val before = snapshot(one, two)
        calls.clear()
        repeat(3) { checks(RelevanceMode.OFF, t3).tick() }
        assertEquals(before, snapshot(one, two))
        assertEquals(emptyList(), calls)

        // 6. Back in auto-hide, the first tick hides nothing that started in record-only.
        checks(RelevanceMode.AUTO_HIDE, t3).tick()
        assertEquals(listOf("APPLIED", "RECORD_ONLY", "HIDE_ALL"), row(twoCase, "stage", "applied_mode", "action"))
        assertUntouched(two, twoShots)
        assertEquals(emptyList(), inTransaction)
    }

    // A junk dump checked in auto-hide with the dark cut-off unset: a record-only chain with the band unknown.
    private fun unsetCutoffChain(): Triple<Batch, List<UUID>, UUID> {
        val b = batch()
        val shots = photos(b, junk = 40)
        val service = checks(RelevanceMode.AUTO_HIDE, NOW, cutoffs = false)
        repeat(5) { service.tick() }
        return Triple(b, shots, only(b))
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun props(mode: RelevanceMode, cutoffs: Boolean = true) = RelevanceProperties(mode = mode,
        junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = if (cutoffs) 5.0 else null),
        llm = RelevanceProperties.Llm(apiKey = "test-key"),
        check = RelevanceProperties.Check(maxBatchesPerTick = 10_000, countTrigger = 1_000_000),
        scan = RelevanceProperties.Scan(maxBatchesPerTick = 10_000))

    // The switch and the measurer both find the fake label client through this.
    @Suppress("UNCHECKED_CAST")
    private fun labelProvider() = (Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionAiClient>)
        .also { Mockito.`when`(it.getIfAvailable()).thenReturn(labels) }

    private fun switch(props: RelevanceProperties) =
        RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = true), labelProvider())

    private fun notifier(props: RelevanceProperties, clock: Clock) =
        RelevanceNotifier(cases, decisionLog(), switch(props), props, clock, tx, publisher)

    private fun decisionLog() = AdminDecisionLogService(decisionRows, messages, publisher)

    private fun holdService(props: RelevanceProperties) = RelevanceHoldService(cases, strikes, systemFlags, adminFlags, decisionLog(),
        switch(props), tx, AdminProperties(flagsEnabled = true), users)

    private fun checks(mode: RelevanceMode, at: OffsetDateTime, cutoffs: Boolean = true): RelevanceCheckService {
        val props = props(mode, cutoffs)
        val clock = Clock.fixed(at.toInstant(), at.offset)
        val switch = switch(props)
        // The pool runs each photo on the calling thread, so a transaction the caller held would be seen by the fakes.
        val measurer = RelevanceMeasurer(batches, storage, labelProvider(), Executor { it.run() }, props, clock, tx, SimpleMeterRegistry())
        val notifier = notifier(props, clock)
        return RelevanceCheckService(batches, cases, props, switch, clock, tx, SimpleMeterRegistry(), measurer, model, events, alerts,
            publisher, LEGAL, holds, photos, decisionLog(), IDLE_POOL, notifier,
            RelevanceConfirmer(cases, strikes, alerts, events, notifier, props, publisher), holdService(props))
    }

    private fun appealService(mode: RelevanceMode, at: OffsetDateTime): RelevanceAppealService {
        val props = props(mode)
        val clock = Clock.fixed(at.toInstant(), at.offset)
        val notifier = notifier(props, clock)
        return RelevanceAppealService(cases, appeals, alerts, notifier, switch(props), props, clock, publisher,
            RelevanceConfirmer(cases, strikes, alerts, events, notifier, props, publisher), decisionLog(), holdService(props), tx, adminRelevance)
    }

    private fun verdict(wire: String) = LlmResult.Answer(
        """{"verdict":"$wire","evidence_used":["sample_counts"],"admin_reason":"test"}""", "claude-test")

    // A public shared race's batch whose newest upload version 1 has scored (no copy found), so a check decides at once.
    private fun batch(terms: String = "1.7", upload: OffsetDateTime = UPLOAD): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', ?)",
            "relchain-${UUID.randomUUID()}@test.local", terms,
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Chain Run', DATE '1985-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relchain-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 0, ?, ?)",
            event, photographer, upload, upload,
        )
        val batchId = id("INSERT INTO batch_relevance (event_id, photographer_id, scored_at, scored_upload_at, copy_low) VALUES (?, ?, ?, ?, false)",
            event, photographer, upload, upload)
        return Batch(batchId, event, photographer)
    }

    // `junk` blurry photos then `good` sharp ones, counted on the batch.
    private fun photos(b: Batch, junk: Int = 0, good: Int = 0): List<UUID> {
        val ids = List(junk) { photo(b, "junk") } + List(good) { photo(b, "good") }
        jdbc.update("UPDATE event_photographer SET photo_count = photo_count + ? WHERE event_id = ? AND photographer_id = ?",
            ids.size, b.event, b.photographer)
        return ids
    }

    // A later upload: new photos, and version 1 has scored it.
    private fun upload(b: Batch, at: OffsetDateTime, junk: Int): List<UUID> {
        val ids = photos(b, junk = junk)
        jdbc.update("UPDATE event_photographer SET last_upload_at = ? WHERE event_id = ? AND photographer_id = ?", at, b.event, b.photographer)
        jdbc.update("UPDATE batch_relevance SET scored_at = ?, scored_upload_at = ? WHERE id = ?", at, at, b.id)
        return ids
    }

    private fun photo(b: Batch, kind: String): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'relchain/thumb', 'relchain/wm', 125, 'LIVE')",
        b.event, b.photographer, "relchain/${b.id}/$kind/${UUID.randomUUID()}",
    )

    // The providers called for this batch's photos or cases since `calls` was last cleared, in order.
    private fun callsOf(b: Batch): List<String> {
        val names = listOf(b.id.toString()) + caseIds(b).map { it.toString() }
        return synchronized(calls) { calls.filter { (_, what) -> names.any { it in what } }.map { it.first } }
    }

    // No photo hidden, no hold, flag, message, automatic decision or e-mail for this batch's photographer.
    private fun assertUntouched(b: Batch, shots: List<UUID>) {
        assertEquals(listOf("LIVE"), shots.map(::status).distinct())
        assertEquals(0, count("SELECT count(*) FROM flag_hidden_photos WHERE batch_id = ?", b.id))
        assertEquals(0, count("SELECT count(*) FROM flags WHERE target_id = ?", b.photographer))
        assertEquals(0, count("SELECT count(*) FROM photographer_messages WHERE photographer_id = ?", b.photographer))
        assertEquals(0, count("SELECT count(*) FROM admin_decision_log WHERE target_user_id = ?", b.photographer))
        // No stored message, so no RelevanceNoticeStored for one: nothing for the e-mail listener to send.
    }

    private fun snapshot(vararg bs: Batch): List<Any?> = bs.flatMap { b ->
        listOf(
            jdbc.queryForList("SELECT id, stage, applied_mode, action, notice_at, closed_at FROM relevance_cases WHERE batch_id = ? ORDER BY id", b.id),
            jdbc.queryForList("SELECT p.id, p.status FROM photos p WHERE p.event_id = ? AND p.photographer_id = ? ORDER BY p.id", b.event, b.photographer),
            jdbc.queryForList("SELECT photo_id, held_by, restored_at FROM flag_hidden_photos WHERE batch_id = ? ORDER BY photo_id, held_by", b.id),
            jdbc.queryForList("SELECT id FROM photographer_messages WHERE photographer_id = ? ORDER BY id", b.photographer),
            jdbc.queryForList("SELECT check_claim, checked_upload_at FROM batch_relevance WHERE id = ?", b.id),
        )
    }

    private fun caseIds(b: Batch): List<UUID> =
        jdbc.queryForList("SELECT id FROM relevance_cases WHERE batch_id = ? ORDER BY created_at", UUID::class.java, b.id)

    private fun only(b: Batch): UUID = caseIds(b).single()

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun count(sql: String, vararg params: Any): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("1985-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
        val LEGAL = LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7")
        val IDLE_POOL = ThreadPoolTaskExecutor().apply { initialize() }
    }
}
