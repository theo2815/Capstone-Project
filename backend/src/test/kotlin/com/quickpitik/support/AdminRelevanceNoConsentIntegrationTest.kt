package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.AdminFlagService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceHoldService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceNotices
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.core.task.SyncTaskExecutor
import org.springframework.http.HttpStatus
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// RV-4.5a (follow-up of RV-L7): a photographer without consent gets a record-only case with the note NO_CONSENT. After
// reviewing it an admin may hide the batch (when its verdict would have hidden it) and, on a Copied case, hold the
// payouts. Both act in auto-hide only. The container is shared, so fixtures sit in 2011 and assertions look only at this
// class's own rows.
class AdminRelevanceNoConsentIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var strikes: RelevanceStrikeRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var decisions: AdminDecisionLogService
    @Autowired private lateinit var systemFlags: SystemFlagService
    @Autowired private lateinit var adminFlags: AdminFlagService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    private lateinit var adminId: UUID

    @BeforeEach fun setUp() {
        adminId = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Admin', 'ADMIN')",
            "relnocon-admin-${UUID.randomUUID()}@test.local")
    }

    @Test
    fun `an admin can hide the batch of a no-consent case that would have hidden it`() {
        for (verdict in listOf("JUNK_DUMP", "WRONG_EVENT", "COPIED")) {
            val b = batch()
            val c = noConsent(b, verdict)
            val photosOf = List(2) { live(b) }

            hide(c, "Checked the samples")
            repeat(2) { checks().reconcileHolds() }

            assertEquals(listOf("APPLIED", verdict, "AUTO_HIDE", "HIDE_ALL", "ADMIN_HIDE", 2, "NONE"),
                row(c, "stage", "verdict", "applied_mode", "action", "verdict_note", "hidden_photos", "scan_state"), verdict)
            assertEquals(listOf("HIDDEN"), photosOf.map(::status).distinct(), verdict)
            assertEquals(2L, holds.countLiveHolds(b.id, "ADMIN"), verdict)
            assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"), verdict)
            assertEquals(listOf(listOf(adminId, "relevance_admin_hidden", "Checked the samples", c.toString())), decisionsOf(b), verdict)
        }
    }

    @Test
    fun `an admin's hide of a no-consent case never counts toward the ladder`() {
        val b = batch()
        val c = noConsent(b, "JUNK_DUMP")
        hide(c)
        jdbc.update("UPDATE relevance_cases SET notice_at = ? WHERE id = ?", OffsetDateTime.now().minusDays(60), c)

        assertEquals(false, cases.findConfirmDue(OffsetDateTime.now(), 10_000).contains(c))
        assertEquals(0, tx.execute { cases.stampConfirmed(c, OffsetDateTime.now()) })
        assertEquals(0L, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE photographer_id = ?", Long::class.java, b.photographer))
    }

    @Test
    fun `a no-consent case that would not have hidden the batch cannot be hidden by an admin`() {
        val mixed = noConsent(batch(), "MIXED")
        val fits = noConsent(batch(), "FITS")
        val unsure = noConsent(batch(), "CANT_TELL")
        // A record-only case without the NO_CONSENT note (the mode was record-only): never acted on (P18).
        val recordOnly = case(batch(), "JUNK_DUMP", mode = "RECORD_ONLY")
        val closed = noConsent(batch(), "COPIED", stage = "CLOSED")

        for (c in listOf(mixed, fits, unsure, recordOnly, closed)) {
            val ex = assertThrows<ApiException> { hide(c) }
            assertEquals(HttpStatus.CONFLICT to ErrorCodes.INVALID_STATE_TRANSITION, ex.status to ex.code)
            assertEquals("RECORD_ONLY", row(c, "applied_mode").single())
        }
    }

    @Test
    fun `a withdrawn photographer's no-consent case can still be hidden by an admin`() {
        val b = batch()
        jdbc.update("UPDATE users SET relevance_consent_withdrawn_at = ? WHERE id = ?", NOW, b.photographer)
        val c = noConsent(b, "WRONG_EVENT")
        val photo = live(b)

        hide(c)
        checks().reconcileHolds()

        assertEquals("HIDDEN", status(photo))
        assertEquals(1L, holds.countLiveHolds(b.id, "ADMIN"))
    }

    @Test
    fun `an admin hide of a no-consent case sends the V2-T24 note and starts the appeal clock`() {
        val b = batch()
        val c = noConsent(b, "COPIED")
        hide(c)

        notifier().notifyOwed()

        val note = RelevanceNotices.adminHideAfterCheck("No Consent Run", "") // only the title and the fixed opening are compared
        val message = jdbc.queryForMap("SELECT title, body FROM photographer_messages WHERE notice_key = ?", "relevance:$c:verdict")
        assertEquals(note.title, message["title"])
        assertEquals(true, (message["body"] as String).startsWith("Our automatic check found a problem with your photos in No Consent Run."))
        assertNotNull(row(c, "notice_at").single())
    }

    @Test
    fun `the automatic hold does not fire on an admin-hidden no-consent Copied case`() {
        val b = batch()
        val c = noConsent(b, "COPIED")
        hide(c)

        assertEquals(false, cases.findHoldDue(10_000).contains(c))
        payoutHold().place(c)

        assertNull(holdFlagOf(c))
        assertEquals(emptyList(), flagsOf(b))
    }

    @Test
    fun `an admin can hold the payouts of a no-consent Copied case and End hold lifts it`() {
        for (hiddenFirst in listOf(false, true)) {
            val b = batch()
            val c = noConsent(b, "COPIED")
            if (hiddenFirst) hide(c)

            val dto = tx.execute { service().holdPayouts(adminId, c, "Matches another set") }!!

            assertEquals(c, dto.caseId)
            val flag = flagsOf(b).single()
            assertEquals(listOf<Any?>("open", true, adminId), listOf(flag["status"], flag["payout_hold_at"] != null, flag["payout_hold_by"]))
            assertEquals(flag["id"], holdFlagOf(c))
            assertEquals(listOf("Payouts on hold"), titles(b).filter { it != "Your photos in No Consent Run are off sale" })
            assertEquals(true, decisionsOf(b).any { it == listOf(adminId, "relevance_payouts_held", "Matches another set", c.toString()) })
            assertNotNull(tx.execute { service().case(c) }!!.payoutHoldAt)

            tx.execute { service().endHold(adminId, c) }

            assertEquals("resolved", flagsOf(b).single()["status"])
            assertEquals(listOf("Payouts on hold", "Payout hold lifted"), titles(b).filter { it != "Your photos in No Consent Run are off sale" })
            val after = tx.execute { service().case(c) }!!
            assertEquals(null to true, after.payoutHoldAt to after.payoutHoldPlaced, "an ended hold is not placed again")
        }
    }

    // Review of PR #174: a hold the admin placed on a case still in record-only goes over to the case that replaces it, so
    // the page (which lists current cases) can still end it, and the first consented check places no second hold.
    @Test
    fun `an admin payout hold stays liftable after a later check replaces the case`() {
        for (consentLater in listOf(false, true)) {
            val b = batch()
            val c = noConsent(b, "COPIED")
            tx.execute { service().holdPayouts(adminId, c, null) }
            val flag = holdFlagOf(c)
            // Without consent a different verdict replaces the record-only case; with consent the same verdict does (P18).
            if (consentLater) jdbc.update("UPDATE users SET terms_version = '1.7' WHERE id = ?", b.photographer)
            val next = decided(b, if (consentLater) "COPIED" else "FITS")

            checks().applyCase(next, OffsetDateTime.now())

            assertEquals(listOf("CLOSED", "SUPERSEDED"), row(c, "stage", "close_reason"), "$consentLater")
            assertEquals(listOf("APPLIED", flag), row(next, "stage", "hold_flag_id"), "$consentLater")
            assertEquals(1, flagsOf(b).size, "one hold only")
            assertNotNull(tx.execute { service().case(next) }!!.payoutHoldAt, "$consentLater")

            tx.execute { service().endHold(adminId, next) }

            assertEquals("resolved", flagsOf(b).single()["status"], "$consentLater")
        }
    }

    @Test
    fun `hold-payouts refuses a case that is not a no-consent Copied case, or already holds`() {
        val junk = noConsent(batch(), "JUNK_DUMP")
        val consented = case(batch(), "COPIED")
        val recordOnly = case(batch(), "COPIED", mode = "RECORD_ONLY")
        val canTell = case(batch(), "CANT_TELL").also { hide(it) }
        val closed = noConsent(batch(), "COPIED", stage = "CLOSED")
        val held = noConsent(batch(), "COPIED").also { c -> tx.execute { service().holdPayouts(adminId, c, null) } }

        for (c in listOf(junk, consented, recordOnly, canTell, closed, held)) {
            val ex = assertThrows<ApiException> { tx.execute { service().holdPayouts(adminId, c, null) } }
            assertEquals(HttpStatus.CONFLICT to ErrorCodes.INVALID_STATE_TRANSITION, ex.status to ex.code)
        }
        for (c in listOf(junk, consented, recordOnly, canTell, closed)) assertNull(holdFlagOf(c))
    }

    @Test
    fun `with the mode record-only both actions answer 409`() {
        val b = batch()
        val c = noConsent(b, "COPIED")

        for (mode in listOf(RelevanceMode.OFF, RelevanceMode.RECORD_ONLY)) {
            for (action in listOf<(AdminRelevanceService) -> Any>({ it.hideBatch(adminId, c, null) }, { it.holdPayouts(adminId, c, null) })) {
                val ex = assertThrows<ApiException> { tx.execute { action(service(mode)) } }
                assertEquals(HttpStatus.CONFLICT to ErrorCodes.RELEVANCE_ACTIONS_OFF, ex.status to ex.code, "$mode")
            }
        }
        assertEquals(listOf("RECORD_ONLY", "HIDE_ALL", "NO_CONSENT", null), row(c, "applied_mode", "action", "verdict_note", "hold_flag_id"))
        assertEquals(emptyList(), flagsOf(b))
        assertEquals(emptyList(), decisionsOf(b))
    }

    private fun hide(caseId: UUID, note: String? = null) = tx.execute { service().hideBatch(adminId, caseId, note) }!!

    private fun props(mode: RelevanceMode) = RelevanceProperties(mode = mode,
        junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0), check = RelevanceProperties.Check(maxBatchesPerTick = 10_000))

    private fun switch(mode: RelevanceMode) = RelevanceSwitch(props(mode), AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false))

    private fun payoutHold(mode: RelevanceMode = RelevanceMode.AUTO_HIDE) = RelevanceHoldService(cases, strikes, systemFlags, adminFlags,
        decisions, switch(mode), tx, AdminProperties(flagsEnabled = true), users)

    private fun service(mode: RelevanceMode = RelevanceMode.AUTO_HIDE) = AdminRelevanceService(cases, alerts, batches, props(mode),
        Mockito.mock(StorageService::class.java), StorageProperties(), Clock.systemUTC(), holds, photos, decisions, tx, switch(mode),
        strikes, payoutHold(mode), users)

    private fun checks(): RelevanceCheckService {
        val mode = RelevanceMode.AUTO_HIDE
        return RelevanceCheckService(batches, cases, props(mode), switch(mode), Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"),
            holds, photos, decisions, SyncTaskExecutor(), Mockito.mock(RelevanceNotifier::class.java),
            Mockito.mock(RelevanceConfirmer::class.java), payoutHold())
    }

    private fun notifier() = RelevanceNotifier(cases, decisions, switch(RelevanceMode.AUTO_HIDE), props(RelevanceMode.AUTO_HIDE),
        Clock.systemUTC(), tx, Mockito.mock(ApplicationEventPublisher::class.java))

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun batch(): Batch {
        val photographer = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "relnocon-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'No Consent Run', DATE '2011-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relnocon-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 40, ?, ?)",
            event, photographer, NOW, NOW)
        val batchId = id("INSERT INTO batch_relevance (event_id, photographer_id, scored_at, scored_upload_at, copy_low, checked_upload_at, checked_photos) " +
            "VALUES (?, ?, ?, ?, false, ?, 40)", event, photographer, NOW, NOW, NOW)
        return Batch(batchId, event, photographer)
    }

    // A case applied as record-only because the photographer had not consented (RV-2.2a), with the action auto-hide would have taken.
    private fun noConsent(b: Batch, verdict: String, stage: String = "APPLIED") =
        case(b, verdict, mode = "RECORD_ONLY", note = "NO_CONSENT", stage = stage)

    private fun case(b: Batch, verdict: String, mode: String = "AUTO_HIDE", note: String? = null, stage: String = "APPLIED"): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, verdict_note, copy_low, " +
            "decided_at, applied_mode, action, applied_at, closed_at, close_reason) VALUES (?, 'QUIET', ?, 40, ?, 40, 0, ?, ?, false, ?, ?, ?, ?, ?, ?)",
        b.id, NOW, stage, verdict, note, NOW, mode, action(verdict), NOW,
        NOW.takeIf { stage == "CLOSED" }, "SUPERSEDED".takeIf { stage == "CLOSED" })

    // A case waiting for the apply step.
    private fun decided(b: Batch, verdict: String): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, copy_low, decided_at) " +
            "VALUES (?, 'QUIET', ?, 40, 'DECIDED', 40, 0, ?, false, ?)", b.id, NOW, verdict, NOW)

    private fun action(verdict: String) = when (verdict) {
        "MIXED" -> "HIDE_JUNK"
        "JUNK_DUMP", "WRONG_EVENT", "COPIED" -> "HIDE_ALL"
        else -> "NONE"
    }

    private fun live(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'relnocon/thumb', 'relnocon/wm', 125, 'LIVE')",
        b.event, b.photographer, "relnocon/${UUID.randomUUID()}")

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun holdFlagOf(caseId: UUID): UUID? =
        jdbc.queryForObject("SELECT hold_flag_id FROM relevance_cases WHERE id = ?", UUID::class.java, caseId)

    private fun flagsOf(b: Batch): List<Map<String, Any?>> = jdbc.queryForList(
        "SELECT id, status, payout_hold_at, payout_hold_by FROM flags WHERE target_id = ? AND reason = 'batch_relevance_low'", b.photographer)

    private fun titles(b: Batch): List<String> =
        jdbc.queryForList("SELECT title FROM photographer_messages WHERE photographer_id = ? ORDER BY created_at", String::class.java, b.photographer)

    private fun decisionsOf(b: Batch): List<List<Any?>> =
        jdbc.queryForList("SELECT admin_id, decision, reason, meta->>'caseId' AS case_id FROM admin_decision_log WHERE target_user_id = ? ORDER BY decided_at",
            b.photographer).map { listOf(it["admin_id"], it["decision"], it["reason"], it["case_id"]) }
            .filter { it[1] != "payout_hold_confirmed" }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2011-03-04T05:06:07Z")
    }
}
