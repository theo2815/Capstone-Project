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
import com.quickpitik.repository.RelevanceAppealRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.repository.RelevanceStrikeRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.admin.AdminFlagService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.earnings.PayoutRequestService
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceAppealService
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceHoldService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.core.task.SyncTaskExecutor
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// RV-4.2 (plan part 2, A.8, O1): a Copied case applied in auto-hide holds the photographer's payouts with no admin, on
// the photographer's open relevance flag (a version 1 one included) or a new system flag. The case's own note (V2-T14)
// and its COPIED alert already tell the photographer and the admins, so placing the hold adds neither. The hold ends
// through AdminFlagService (an admin's "End hold" resolves the flag; an admin restore or a won appeal dismisses it),
// which sends the existing "Payout hold lifted" note. Assertions look only at this class's own photographers.
class RelevanceHoldIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var strikes: RelevanceStrikeRepository
    @Autowired private lateinit var batchRepo: BatchRelevanceRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var appeals: RelevanceAppealRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photoRepo: PhotoRepository
    @Autowired private lateinit var systemFlags: SystemFlagService
    @Autowired private lateinit var adminFlags: AdminFlagService
    @Autowired private lateinit var payouts: PayoutRequestService
    @Autowired private lateinit var decisionLog: AdminDecisionLogService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    private val publisher = Mockito.mock(ApplicationEventPublisher::class.java)

    @Test
    fun `with the mode off nothing runs`() {
        val b = batch()
        val c = applied(b)

        hold(RelevanceMode.OFF).place(c)

        assertEquals(emptyList(), flagsOf(b))
        assertNull(holdFlagOf(c))
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val b = batch()
        val c = applied(b)

        hold(RelevanceMode.RECORD_ONLY).place(c)

        assertEquals(emptyList(), flagsOf(b))
        assertEquals(0, messages(b))
        assertEquals(0, decisions(b))
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch()
        val c = applied(b, mode = "RECORD_ONLY")

        hold().place(c)

        assertEquals(emptyList(), flagsOf(b))
        assertNull(holdFlagOf(c))
    }

    @Test
    fun `a copied case in auto-hide places the hold with no admin and the payout request is refused with PAYOUT_ON_HOLD`() {
        val b = batch()
        val c = decided(b)

        checks().applyCase(c, OffsetDateTime.now())

        val flag = flagsOf(b).single()
        assertEquals(listOf<Any?>("open", "user", null, true, null), listOf(flag["status"], flag["target_kind"],
            flag["reporter_id"], flag["payout_hold_at"] != null, flag["payout_hold_by"]))
        assertEquals(flag["id"], holdFlagOf(c))
        val refused = assertFailsWith<ApiException> { tx.execute { payouts.request(b.photographer) } }
        assertEquals(ErrorCodes.PAYOUT_ON_HOLD, refused.code)
        // The case's own note and alert tell people: no version 1 hold message, no second alert.
        assertEquals(0, messages(b))
        assertEquals(listOf("copied:$c"), jdbc.queryForList("SELECT dedupe_key FROM relevance_alerts WHERE batch_id = ?", String::class.java, b.id))
        // Only the COPIED alert's admin ping: no version 1 hold event (its e-mail).
        assertEquals(emptyList(), Mockito.mockingDetails(publisher).invocations.flatMap { it.arguments.toList() }
            .filterIsInstance<com.quickpitik.service.admin.PayoutHoldPlacedEvent>())
    }

    @Test
    fun `the hold is logged as automatic`() {
        val b = batch()
        val c = applied(b)

        hold().place(c)

        val row = jdbc.queryForMap("SELECT admin_id, automatic, decision, meta::text AS meta FROM admin_decision_log WHERE target_user_id = ?", b.photographer)
        assertEquals(listOf<Any?>(null, true, "payout_hold_auto"), listOf(row["admin_id"], row["automatic"], row["decision"]))
        assertEquals(true, (row["meta"] as String).contains(c.toString()))
    }

    @Test
    fun `a copied case in record-only places no hold`() {
        val b = batch()
        val c = decided(b)

        checks(RelevanceMode.RECORD_ONLY).applyCase(c, OffsetDateTime.now())

        assertEquals(listOf("APPLIED", "RECORD_ONLY"), row(c, "stage", "applied_mode"))
        assertEquals(emptyList(), flagsOf(b))
        assertEquals(false, payouts.relevanceHoldOpen(b.photographer))
    }

    @Test
    fun `with the flags switch off no hold is placed`() {
        val b = batch()
        val c = decided(b)

        // RelevanceSwitch never yields auto-hide with the flags switch off, so the case is applied as record-only.
        checks(flagsEnabled = false).applyCase(c, OffsetDateTime.now())
        hold(flagsEnabled = false).place(c)

        assertEquals("RECORD_ONLY", row(c, "applied_mode").single())
        assertEquals(emptyList(), flagsOf(b))
    }

    @Test
    fun `placing twice makes one flag and one hold`() {
        val b = batch()
        val c = applied(b)

        hold().place(c)
        hold().place(c)

        assertEquals(1, flagsOf(b).size)
        assertEquals(1, decisions(b))
    }

    @Test
    fun `an admin ending the hold lifts it and sends the lifted note`() {
        val b = batch()
        val c = applied(b)
        hold().place(c)
        val admin = admin()
        assertNotNull(tx.execute { admin(RelevanceMode.AUTO_HIDE).case(c) }!!.payoutHoldAt)

        tx.execute { admin(RelevanceMode.AUTO_HIDE).endHold(admin, c) }

        assertEquals(listOf<Any?>("resolved", admin), flagsOf(b).single().let { listOf(it["status"], it["resolved_by"]) })
        assertEquals(false, payouts.relevanceHoldOpen(b.photographer))
        assertEquals(listOf("Payout hold lifted"), titles(b))
        assertEquals(listOf("APPLIED", "HIDE_ALL"), row(c, "stage", "action"), "the case stays")
        assertNull(tx.execute { admin(RelevanceMode.AUTO_HIDE).case(c) }!!.payoutHoldAt)
        val again = assertFailsWith<ApiException> { tx.execute { admin(RelevanceMode.AUTO_HIDE).endHold(admin, c) } }
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, again.code)
    }

    @Test
    fun `an admin restore ends the hold`() {
        val b = batch()
        val c = applied(b)
        hold().place(c)

        admin(RelevanceMode.AUTO_HIDE).restore(admin(), c, null)

        assertEquals("dismissed", flagsOf(b).single()["status"])
        assertEquals(false, payouts.relevanceHoldOpen(b.photographer))
        assertEquals(listOf("Payout hold lifted"), titles(b))
        assertEquals("RESTORED_BY_ADMIN", tx.execute { admin(RelevanceMode.AUTO_HIDE).case(c) }!!.closeReason)
    }

    @Test
    fun `a won appeal ends the hold`() {
        val b = batch()
        val c = applied(b)
        hold().place(c)
        val admin = admin()
        val appeal = id("INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason, evidence_opened_at, evidence_opened_by) " +
            "VALUES (?, ?, ?, 'These are mine', now(), ?)", c, b.id, b.photographer, admin)

        tx.execute { appealService().decide(admin, appeal, "WON", null) }

        assertEquals("dismissed", flagsOf(b).single()["status"])
        assertEquals(false, payouts.relevanceHoldOpen(b.photographer))
        assertEquals(listOf("Payout hold lifted"), titles(b))
    }

    @Test
    fun `a photographer with an open version 1 flag gets the hold on that flag`() {
        val b = batch()
        val v1 = id("INSERT INTO flags (target_kind, target_id, reason, note) VALUES ('user', ?, 'batch_relevance_low', 'version 1')", b.photographer)
        val c = applied(b)

        hold().place(c)

        assertEquals(listOf(v1), flagsOf(b).map { it["id"] })
        assertNotNull(flagsOf(b).single()["payout_hold_at"])
        assertEquals(v1, holdFlagOf(c))
        assertEquals(true, payouts.relevanceHoldOpen(b.photographer))
    }

    @Test
    fun `the case page shows a current case's close reason as null`() {
        val c = applied(batch())

        val detail = tx.execute { admin(RelevanceMode.AUTO_HIDE).case(c) }!!

        assertNull(detail.closeReason)
        assertNull(detail.payoutHoldAt)
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun props(mode: RelevanceMode) = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))

    private fun switch(mode: RelevanceMode, flagsEnabled: Boolean) =
        RelevanceSwitch(props(mode), AdminProperties(flagsEnabled = flagsEnabled), AiApiProperties(enabled = false))

    private fun hold(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, flagsEnabled: Boolean = true) =
        RelevanceHoldService(cases, strikes, systemFlags, adminFlags, decisionLog, switch(mode, flagsEnabled), tx)

    private fun checks(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, flagsEnabled: Boolean = true): RelevanceCheckService {
        val props = props(mode)
        return RelevanceCheckService(batchRepo, cases, props, switch(mode, flagsEnabled), Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts, publisher,
            LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"), holds, photoRepo, decisionLog,
            SyncTaskExecutor(), Mockito.mock(RelevanceNotifier::class.java), Mockito.mock(RelevanceConfirmer::class.java),
            hold(mode, flagsEnabled))
    }

    private fun admin(mode: RelevanceMode): AdminRelevanceService {
        val props = props(mode)
        return AdminRelevanceService(cases, alerts, batchRepo, props, Mockito.mock(StorageService::class.java), StorageProperties(),
            Clock.systemUTC(), holds, photoRepo, decisionLog, tx, switch(mode, true), strikes, hold(mode))
    }

    private fun appealService(): RelevanceAppealService {
        val props = props(RelevanceMode.AUTO_HIDE)
        return RelevanceAppealService(cases, appeals, alerts, Mockito.mock(RelevanceNotifier::class.java), switch(RelevanceMode.AUTO_HIDE, true),
            props, Clock.systemUTC(), publisher, Mockito.mock(RelevanceConfirmer::class.java), decisionLog, hold())
    }

    private fun admin(): UUID = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Adm', 'ADMIN')",
        "relhold-admin-${UUID.randomUUID()}@test.local")

    private fun batch(): Batch {
        val photographer = id("INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "relhold-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Hold Run', DATE '2007-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relhold-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 40, ?, ?)",
            event, photographer, UPLOAD, UPLOAD)
        val batch = id("INSERT INTO batch_relevance (event_id, photographer_id, scored_at, scored_upload_at, copy_low, checked_upload_at, checked_photos) " +
            "VALUES (?, ?, ?, ?, true, ?, 40)", event, photographer, UPLOAD, UPLOAD, UPLOAD)
        return Batch(batch, event, photographer)
    }

    // A Copied case waiting for the apply step.
    private fun decided(b: Batch): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, copy_low, decided_at) " +
            "VALUES (?, 'QUIET', ?, 40, 'DECIDED', 40, 0, 'COPIED', true, ?)", b.id, UPLOAD, UPLOAD)

    // A current Copied case, applied in `mode`.
    private fun applied(b: Batch, mode: String = "AUTO_HIDE"): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, copy_low, decided_at, " +
            "applied_mode, action, applied_at, notice_at) VALUES (?, 'QUIET', ?, 40, 'APPLIED', 40, 0, 'COPIED', true, ?, ?, 'HIDE_ALL', ?, ?)",
        b.id, UPLOAD, UPLOAD, mode, UPLOAD, UPLOAD)

    private fun flagsOf(b: Batch): List<Map<String, Any?>> = jdbc.queryForList(
        "SELECT id, status, target_kind, reporter_id, payout_hold_at, payout_hold_by, resolved_by FROM flags " +
            "WHERE target_id = ? AND reason = 'batch_relevance_low'", b.photographer)

    private fun holdFlagOf(caseId: UUID): UUID? =
        jdbc.queryForObject("SELECT hold_flag_id FROM relevance_cases WHERE id = ?", UUID::class.java, caseId)

    private fun messages(b: Batch): Int =
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages WHERE photographer_id = ?", Int::class.java, b.photographer)!!

    private fun titles(b: Batch): List<String> =
        jdbc.queryForList("SELECT title FROM photographer_messages WHERE photographer_id = ? ORDER BY created_at", String::class.java, b.photographer)

    private fun decisions(b: Batch): Int = jdbc.queryForObject(
        "SELECT count(*) FROM admin_decision_log WHERE target_user_id = ? AND decision = 'payout_hold_auto'", Int::class.java, b.photographer)!!

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val UPLOAD: OffsetDateTime = OffsetDateTime.parse("2007-03-04T05:06:07Z")
    }
}
