package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.exception.NotFoundException
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
import com.quickpitik.service.admin.AdminUserService
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceConfirmer
import com.quickpitik.service.relevance.RelevanceHoldService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceNotifier
import com.quickpitik.service.relevance.RelevanceSwitch
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
import kotlin.test.assertTrue

// RV-L7 part a (legal 1.7): an admin records that a photographer withdrew the consent to automatic actions. From then on
// that photographer's cases apply as record-only with NO_CONSENT, and no automatic step hides one more photo or places a
// payout hold, also on a case applied in auto-hide before; what is already hidden stays. Fixtures sit in 2008 and
// assertions look only at this class's own rows (the container is shared).
class RelevanceConsentWithdrawalIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batchRepo: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photoRepo: PhotoRepository
    @Autowired private lateinit var strikes: RelevanceStrikeRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var systemFlags: SystemFlagService
    @Autowired private lateinit var adminFlags: AdminFlagService
    @Autowired private lateinit var decisionLog: AdminDecisionLogService
    @Autowired private lateinit var adminRelevance: AdminRelevanceService
    @Autowired private lateinit var adminUsers: AdminUserService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `a photographer who withdrew is applied as record-only with the no-consent mark`() {
        val b = batch(photos = 600)
        val c = decided(b, "JUNK_DUMP")
        adminRelevance.withdrawConsent(admin(), b.photographer)

        checks().applyCase(c, NOW)

        assertEquals(listOf("APPLIED", "RECORD_ONLY", "HIDE_ALL", null, true, "NONE"),
            row(c, "stage", "applied_mode", "action", "verdict_note", "no_consent", "scan_state"))
        // The same "would have" alert as today's no-consent path.
        assertEquals(listOf(listOf("BIG_HIDE", true, 600)), alertsOf(b))
        assertEquals(0, messages(b))
    }

    @Test
    fun `clearing the withdrawal lets the next case apply in auto-hide`() {
        val b = batch(photos = 40)
        val admin = admin()
        adminRelevance.withdrawConsent(admin, b.photographer)
        val cleared = adminRelevance.clearConsentWithdrawal(admin, b.photographer)
        val c = decided(b, "JUNK_DUMP")

        checks().applyCase(c, NOW)

        assertNull(cleared.consentWithdrawnAt)
        assertEquals(listOf("APPLIED", "AUTO_HIDE", "HIDE_ALL", null), row(c, "stage", "applied_mode", "action", "verdict_note"))
    }

    @Test
    fun `a withdrawal stops further automatic hides on a case applied earlier and leaves what is hidden`() {
        val b = batch(photos = 7)
        val c = applied(b, "JUNK_DUMP", "HIDE_ALL")
        val service = checks(chunkSize = 3, chunksPerTick = 1)
        service.reconcileHolds()
        assertEquals(3, hidden(b), "one chunk before the withdrawal")

        adminRelevance.withdrawConsent(admin(), b.photographer)
        repeat(3) { service.reconcileHolds() }

        assertEquals(3, hidden(b))
        // The case stays a target (so restores still run), marked withdrawn; the guard read before each hiding chunk says no.
        assertEquals(true, cases.findHoldTargets().single { it.caseId == c }.withdrawn)
        assertEquals(false, cases.isHideTarget(c, "HIDE_ALL"))
        assertEquals(3L, holds.countLiveHolds(b.id, "AUTO"))
        assertEquals(listOf(null), row(c, "hide_finished_at"))
        // The scan queue skips the photographer too: a queued scan would only raise a target nothing acts on.
        val scanned = batch(photos = 3, photographer = b.photographer)
        val queued = applied(scanned, "FITS", "NONE", scan = "QUEUED")
        assertTrue(cases.findScans(10_000).none { it.caseId == queued })
    }

    @Test
    fun `a withdrawal stops a new copied payout hold on a case applied earlier`() {
        val b = batch(photos = 40)
        val c = applied(b, "COPIED", "HIDE_ALL")
        adminRelevance.withdrawConsent(admin(), b.photographer)

        // The query that picks the work leaves it out, and place (also called right after apply) checks under the lock.
        assertTrue(c !in cases.findHoldDue(10_000))
        checks().placeOwedHolds()
        hold().place(c)

        assertNull(jdbc.queryForObject("SELECT hold_flag_id FROM relevance_cases WHERE id = ?", UUID::class.java, c))
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM flags WHERE target_id = ?", Int::class.java, b.photographer))
    }

    // Review of PR 171: the withdrawal stops only steps that take one more photo off sale; restores still run.
    @Test
    fun `a won appeal on a whole-batch case still puts the good photos back for a photographer who withdrew`() {
        val b = batch(photos = 0)
        val c = applied(b, "JUNK_DUMP", "HIDE_ALL")
        val clean = List(2) { held(b, c, sharpness = 50.0) }
        val junk = held(b, c, sharpness = 10.0)
        adminRelevance.withdrawConsent(admin(), b.photographer)
        // Uploaded and measured as junk after the withdrawal: the post-appeal scan must not hide it.
        val later = measured(b, sharpness = 10.0)
        win(b, c)
        assertEquals(listOf("HIDE_JUNK", "QUEUED"), row(c, "action", "scan_state"))

        val check = checks(scanning = true)
        repeat(10) {
            check.reconcileHolds()
            if (row(c, "stage").single() != "CLOSED") check.scanTurns()
        }

        assertEquals(listOf("LIVE", "LIVE"), clean.map(::status))
        assertEquals("HIDDEN", status(junk), "what was hidden stays hidden unless the scan clears it")
        assertEquals("LIVE", status(later))
        assertEquals(listOf("CLOSED", "APPEAL_WON"), row(c, "stage", "close_reason"))
    }

    @Test
    fun `a won appeal on a mixed case still puts the photos back for a photographer who withdrew`() {
        val b = batch(photos = 0)
        val c = applied(b, "MIXED", "HIDE_JUNK", scan = "FINISHED")
        val junk = held(b, c, sharpness = 10.0)
        adminRelevance.withdrawConsent(admin(), b.photographer)
        win(b, c)
        assertEquals(listOf("NONE"), row(c, "action"))

        checks().reconcileHolds()

        assertEquals("LIVE", status(junk))
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `an admin's hide still takes the photos off sale for a photographer who withdrew`() {
        val b = batch(photos = 3)
        val c = applied(b, "CANT_TELL", "NONE")
        val admin = admin()
        adminRelevance.withdrawConsent(admin, b.photographer)

        tx.execute { adminService().hideBatch(admin, c, null) }
        checks().reconcileHolds()

        assertEquals(3, hidden(b))
        assertEquals(3L, holds.countLiveHolds(b.id, "ADMIN"))
    }

    @Test
    fun `setting the withdrawal twice keeps the first date`() {
        val b = batch(photos = 1)
        val admin = admin()

        val first = adminRelevance.withdrawConsent(admin, b.photographer)
        val second = adminRelevance.withdrawConsent(admin, b.photographer)

        assertEquals(b.photographer, first.photographerId)
        assertNotNull(first.consentWithdrawnAt)
        assertEquals(first.consentWithdrawnAt, second.consentWithdrawnAt)
        assertEquals(first.consentWithdrawnAt, users.findById(b.photographer).get().relevanceConsentWithdrawnAt?.toString())
    }

    @Test
    fun `each route writes one admin decision row`() {
        val b = batch(photos = 1)
        val admin = admin()

        adminRelevance.withdrawConsent(admin, b.photographer)
        adminRelevance.withdrawConsent(admin, b.photographer)
        adminRelevance.clearConsentWithdrawal(admin, b.photographer)
        adminRelevance.clearConsentWithdrawal(admin, b.photographer)

        assertEquals(listOf(listOf<Any?>(admin, "relevance_consent_withdrawn"), listOf<Any?>(admin, "relevance_consent_withdrawal_cleared")),
            jdbc.queryForList("SELECT admin_id, decision FROM admin_decision_log WHERE target_user_id = ? ORDER BY decided_at, decision DESC", b.photographer)
                .map { listOf(it["admin_id"], it["decision"]) })
        // 404 for an id that is not a photographer.
        val runner = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Run', 'RUNNER')", "relwd-${UUID.randomUUID()}@test.local")
        for (other in listOf(runner, UUID.randomUUID())) {
            assertFailsWith<NotFoundException> { adminRelevance.withdrawConsent(admin, other) }
            assertFailsWith<NotFoundException> { adminRelevance.clearConsentWithdrawal(admin, other) }
        }
    }

    @Test
    fun `the admin photographer detail carries the date`() {
        val b = batch(photos = 1)
        assertNull(adminUsers.detail(b.photographer).relevanceConsentWithdrawnAt)

        val set = adminRelevance.withdrawConsent(admin(), b.photographer)

        assertEquals(set.consentWithdrawnAt, adminUsers.detail(b.photographer).relevanceConsentWithdrawnAt)
    }

    @Test
    fun `with the mode unset the withdrawal changes nothing`() {
        val withdrawn = batch(photos = 600)
        val w = decided(withdrawn, "JUNK_DUMP")
        val twin = batch(photos = 600)
        val t = decided(twin, "JUNK_DUMP")
        adminRelevance.withdrawConsent(admin(), withdrawn.photographer)

        checks(mode = RelevanceMode.OFF).tick()
        assertEquals(listOf("DECIDED", null), row(w, "stage", "verdict_note"))

        // Record-only applies a withdrawn photographer's case exactly as anyone else's.
        listOf(w, t).forEach { checks(mode = RelevanceMode.RECORD_ONLY).applyCase(it, NOW) }
        assertEquals(row(t, "stage", "applied_mode", "action", "verdict_note"), row(w, "stage", "applied_mode", "action", "verdict_note"))
        assertEquals(listOf("RECORD_ONLY", null), row(w, "applied_mode", "verdict_note"))
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun props(mode: RelevanceMode, chunkSize: Int, chunksPerTick: Int) = RelevanceProperties(mode = mode,
        junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0), check = RelevanceProperties.Check(maxBatchesPerTick = 10_000),
        hide = RelevanceProperties.Hide(chunkSize = chunkSize, chunksPerTick = chunksPerTick))

    private fun switch(mode: RelevanceMode, chunkSize: Int = 1000, chunksPerTick: Int = 10, scanning: Boolean = false): RelevanceSwitch {
        @Suppress("UNCHECKED_CAST")
        val provider = (Mockito.mock(org.springframework.beans.factory.ObjectProvider::class.java)
            as org.springframework.beans.factory.ObjectProvider<com.quickpitik.service.ai.RekognitionAiClient>)
            .also { Mockito.`when`(it.getIfAvailable()).thenReturn(Mockito.mock(com.quickpitik.service.ai.RekognitionAiClient::class.java)) }
        return RelevanceSwitch(props(mode, chunkSize, chunksPerTick), AdminProperties(flagsEnabled = true), AiApiProperties(enabled = scanning), provider)
    }

    private fun hold(mode: RelevanceMode = RelevanceMode.AUTO_HIDE) =
        RelevanceHoldService(cases, strikes, systemFlags, adminFlags, decisionLog, switch(mode), tx, AdminProperties(flagsEnabled = true), users)

    private fun checks(mode: RelevanceMode = RelevanceMode.AUTO_HIDE, chunkSize: Int = 1000, chunksPerTick: Int = 10, scanning: Boolean = false) =
        RelevanceCheckService(batchRepo, cases, props(mode, chunkSize, chunksPerTick), switch(mode, chunkSize, chunksPerTick, scanning), Clock.systemUTC(),
            tx, SimpleMeterRegistry(), Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"),
            holds, photoRepo, decisionLog, if (scanning) org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor().apply { initialize() } else SyncTaskExecutor(),
            Mockito.mock(RelevanceNotifier::class.java),
            Mockito.mock(RelevanceConfirmer::class.java), hold(mode))

    private fun adminService() = AdminRelevanceService(cases, alerts, batchRepo, props(RelevanceMode.AUTO_HIDE, 1000, 10),
        Mockito.mock(com.quickpitik.service.storage.StorageService::class.java), com.quickpitik.config.StorageProperties(), Clock.systemUTC(),
        holds, photoRepo, decisionLog, tx, switch(RelevanceMode.AUTO_HIDE), strikes, hold(), users)

    // A measured photo of the batch, hidden and held AUTO under the case.
    private fun held(b: Batch, caseId: UUID, sharpness: Double): UUID {
        val photo = measured(b, sharpness, status = "HIDDEN")
        jdbc.update("INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at) VALUES (?, ?, 'AUTO', ?, ?)", b.id, photo, caseId, NOW)
        return photo
    }

    private fun measured(b: Batch, sharpness: Double, status: String = "LIVE"): UUID {
        val photo = id("INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'relwd/thumb', 'relwd/wm', 125, ?)", b.event, b.photographer, "relwd/${UUID.randomUUID()}", status)
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, sharpness, brightness, person_conf, race_conf) " +
            "VALUES (?, ?, ?, ?, ?, 50, 0.9, 0.9)", photo, b.id, NOW, sharpness, sharpness)
        return photo
    }

    // The rows an admin's WON decision leaves (RelevanceAppealService.decide): the appeal won, the target lowered.
    private fun win(b: Batch, caseId: UUID) {
        jdbc.update("INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason, status, decided_at) VALUES (?, ?, ?, 'mine', 'WON', ?)",
            caseId, b.id, b.photographer, NOW)
        tx.execute { cases.winAppeal(caseId) }
    }

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun admin(): UUID = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Adm', 'ADMIN')",
        "relwd-admin-${UUID.randomUUID()}@test.local")

    // A public shared race's batch with `photos` live photos; the photographer accepted Terms 1.7.
    private fun batch(photos: Int, photographer: UUID? = null): Batch {
        val p = photographer ?: id("INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "relwd-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Withdraw Run', DATE '2008-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relwd-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?, ?)",
            event, p, photos, UPLOAD, UPLOAD)
        val batchId = id("INSERT INTO batch_relevance (event_id, photographer_id, scored_at, scored_upload_at, copy_low, checked_upload_at, checked_photos) " +
            "VALUES (?, ?, ?, ?, false, ?, ?)", event, p, NOW, UPLOAD, UPLOAD, photos)
        jdbc.update("INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) " +
            "SELECT ?, ?, 'relwd/' || gen_random_uuid(), 125, 'LIVE' FROM generate_series(1, ?)", event, p, photos)
        return Batch(batchId, event, p)
    }

    // A case waiting for the apply step.
    private fun decided(b: Batch, verdict: String): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, copy_low, decided_at, started_mode) " +
            "VALUES (?, 'QUIET', ?, (SELECT photo_count FROM event_photographer WHERE event_id = ?), 'DECIDED', 40, 0, ?, false, ?, 'AUTO_HIDE')",
        b.id, UPLOAD, b.event, verdict, NOW)

    // A current case applied in auto-hide before the withdrawal.
    private fun applied(b: Batch, verdict: String, action: String, scan: String = "NONE"): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, copy_low, decided_at, " +
            "applied_mode, action, applied_at, scan_state) VALUES (?, 'QUIET', ?, 40, 'APPLIED', 40, 0, ?, false, ?, 'AUTO_HIDE', ?, ?, ?)",
        b.id, UPLOAD, verdict, NOW, action, NOW, scan)

    private fun hidden(b: Batch): Int =
        jdbc.queryForObject("SELECT count(*) FROM photos WHERE event_id = ? AND status = 'HIDDEN'", Int::class.java, b.event)!!

    private fun messages(b: Batch): Int =
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages WHERE photographer_id = ?", Int::class.java, b.photographer)!!

    private fun alertsOf(b: Batch): List<List<Any?>> =
        jdbc.queryForList("SELECT kind, would_have, photos FROM relevance_alerts WHERE batch_id = ? ORDER BY kind", b.id)
            .map { listOf(it["kind"], it["would_have"], it["photos"]) }

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2008-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
    }
}
