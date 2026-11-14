package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

// RV-2.5a (G10): an admin restores a batch or one photo, keeps a batch hidden, and reads its holds. The tick finishes a
// restore that a hide chunk outran. The container is shared, so fixtures sit in 2010 and assertions look only at this
// class's own rows.
class AdminRelevanceActionsIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var admin: AdminRelevanceService
    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var decisions: AdminDecisionLogService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    @MockitoBean lateinit var storage: StorageService
    private lateinit var adminId: UUID

    @BeforeEach fun setUp() {
        Mockito.`when`(storage.presignedGetUrl(anyArg(), anyArg())).thenAnswer { "https://synthetic.test/${it.arguments[0]}" }
        adminId = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Admin', 'ADMIN')",
            "reladmin-${UUID.randomUUID()}@test.local")
    }
    private fun <T> anyArg(): T = Mockito.any()

    @Test
    fun `restoring a batch brings back every automatic and admin hold and closes the case`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP")
        val auto = List(3) { hidden(b, "AUTO", c) }
        val byAdmin = hidden(b, "ADMIN", c)

        val result = admin.restore(adminId, c, "  Honest batch  ")

        assertEquals(4, result.restoredPhotos)
        assertEquals(listOf("LIVE"), (auto + byAdmin).map(::status).distinct())
        assertEquals(listOf("ADMIN"), jdbc.queryForList("SELECT DISTINCT restored_by FROM flag_hidden_photos WHERE batch_id = ?", String::class.java, b.id))
        assertEquals(listOf("CLOSED", "RESTORED_BY_ADMIN", adminId), row(c, "stage", "close_reason", "closed_by"))
        assertEquals(listOf(listOf(adminId, "relevance_restored", "Honest batch", c.toString())), decisionsOf(b))
    }

    @Test
    fun `restoring a batch leaves a photographer's own hides in place`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP")
        val own = hidden(b, "PHOTOGRAPHER", null)
        val flagged = hidden(b, "AUTO", c).also {
            jdbc.update("INSERT INTO flags (target_kind, target_id, reason, status) VALUES ('photo', ?, 'other', 'hidden')", it)
        }

        admin.restore(adminId, c, null)

        assertEquals(listOf("HIDDEN", "HIDDEN"), listOf(own, flagged).map(::status))
        assertEquals(1L, holds.countLiveHolds(b.id, "PHOTOGRAPHER"))
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `a batch larger than one chunk is restored in one request`() {
        val b = batch()
        val c = case(b, "WRONG_EVENT")
        val held = List(5) { hidden(b, "AUTO", c) }

        assertEquals(5, service(chunkSize = 2).restore(adminId, c, null).restoredPhotos)

        assertEquals(listOf("LIVE"), held.map(::status).distinct())
    }

    @Test
    fun `restoring a restored case again finishes what was left and logs no second decision`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP")
        hidden(b, "AUTO", c)
        admin.restore(adminId, c, null)
        val late = hidden(b, "AUTO", c)

        assertEquals(1, admin.restore(adminId, c, null).restoredPhotos)
        assertEquals(0, admin.restore(adminId, c, null).restoredPhotos)

        assertEquals("LIVE", status(late))
        assertEquals(1, decisionsOf(b).size)
    }

    @Test
    fun `restore is refused for an unknown case or a case in the wrong state`() {
        assertEquals(ErrorCodes.RELEVANCE_CASE_NOT_FOUND, assertThrows<NotFoundException> { admin.restore(adminId, UUID.randomUUID(), null) }.code)
        val b = batch()
        val decided = case(b, "JUNK_DUMP", stage = "DECIDED")
        val superseded = case(b, "JUNK_DUMP", stage = "CLOSED", closeReason = "SUPERSEDED")
        // A restored case whose batch has a newer current case: restoring again would undo the newer case's hide.
        val other = batch()
        val restored = case(other, "JUNK_DUMP", stage = "CLOSED", closeReason = "RESTORED_BY_ADMIN")
        case(other, "COPIED")

        for (id in listOf(decided, superseded, restored)) {
            val ex = assertThrows<ApiException> { admin.restore(adminId, id, null) }
            assertEquals(HttpStatus.CONFLICT to ErrorCodes.INVALID_STATE_TRANSITION, ex.status to ex.code)
        }
        assertEquals(emptyList(), decisionsOf(b) + decisionsOf(other))
    }

    @Test
    fun `after an admin restore the same verdict does not hide the batch again`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP")
        val held = List(2) { hidden(b, "AUTO", c) }
        admin.restore(adminId, c, null)
        val again = case(b, "JUNK_DUMP", stage = "DECIDED")
        val check = checkService()

        check.applyCase(again, NOW)
        check.reconcileHolds()

        assertEquals(listOf("CLOSED", "SAME_VERDICT"), row(again, "stage", "close_reason"))
        assertNull(cases.currentFor(b.id))
        assertEquals(listOf("LIVE"), held.map(::status).distinct())
    }

    @Test
    fun `keep hidden does not confirm the case`() {
        val b = batch()
        val c = case(b, "COPIED")
        val held = hidden(b, "AUTO", c)
        val alert = id("INSERT INTO relevance_alerts (batch_id, case_id, kind, dedupe_key, would_have, photos) VALUES (?, ?, 'COPIED', ?, false, 1)",
            b.id, c, "copied:$c")

        val dto = admin.keepHidden(adminId, c, "Looked at the samples")

        assertEquals(c, dto.caseId)
        assertEquals(listOf("APPLIED", null, "HIDE_ALL"), row(c, "stage", "confirmed_at", "action"))
        assertEquals("HIDDEN", status(held))
        assertEquals(adminId, jdbc.queryForObject("SELECT seen_by FROM relevance_alerts WHERE id = ?", UUID::class.java, alert))
        assertEquals(listOf(listOf(adminId, "relevance_kept_hidden", "Looked at the samples", c.toString())), decisionsOf(b))
        assertEquals(0L, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE photographer_id = ?", Long::class.java, b.photographer))
    }

    @Test
    fun `keep hidden is refused for a case that hides nothing or is not current`() {
        val b = batch()
        val fits = case(b, "FITS")
        val other = batch()
        val closed = case(other, "JUNK_DUMP", stage = "CLOSED", closeReason = "RESTORED_BY_ADMIN")

        for (id in listOf(fits, closed)) {
            val ex = assertThrows<ApiException> { admin.keepHidden(adminId, id, null) }
            assertEquals(HttpStatus.CONFLICT to ErrorCodes.INVALID_STATE_TRANSITION, ex.status to ex.code)
        }
        assertEquals(emptyList(), decisionsOf(b) + decisionsOf(other))
    }

    @Test
    fun `restoring one photo brings it back and its case does not hide it again`() {
        val b = batch()
        val older = case(b, "JUNK_DUMP", stage = "CLOSED", closeReason = "SUPERSEDED")
        val c = case(b, "COPIED")
        val one = hidden(b, "AUTO", older)
        val rest = hidden(b, "AUTO", c)

        val dto = admin.restorePhoto(adminId, c, one)
        checkService().reconcileHolds()

        assertEquals(one to "LIVE", dto.photoId to dto.status)
        assertEquals(listOf("LIVE", "HIDDEN"), listOf(one, rest).map(::status))
        assertEquals(listOf("ADMIN", c), jdbc.queryForList("SELECT restored_by, case_id FROM flag_hidden_photos WHERE photo_id = ?", one)
            .single().let { listOf(it["restored_by"], it["case_id"]) })
        // The tick also logged its own automatic hide of the batch; the admin's row is the one this action wrote.
        assertEquals(listOf(listOf(adminId, "relevance_photo_restored", null, c.toString())), decisionsOf(b).filter { it[0] != null })
    }

    @Test
    fun `a photo another hold still keeps stays hidden after a single restore`() {
        val b = batch()
        val c = case(b, "COPIED")
        val both = hidden(b, "AUTO", c).also { hold(b, it, "PHOTOGRAPHER", null) }

        assertEquals("HIDDEN", admin.restorePhoto(adminId, c, both).status)
        assertEquals(1L, holds.countLiveHolds(b.id, "PHOTOGRAPHER"))
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `a single restore is refused for a photo of another batch or a case that is not current`() {
        val b = batch()
        val c = case(b, "COPIED")
        val stranger = hidden(batch(), "AUTO", null)
        val closed = case(batch(), "JUNK_DUMP", stage = "CLOSED", closeReason = "RESTORED_BY_ADMIN")

        assertEquals(ErrorCodes.PHOTO_NOT_FOUND, assertThrows<NotFoundException> { admin.restorePhoto(adminId, c, stranger) }.code)
        val ex = assertThrows<ApiException> { admin.restorePhoto(adminId, closed, stranger) }
        assertEquals(HttpStatus.CONFLICT to ErrorCodes.INVALID_STATE_TRANSITION, ex.status to ex.code)
        assertEquals("HIDDEN", status(stranger))
    }

    @Test
    fun `the hidden-photos read pages the batch's holds, restored ones included, with preview thumbnails`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP")
        val held = List(3) { hidden(b, "AUTO", c) }
        jdbc.update("UPDATE flag_hidden_photos SET restored_at = ?, restored_by = 'ADMIN' WHERE photo_id = ?", NOW, held[0])

        val page = admin.hiddenPhotos(c, PaginationParams.of(0, 2))

        assertEquals(3L, page.total)
        assertEquals(2, page.items.size)
        val all = page.items + admin.hiddenPhotos(c, PaginationParams.of(2, 2)).items
        assertEquals(held.toSet(), all.map { it.photoId }.toSet())
        assertEquals(1, all.count { it.restoredBy == "ADMIN" })
        assertEquals(setOf("https://synthetic.test/reladmin/wm"), all.map { it.thumbnailUrl }.toSet())
        assertEquals(ErrorCodes.RELEVANCE_CASE_NOT_FOUND,
            assertThrows<NotFoundException> { admin.hiddenPhotos(UUID.randomUUID(), PaginationParams.of(0, 2)) }.code)
    }

    @Test
    fun `the tick finishes a restore that a hide chunk outran, and leaves a newer case's hide`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP")
        admin.restore(adminId, c, null)
        val late = hidden(b, "AUTO", c)
        val newer = batch()
        case(newer, "JUNK_DUMP", stage = "CLOSED", closeReason = "RESTORED_BY_ADMIN")
        val now = case(newer, "COPIED")
        val kept = hidden(newer, "AUTO", now)

        checkService().reconcileHolds()

        assertEquals("LIVE", status(late))
        assertEquals("ADMIN", jdbc.queryForObject("SELECT restored_by FROM flag_hidden_photos WHERE photo_id = ?", String::class.java, late))
        assertEquals("HIDDEN", status(kept))
    }

    @Test
    fun `with the mode off nothing runs`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP")
        admin.restore(adminId, c, null)
        val late = hidden(b, "AUTO", c)

        checkService(mode = RelevanceMode.OFF).tick()

        assertEquals("HIDDEN", status(late))
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP")
        admin.restore(adminId, c, null)
        hidden(b, "AUTO", c)
        val before = untouched()

        checkService(mode = RelevanceMode.RECORD_ONLY).tick()

        assertEquals(before, untouched())
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch()
        val c = case(b, "COPIED", mode = "RECORD_ONLY")
        val held = hidden(b, "AUTO", null)

        checkService().reconcileHolds()

        assertEquals("HIDDEN", status(held))
        assertEquals(listOf("APPLIED"), row(c, "stage"))
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun service(chunkSize: Int): AdminRelevanceService {
        val props = RelevanceProperties(hide = RelevanceProperties.Hide(chunkSize = chunkSize))
        return AdminRelevanceService(cases, alerts, batches, props, storage, StorageProperties(), Clock.systemUTC(),
            holds, photos, decisions, tx)
    }

    private fun checkService(mode: RelevanceMode = RelevanceMode.AUTO_HIDE): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false))
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"),
            holds, photos, decisions, org.springframework.core.task.SyncTaskExecutor())
    }

    private fun batch(): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "reladmin-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Admin Run', DATE '2010-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "reladmin-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, 600, ?, ?)",
            event, photographer, UPLOAD, UPLOAD,
        )
        val batchId = id(
            "INSERT INTO batch_relevance (event_id, photographer_id, scored_at, scored_upload_at, copy_low, checked_upload_at, checked_photos) " +
                "VALUES (?, ?, ?, ?, false, ?, 600)",
            event, photographer, NOW, UPLOAD, UPLOAD,
        )
        return Batch(batchId, event, photographer)
    }

    // A case of the batch, current (APPLIED in auto-hide) unless `stage` says otherwise.
    private fun case(b: Batch, verdict: String, stage: String = "APPLIED", mode: String = "AUTO_HIDE", closeReason: String? = null): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, copy_low, decided_at, applied_mode, action, " +
            "applied_at, closed_at, close_reason) VALUES (?, 'QUIET', ?, 600, ?, ?, false, ?, ?, ?, ?, ?, ?)",
        b.id, UPLOAD, stage, verdict, NOW, mode.takeIf { stage != "DECIDED" }, action(verdict).takeIf { stage != "DECIDED" },
        NOW.takeIf { stage != "DECIDED" }, closeReason?.let { NOW }, closeReason,
    )

    private fun action(verdict: String) = when (verdict) {
        "MIXED" -> "HIDE_JUNK"
        "JUNK_DUMP", "WRONG_EVENT", "COPIED" -> "HIDE_ALL"
        else -> "NONE"
    }

    // A HIDDEN photo of the batch with one live hold of kind `heldBy`.
    private fun hidden(b: Batch, heldBy: String, caseId: UUID?): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'reladmin/thumb', 'reladmin/wm', 125, 'HIDDEN')",
        b.event, b.photographer, "reladmin/${UUID.randomUUID()}",
    ).also { hold(b, it, heldBy, caseId) }

    private fun hold(b: Batch, photo: UUID, heldBy: String, caseId: UUID?) {
        jdbc.update("INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at) VALUES (?, ?, ?, ?, ?)",
            b.id, photo, heldBy, caseId, NOW)
    }

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun decisionsOf(b: Batch): List<List<Any?>> =
        jdbc.queryForList("SELECT admin_id, decision, reason, meta->>'caseId' AS case_id FROM admin_decision_log WHERE target_user_id = ? ORDER BY decided_at",
            b.photographer).map { listOf(it["admin_id"], it["decision"], it["reason"], it["case_id"]) }

    private fun untouched(): List<Any?> = listOf(
        jdbc.queryForList("SELECT status, count(*) AS n FROM photos GROUP BY status ORDER BY status"),
        jdbc.queryForList("SELECT held_by, restored_by, count(*) AS n FROM flag_hidden_photos GROUP BY held_by, restored_by ORDER BY 1, 2"),
        jdbc.queryForList("SELECT status, count(*) AS n FROM flags GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages", Long::class.java),
        jdbc.queryForObject("SELECT count(*) FROM admin_decision_log", Long::class.java),
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2010-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
    }
}
