package com.quickpitik.support

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-1.10 (plan 5.11): the admin Photo Relevance page queries against real Postgres. The container is shared, so
// fixtures sit in 2003 and assertions look only at this class's own rows.
class AdminRelevanceQueriesIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `each tab lists only its current cases, newest first`() {
        val b = batch()
        val hidden = case(b, "APPLIED", "JUNK_DUMP", "HIDE_ALL", appliedAt = NOW)
        val b2 = batch()
        val cantTell = case(b2, "APPLIED", "CANT_TELL", "NONE", appliedAt = NOW.plusMinutes(1))
        val closed = case(b2, "CLOSED", "CANT_TELL", "NONE", appliedAt = NOW.plusMinutes(2))
        val inFlight = case(b2, "DECIDED", "MIXED", null, appliedAt = null)
        val mine = setOf(hidden, cantTell, closed, inFlight)

        fun tab(t: String) = cases.pageCurrentCases(t, OffsetLimitPageable(0, 200)).content.map { it.caseId }.filter { it in mine }

        assertEquals(listOf(hidden), tab("hidden"))
        assertEquals(listOf(cantTell), tab("cant_tell"))
        assertEquals(listOf(cantTell, hidden), tab("all"))
        val row = cases.caseRow(hidden)!!
        assertEquals(listOf("Rel Run", "rel-${b.photographer}", "Rel", 600), listOf(row.eventName, row.photographerHandle, row.photographerName, row.batchPhotos))
        assertTrue(cases.pageCurrentCases("all", OffsetLimitPageable(0, 1)).totalElements >= 2)
        assertNull(cases.caseRow(UUID.randomUUID()))
    }

    @Test
    fun `the measured sample leaves out unmeasured, failed, trashed and removal photos and carries no original key`() {
        val b = batch()
        val c = case(b, "APPLIED", "MIXED", "HIDE_JUNK", appliedAt = NOW)
        val measured = photo(b, c, sharpness = 12.0)
        photo(b, c, sharpness = null)
        photo(b, c, sharpness = 30.0, error = "BAD_IMAGE")
        val trashed = photo(b, c, sharpness = 50.0)
        jdbc.update("UPDATE photos SET deleted_at = ? WHERE id = ?", NOW, trashed)
        val underRemoval = photo(b, c, sharpness = 60.0)
        jdbc.update("UPDATE photos SET removal_state = 'REVIEW' WHERE id = ?", underRemoval)

        val rows = cases.measuredSample(c)

        assertEquals(listOf(measured), rows.map { it.photoId })
        assertEquals(listOf("wm/$measured", "th/$measured", 12.0), rows.single().let { listOf(it.watermarkKey, it.thumbnailKey, it.fgSharpness) })
    }

    @Test
    fun `the unseen alert list leaves out seen alerts and a batch lists its own`() {
        val b = batch()
        val c = case(b, "APPLIED", "COPIED", "HIDE_ALL", appliedAt = NOW)
        val seen = alert(b, c, "BIG_HIDE", seenAt = NOW)
        val unseen = alert(b, c, "COPIED", seenAt = null)
        val mine = setOf(seen, unseen)

        fun page(unseenOnly: Boolean) = alerts.pageAlerts(unseenOnly, OffsetLimitPageable(0, 200)).content.map { it.alertId }.filter { it in mine }

        assertEquals(listOf(unseen), page(true))
        assertEquals(mine, page(false).toSet())
        assertEquals(mine, alerts.alertsForBatch(b.id).map { it.alertId }.toSet())
        assertEquals("Rel Run", alerts.alertRow(unseen)!!.eventName)
    }

    @Test
    fun `marking an alert seen stamps it once and keeps the first admin`() {
        val b = batch()
        val a = alert(b, case(b, "APPLIED", "COPIED", "HIDE_ALL", appliedAt = NOW), "COPIED", seenAt = null)
        val first = user("ADMIN"); val second = user("ADMIN")

        assertEquals(1, tx.execute { alerts.markSeen(a, first, NOW) })
        assertEquals(0, tx.execute { alerts.markSeen(a, second, NOW.plusHours(1)) })

        val row = jdbc.queryForMap("SELECT seen_by, seen_at FROM relevance_alerts WHERE id = ?", a)
        assertEquals(first, row["seen_by"])
        assertEquals(NOW.toInstant(), (row["seen_at"] as java.sql.Timestamp).toInstant())
        assertFalse(alerts.alertRow(a)!!.seenAt == null)
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun batch(): Batch {
        val photographer = user("PHOTOGRAPHER")
        jdbc.update("INSERT INTO photographer_settings (user_id, handle) VALUES (?, ?)", photographer, "rel-$photographer")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2003-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "reladmin-${UUID.randomUUID()}")
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer), event, photographer)
    }

    private fun user(role: String): UUID =
        id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', ?)", "reladmin-${UUID.randomUUID()}@test.local", role)

    private fun case(b: Batch, stage: String, verdict: String, action: String?, appliedAt: OffsetDateTime?): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, measured, junk, verdict, decided_at, " +
            "applied_mode, action, applied_at) VALUES (?, 'QUIET', ?, 600, ?, 300, 30, ?, ?, ?, ?, ?)",
        b.id, NOW, stage, verdict, NOW, action?.let { "RECORD_ONLY" }, action, appliedAt,
    )

    private fun photo(b: Batch, caseId: UUID, sharpness: Double?, error: String? = null): UUID {
        val photo = id("INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, 125, 'LIVE')",
            b.event, b.photographer, "original/${UUID.randomUUID()}")
        jdbc.update("UPDATE photos SET watermark_s3_key = ?, thumbnail_s3_key = ? WHERE id = ?", "wm/$photo", "th/$photo", photo)
        jdbc.update("INSERT INTO relevance_case_photos (case_id, photo_id, round) VALUES (?, ?, 1)", caseId, photo)
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, sharpness, brightness, person_conf, race_conf, measure_error) " +
            "VALUES (?, ?, ?, ?, ?, 50, 0.9, 0.9, ?)", photo, b.id, if (sharpness == null && error == null) null else NOW, sharpness, sharpness, error)
        return photo
    }

    private fun alert(b: Batch, caseId: UUID, kind: String, seenAt: OffsetDateTime?): UUID = id(
        "INSERT INTO relevance_alerts (batch_id, case_id, kind, dedupe_key, would_have, photos, created_at, seen_at) VALUES (?, ?, ?, ?, true, 600, ?, ?)",
        b.id, caseId, kind, "${kind.lowercase()}:$caseId", NOW, seenAt,
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2003-03-04T05:06:07.123456Z")
    }
}
