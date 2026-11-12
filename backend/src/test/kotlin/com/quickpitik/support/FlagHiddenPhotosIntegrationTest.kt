package com.quickpitik.support

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.admin.AdminFlagService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.sql.DataSource
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

// PE-2.4: the batch hide's record (flag_hidden_photos) and the one restore rule, against real Postgres.
// Every test builds its own users, events and flags, so nothing assumes an empty table.
class FlagHiddenPhotosIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var flags: FlagRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var flagAdmin: AdminFlagService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var dataSource: DataSource
    @Autowired private lateinit var txManager: PlatformTransactionManager

    private fun <T> tx(block: () -> T): T = TransactionTemplate(txManager).execute { block() }!!

    @Test
    fun `recording twice makes one row per photo and hides each once`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me)
        photo(e, me); photo(e, me)

        assertEquals(2, tx { flags.recordBatchHide(flag, e, me, NOW) })
        assertEquals(0, tx { flags.recordBatchHide(flag, e, me, NOW.plusHours(1) ) })
        assertEquals(2, count("SELECT count(*) FROM flag_hidden_photos WHERE flag_id = ?", flag))
        assertEquals(2, tx { photos.hideRecordedForFlag(flag) })
        assertEquals(0, tx { photos.hideRecordedForFlag(flag) })
    }

    @Test
    fun `a second hide batch leaves restored photos on sale`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me)
        val kept = photo(e, me); val back = photo(e, me)
        tx { flags.recordBatchHide(flag, e, me, NOW); photos.hideRecordedForFlag(flag) }
        tx { flags.markRestored(flag, back, NOW.plusHours(1)) }
        jdbc.update("UPDATE photos SET status = 'LIVE' WHERE id = ?", back)

        assertEquals(0, tx { flags.recordBatchHide(flag, e, me, NOW.plusHours(2)) })
        assertEquals(0, tx { photos.hideRecordedForFlag(flag) })

        assertEquals("LIVE", status(back))
        assertEquals("HIDDEN", status(kept))
        assertNotNull(restoredAt(flag, back))
    }

    @Test
    fun `a photo that was already hidden is recorded and its status is left alone`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me)
        val alreadyHidden = photo(e, me, status = "HIDDEN"); val live = photo(e, me)

        assertEquals(2, tx { flags.recordBatchHide(flag, e, me, NOW) })
        assertEquals(1, tx { photos.hideRecordedForFlag(flag) }) // only the live one changes
        assertEquals(setOf(alreadyHidden, live), recorded(flag).toSet())
        assertEquals("HIDDEN", status(alreadyHidden))
    }

    @Test
    fun `the bulk restore skips photos another flag still holds`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me)
        val byPhotoFlag = photo(e, me); val byOtherBatch = photo(e, me)
        val noPreview = photo(e, me, watermark = null); val withPreview = photo(e, me)
        tx { flags.recordBatchHide(flag, e, me, NOW); photos.hideRecordedForFlag(flag) }
        jdbc.update("INSERT INTO flags (target_kind, target_id, reason, status) VALUES ('photo', ?, 'other', 'hidden')", byPhotoFlag)
        val other = relevanceFlag(me)
        jdbc.update("INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at) VALUES (?, ?, ?)", other, byOtherBatch, NOW)
        jdbc.update("UPDATE photos SET processing_attempts = 3 WHERE id = ?", noPreview)

        assertEquals(2, tx { photos.unhideRecordedForFlag(flag) })

        assertEquals("HIDDEN", status(byPhotoFlag))
        assertEquals("HIDDEN", status(byOtherBatch))
        assertEquals("PROCESSING", status(noPreview))
        assertEquals("LIVE", status(withPreview))
        assertEquals(0, count("SELECT processing_attempts FROM photos WHERE id = ?", noPreview))
    }

    @Test
    fun `trashed photos are neither recorded nor hidden`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me)
        val trashed = photo(e, me, deleted = true); val live = photo(e, me)
        photo(e, user()) // another photographer's photo in the same event
        photo(event(), me) // the same photographer's photo in another event

        assertEquals(1, tx { flags.recordBatchHide(flag, e, me, NOW) })
        assertEquals(listOf(live), recorded(flag))
        tx { photos.hideRecordedForFlag(flag) }
        assertEquals("LIVE", status(trashed))
    }

    @Test
    fun `marking one photo restored stamps only that row`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me)
        val a = photo(e, me); val b = photo(e, me)
        tx { flags.recordBatchHide(flag, e, me, NOW) }

        assertEquals(1, tx { flags.markRestored(flag, a, NOW.plusHours(1)) })
        assertEquals(0, tx { flags.markRestored(flag, a, NOW.plusHours(2)) }) // a second restore changes nothing

        assertNotNull(restoredAt(flag, a))
        assertNull(restoredAt(flag, b))
    }

    @Test
    fun `dismissing stamps every un-restored row of that flag and no other flag's`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me); val other = relevanceFlag(user())
        val a = photo(e, me); val b = photo(e, me)
        tx { flags.recordBatchHide(flag, e, me, NOW) }
        jdbc.update("INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at) VALUES (?, ?, ?)", other, a, NOW)
        tx { flags.markRestored(flag, b, NOW.plusHours(1)) }

        assertEquals(1, tx { flags.markAllRestored(flag, NOW.plusHours(2)) })

        assertNotNull(restoredAt(flag, a))
        assertEquals(NOW.plusHours(1).toInstant(), restoredAt(flag, b)) // the earlier stamp stays
        assertNull(restoredAt(other, a))
    }

    @Test
    fun `another batch hold is seen only while it is not restored`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me); val other = relevanceFlag(me)
        val p = photo(e, me)
        jdbc.update("INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at) VALUES (?, ?, ?)", other, p, NOW)

        assertTrue(flags.existsOtherBatchHold(p, flag))
        assertTrue(!flags.existsOtherBatchHold(p, other)) // its own row is not "another"
        tx { flags.markRestored(other, p, NOW) }
        assertTrue(!flags.existsOtherBatchHold(p, flag))
    }

    @Test
    fun `unrestored photo ids come sorted and leave out restored rows and other flags`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me)
        val a = photo(e, me); val b = photo(e, me); val c = photo(e, me)
        tx { flags.recordBatchHide(flag, e, me, NOW) }
        tx { flags.markRestored(flag, c, NOW) }
        jdbc.update("INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at) VALUES (?, ?, ?)", relevanceFlag(me), photo(e, me), NOW)

        assertEquals(listOf(a, b).map { it.toString() }.sorted(), flags.unrestoredPhotoIds(flag).map { it.toString() }) // Postgres orders uuids bytewise = by text
        assertTrue(flags.isRecorded(flag, c))
        assertTrue(!flags.isRecorded(flag, UUID.randomUUID()))
    }

    @Test
    fun `hidden photos page in a stable order`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me)
        val ids = List(5) { photo(e, me) }
        tx { flags.recordBatchHide(flag, e, me, NOW) } // one timestamp for all: the photo id breaks the tie
        tx { flags.markRestored(flag, ids[0], NOW.plusHours(1)) }
        jdbc.update("UPDATE photos SET watermark_s3_key = NULL, thumbnail_s3_key = 'thumb/x.jpg' WHERE id = ?", ids[1])

        val first = flags.pageHiddenPhotos(flag, OffsetLimitPageable(PaginationParams.of(0, 3)))
        val second = flags.pageHiddenPhotos(flag, OffsetLimitPageable(PaginationParams.of(3, 3)))

        assertEquals(5L, first.totalElements)
        assertEquals(ids.map { it.toString() }.sorted(), (first.content + second.content).map { it.photoId.toString() })
        val rows = (first.content + second.content).associateBy { it.photoId }
        assertEquals(NOW.toInstant(), rows.getValue(ids[2]).hiddenAt)
        assertNull(rows.getValue(ids[2]).restoredAt)
        assertEquals(NOW.plusHours(1).toInstant(), rows.getValue(ids[0]).restoredAt) // restored rows stay in the list
        assertEquals("thumb/x.jpg", rows.getValue(ids[1]).thumbnailKey)
        assertEquals(e, rows.getValue(ids[2]).eventId)
    }

    @Test
    fun `a batch is hidden, one photo restored and the rest restored on dismiss through the service`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me); val admin = user("ADMIN")
        batch(e, me, flag)
        val a = photo(e, me); val b = photo(e, me); val c = photo(e, me)

        val dto = flagAdmin.hideBatch(admin, flag, e, "copied photos")
        assertEquals("open", dto.status)
        assertEquals(listOf("HIDDEN", "HIDDEN", "HIDDEN"), listOf(a, b, c).map { status(it) })
        assertEquals(1, count("SELECT count(*) FROM admin_decision_log WHERE target_user_id = ? AND decision = 'batch_hidden'", me))

        flagAdmin.restoreHiddenPhoto(admin, flag, a)
        assertEquals("LIVE", status(a))
        assertEquals("HIDDEN", status(b))

        flagAdmin.dismiss(admin, flag, "false alarm")
        assertEquals(listOf("LIVE", "LIVE", "LIVE"), listOf(a, b, c).map { status(it) })
        assertEquals(0, count("SELECT count(*) FROM flag_hidden_photos WHERE flag_id = ? AND restored_at IS NULL", flag))
        assertEquals(3L, flagAdmin.listHiddenPhotos(flag, PaginationParams.of(0, 10)).total)
    }

    // ─── Two admins at once (R2-04): the answer must not depend on who gets there first ───

    @Test
    fun `a photo flag and the relevance flag dismissed together leave the photo on sale`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me); val admin = user("ADMIN")
        batch(e, me, flag)
        val p = photo(e, me)
        flagAdmin.hideBatch(admin, flag, e, null)
        val photoFlag = jdbc.queryForObject(
            "INSERT INTO flags (target_kind, target_id, reason, status) VALUES ('photo', ?, 'other', 'hidden') RETURNING id", UUID::class.java, p,
        )!!

        val results = race({ flagAdmin.dismiss(admin, photoFlag, null) }, { flagAdmin.dismiss(admin, flag, null) })

        assertTrue(results.all { it.isSuccess }, results.toString())
        assertEquals("LIVE", status(p)) // each rule read saw the other's row; a lost update would strand it HIDDEN
    }

    @Test
    fun `a photo flag hidden while the relevance flag is dismissed keeps the photo hidden`() {
        val me = user(); val e = event(); val flag = relevanceFlag(me); val admin = user("ADMIN")
        batch(e, me, flag)
        val p = photo(e, me)
        flagAdmin.hideBatch(admin, flag, e, null)
        val photoFlag = jdbc.queryForObject(
            "INSERT INTO flags (target_kind, target_id, reason) VALUES ('photo', ?, 'other') RETURNING id", UUID::class.java, p,
        )!!

        val results = race({ flagAdmin.hide(admin, photoFlag, null) }, { flagAdmin.dismiss(admin, flag, null) })

        assertTrue(results.all { it.isSuccess }, results.toString())
        assertEquals("HIDDEN", status(p))
    }

    // Both transactions queue on a table lock the test holds, so they start together and the later one must
    // see the earlier one's committed state after waiting.
    private fun race(first: () -> Any?, second: () -> Any?): List<Result<Any?>> {
        val pool = Executors.newFixedThreadPool(2)
        val barrier = CyclicBarrier(2)
        try {
            dataSource.connection.use { lock ->
                lock.autoCommit = false
                lock.createStatement().use { it.execute("LOCK TABLE flags IN SHARE MODE") }
                val futures = listOf(first, second).map { action ->
                    pool.submit<Result<Any?>> { runCatching { barrier.await(10, TimeUnit.SECONDS); action() } }
                }
                try {
                    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
                    while (count("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'") < 2) {
                        check(System.nanoTime() < deadline) { "Expected 2 PostgreSQL lock waits" }
                        Thread.sleep(20)
                    }
                } finally { lock.commit() }
                return futures.map { it.get(25, TimeUnit.SECONDS) }
            }
        } finally { pool.shutdownNow() }
    }

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun recorded(flag: UUID): List<UUID> =
        jdbc.queryForList("SELECT photo_id FROM flag_hidden_photos WHERE flag_id = ? ORDER BY photo_id", UUID::class.java, flag)

    private fun restoredAt(flag: UUID, photo: UUID) = jdbc.queryForObject(
        "SELECT restored_at FROM flag_hidden_photos WHERE flag_id = ? AND photo_id = ?", java.sql.Timestamp::class.java, flag, photo,
    )?.toInstant()

    private fun count(sql: String, vararg params: Any?): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private fun user(role: String = "PHOTOGRAPHER"): UUID = id(
        "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Hide', ?)", "hide-${UUID.randomUUID()}@test.local", role,
    )

    private fun event(): UUID = id(
        "INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Hide Run', ?, 'Cebu', 'ACTIVE')",
        "hide-${UUID.randomUUID()}", LocalDate.of(2002, 6, 9),
    )

    private fun relevanceFlag(photographer: UUID): UUID =
        id("INSERT INTO flags (target_kind, target_id, reason) VALUES ('user', ?, 'batch_relevance_low')", photographer)

    private fun batch(event: UUID, photographer: UUID, flag: UUID): UUID =
        id("INSERT INTO batch_relevance (event_id, photographer_id, flag_id) VALUES (?, ?, ?)", event, photographer, flag)

    private fun photo(
        event: UUID,
        photographer: UUID,
        status: String = "LIVE",
        deleted: Boolean = false,
        watermark: String? = "wm/x.jpg",
    ): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, watermark_s3_key, price_php, uploaded_at, status, deleted_at) " +
            "VALUES (?, ?, ?, ?, 125, ?, ?, ?)",
        event, photographer, "hide/${UUID.randomUUID()}", watermark, NOW, status, if (deleted) NOW else null,
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2002-06-09T08:00:00Z")
    }
}
