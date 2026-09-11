package com.quickpitik.mobile.data.local

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Covers the upload queue's persistence contract — the table every tether and
 * card-import path writes through, and the one place where a wrong query means
 * a photographer silently loses (or re-uploads) a frame.
 *
 * Built on an in-memory database rather than [AppDatabase.getDatabase], whose
 * singleton would leak rows between tests.
 */
@RunWith(RobolectricTestRunner::class)
class UploadQueueDaoTest {

    private lateinit var db: AppDatabase
    private lateinit var dao: UploadQueueDao

    private fun record(
        eventId: String = EVENT,
        status: String = "QUEUED",
        filePath: String = "/cache/frame.jpg",
        retryCount: Int = 0,
    ) = UploadRecord(
        filePath = filePath,
        eventId = eventId,
        photographerId = "shooter@example.com",
        captureTimestamp = 1_700_000_000_000L,
        uploadStatus = status,
        retryCount = retryCount,
    )

    @Before
    fun setUp() {
        db = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(),
            AppDatabase::class.java,
        ).allowMainThreadQueries().build()
        dao = db.uploadQueueDao()
    }

    @After
    fun tearDown() = db.close()

    @Test fun `cursor drains more than a thousand photos without retrying earlier rows`() = runTest {
        repeat(1005) { dao.insertRecord(record()) }
        dao.insertRecord(record().copy(photographerId = "other@example.com"))
        var cursor = 0L
        val visited = mutableListOf<Long>()
        while (true) {
            val batch = dao.getOwnedRecords("shooter@example.com", "QUEUED", cursor)
            if (batch.isEmpty()) break
            assertTrue(batch.size <= 24)
            visited.addAll(batch.map { it.id })
            cursor = batch.last().id
        }
        assertEquals(1005, visited.size)
        assertEquals(1005, visited.toSet().size)
        assertEquals(24, dao.getOwnedRecords("shooter@example.com", "QUEUED", 0).size)
    }

    @Test
    fun `insert autogenerates an id and round-trips every column`() = runTest {
        val id = dao.insertRecord(record(filePath = "/cache/DSC_0001.jpg"))
        assertTrue("autoGenerate should hand out a positive id", id > 0)

        val stored = dao.getRecordById(id)
        assertNotNull(stored)
        requireNotNull(stored)
        assertEquals("/cache/DSC_0001.jpg", stored.filePath)
        assertEquals(EVENT, stored.eventId)
        assertEquals("shooter@example.com", stored.photographerId)
        assertEquals(1_700_000_000_000L, stored.captureTimestamp)
        assertEquals("QUEUED", stored.uploadStatus)
        assertEquals(0, stored.retryCount)
        assertNull(stored.errorMessage)
    }

    @Test
    fun `getRecordsWithStatus filters by status and orders newest first`() = runTest {
        val first = dao.insertRecord(record(filePath = "/cache/a.jpg"))
        dao.insertRecord(record(filePath = "/cache/b.jpg", status = "COMPLETED"))
        val third = dao.insertRecord(record(filePath = "/cache/c.jpg"))

        val queued = dao.getRecordsWithStatus("QUEUED")

        assertEquals(listOf(third, first), queued.map { it.id })
    }

    @Test
    fun `requeueFailed resets only failed rows and reports the count`() = runTest {
        val failed = dao.insertRecord(record(status = "FAILED", retryCount = 3))
        dao.updateStatus(failed, "FAILED", "Upload rejected by server.")
        val completed = dao.insertRecord(record(status = "COMPLETED"))
        val queued = dao.insertRecord(record(status = "QUEUED"))

        val requeued = dao.requeueFailed()

        assertEquals(1, requeued)
        val revived = requireNotNull(dao.getRecordById(failed))
        assertEquals("QUEUED", revived.uploadStatus)
        assertEquals(0, revived.retryCount)
        assertNull(revived.errorMessage)
        assertEquals("COMPLETED", requireNotNull(dao.getRecordById(completed)).uploadStatus)
        assertEquals("QUEUED", requireNotNull(dao.getRecordById(queued)).uploadStatus)
    }

    @Test
    fun `updateStatus writes both the status and the error message`() = runTest {
        val id = dao.insertRecord(record())

        dao.updateStatus(id, "FAILED", "Upload rejected by server.")

        val stored = requireNotNull(dao.getRecordById(id))
        assertEquals("FAILED", stored.uploadStatus)
        assertEquals("Upload rejected by server.", stored.errorMessage)
    }

    @Test
    fun `incrementRetryCount bumps in place without touching other columns`() = runTest {
        val id = dao.insertRecord(record(retryCount = 2))

        dao.incrementRetryCount(id)
        dao.incrementRetryCount(id)

        val stored = requireNotNull(dao.getRecordById(id))
        assertEquals(4, stored.retryCount)
        assertEquals("QUEUED", stored.uploadStatus)
    }

    @Test
    fun `deleteByStatus removes only that status and reports how many went`() = runTest {
        dao.insertRecord(record(status = "FAILED"))
        dao.insertRecord(record(status = "FAILED"))
        val survivor = dao.insertRecord(record(status = "QUEUED"))

        val deleted = dao.deleteByStatus("FAILED")

        assertEquals(2, deleted)
        assertEquals(listOf(survivor), dao.getAllRecords().first().map { it.id })
    }

    /**
     * The re-import guard. FAILED is deliberately excluded so a photographer can
     * retry a frame from the card after a transient backend hiccup — if this
     * query ever starts matching FAILED, that retry silently stops working.
     * DUPLICATE blocks: the backend refuses those bytes every time.
     */
    @Test
    fun `getActiveOrCompletedForEvent covers in-flight, landed and duplicates but never failed`() = runTest {
        val queued = dao.insertRecord(record(status = "QUEUED"))
        val uploading = dao.insertRecord(record(status = "UPLOADING"))
        val completed = dao.insertRecord(record(status = "COMPLETED"))
        val duplicate = dao.insertRecord(record(status = "DUPLICATE"))
        dao.insertRecord(record(status = "FAILED"))

        val blocking = dao.getActiveOrCompletedForEvent(EVENT).map { it.id }.toSet()

        assertEquals(setOf(queued, uploading, completed, duplicate), blocking)
    }

    /** Only the backend's "already in another event" refusal counts; the row and its message stay. */
    @Test
    fun `markFailedDuplicates settles only this owner's duplicate failures`() = runTest {
        val dup = dao.insertRecord(record(status = "FAILED"))
        dao.updateStatus(dup, "FAILED", DUPLICATE)
        val otherFailure = dao.insertRecord(record(status = "FAILED"))
        dao.updateStatus(otherFailure, "FAILED", "Upload rejected by server.")
        val otherOwner = dao.insertRecord(record(status = "FAILED").copy(photographerId = "other@example.com"))
        dao.updateStatus(otherOwner, "FAILED", DUPLICATE)

        assertEquals(1, dao.countFailedDuplicates("shooter@example.com").first())
        assertEquals(1, dao.markFailedDuplicates("shooter@example.com"))

        val settled = requireNotNull(dao.getRecordById(dup))
        assertEquals("DUPLICATE", settled.uploadStatus)
        assertEquals(DUPLICATE, settled.errorMessage)
        assertEquals("FAILED", requireNotNull(dao.getRecordById(otherFailure)).uploadStatus)
        assertEquals("FAILED", requireNotNull(dao.getRecordById(otherOwner)).uploadStatus)
        assertEquals(0, dao.countFailedDuplicates("shooter@example.com").first())
        // Retry never revives a duplicate: only the other failure goes back to the queue.
        assertEquals(1, dao.requeueFailed("shooter@example.com"))
        assertEquals("DUPLICATE", requireNotNull(dao.getRecordById(dup)).uploadStatus)
    }

    @Test
    fun `getActiveOrCompletedForEvent is scoped to one event`() = runTest {
        val mine = dao.insertRecord(record(eventId = EVENT))
        dao.insertRecord(record(eventId = "other-event"))

        val blocking = dao.getActiveOrCompletedForEvent(EVENT)

        assertEquals(listOf(mine), blocking.map { it.id })
    }

    @Test
    fun `pruning removes only completed rows with verified local deletion older than cutoff`() = runTest {
        val old = 1_600_000_000_000L
        val cutoff = 1_650_000_000_000L
        dao.insertRecord(record(status = "COMPLETED", filePath = "/spool/old-done.jpg").copy(captureTimestamp = old, localDeletedAt = old))
        val oldQueued = dao.insertRecord(record(status = "QUEUED", filePath = "/spool/old-queued.jpg").copy(captureTimestamp = old))
        val newDone = dao.insertRecord(record(status = "COMPLETED", filePath = "/spool/new-done.jpg"))

        assertEquals(1, dao.deleteCompletedBefore(cutoff))
        assertEquals(setOf(oldQueued, newDone), dao.getAllRecords().first().map { it.id }.toSet())
        assertEquals(mapOf("QUEUED" to 1, "COMPLETED" to 1), dao.getStatusCounts().first().associate { it.status to it.count })
    }

    @Test
    fun `getAllRecords emits the current queue`() = runTest {
        assertTrue(dao.getAllRecords().first().isEmpty())

        val id = dao.insertRecord(record())

        assertEquals(listOf(id), dao.getAllRecords().first().map { it.id })
    }

    private companion object {
        const val EVENT = "11111111-1111-1111-1111-111111111111"
        const val DUPLICATE = "This photo already exists in your event 'TEST MOBILE'."
    }
}
