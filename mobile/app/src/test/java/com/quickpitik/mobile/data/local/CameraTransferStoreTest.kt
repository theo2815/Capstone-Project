package com.quickpitik.mobile.data.local

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.usb.ptp.PtpObjectInfo
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadows.ShadowStatFs
import java.io.File
import java.io.IOException

@RunWith(RobolectricTestRunner::class)
class CameraTransferStoreTest {
    private lateinit var context: Context
    private lateinit var db: AppDatabase
    private lateinit var store: CameraTransferStore
    private val jpeg = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 0xff.toByte(), 0xd9.toByte())
    private fun photo(handle: Long, name: String = "IMG_0001.JPG", camera: String = "camera-a") =
        store.transfer("owner", "event", camera, handle, PtpObjectInfo(0x3801, 4, name, 1, 1, "20260907T120000"))

    @Before fun setup() {
        context = ApplicationProvider.getApplicationContext()
        ShadowStatFs.registerStats(context.filesDir.absolutePath, 4_000_000, 2_000_000, 2_000_000)
        db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).allowMainThreadQueries().build()
        store = CameraTransferStore(context, db)
        UploadSpool.dir(context).listFiles()?.filter { it.name.startsWith("camera_") }?.forEach { it.delete() }
    }
    @After fun close() { db.close() }

    @Test fun `failed lower handle survives later success and a new store instance`() = runTest {
        val baseline = photo(10)
        store.reconcile(baseline.scope, listOf(baseline))
        val failed = photo(11)
        val succeeded = photo(12)
        store.reconcile(baseline.scope, listOf(baseline, failed, succeeded))
        repeat(4) { store.fail(failed, "Cable interrupted") }
        store.save(succeeded) { it.write(jpeg) }
        val reopened = CameraTransferStore(context, db)
        val rows = reopened.reconcile(baseline.scope, listOf(baseline, failed, succeeded))
        assertEquals("FAILED", rows.first { it.handle == 11L }.status)
        assertEquals(4, rows.first { it.handle == 11L }.attempts)
        assertEquals("QUEUED", rows.first { it.handle == 12L }.status)
    }

    @Test fun `more than 200 catchup objects remain pending`() = runTest {
        val old = photo(1)
        store.reconcile(old.scope, listOf(old))
        val rows = store.reconcile(old.scope, (1L..302L).map { photo(it) })
        assertEquals(301, rows.count { it.status == "PENDING" })
    }

    @Test fun `manual selection survives interruption without importing unrelated baseline photos`() = runTest {
        val selected = photo(1)
        val untouched = photo(2)
        val later = photo(3)
        store.reconcile(selected.scope, listOf(selected, untouched, later))
        store.selectForImport(listOf(selected, later))
        store.save(selected) { it.write(jpeg) }
        val reopened = CameraTransferStore(context, db)
        val rows = reopened.reconcile(selected.scope, listOf(selected, untouched, later))
        assertEquals("QUEUED", rows.first { it.handle == 1L }.status)
        assertEquals("BASELINE", rows.first { it.handle == 2L }.status)
        assertEquals("PENDING", rows.first { it.handle == 3L }.status)
    }

    @Test fun `safe file naming ignores path supplied by camera`() = runTest {
        val record = photo(1, "../../outside.jpg")
        val id = store.save(record) { it.write(jpeg) }
        val upload = requireNotNull(db.uploadQueueDao().getRecordById(id))
        assertEquals(UploadSpool.dir(context).canonicalFile, File(upload.filePath).canonicalFile.parentFile)
        assertEquals("../../outside.jpg", upload.originalFilename)
        assertArrayEquals(jpeg, File(upload.filePath).readBytes())
        assertNotNull(upload.contentHash)
    }

    @Test fun `truncated transfer is never enqueued`() = runTest {
        val record = photo(1)
        try { store.save(record) { it.write(jpeg, 0, 2) }; fail("Expected validation failure") }
        catch (_: IOException) { }
        assertTrue(db.uploadQueueDao().getRecordsWithStatus("QUEUED").isEmpty())
        assertEquals("FAILED", store.dao.get(record.key)?.status)
        assertFalse(UploadSpool.dir(context).listFiles().orEmpty().any { it.name.endsWith(".part") })
    }

    @Test fun `same filename does not suppress different bytes and content dedupe is account scoped`() = runTest {
        val first = store.save(photo(1)) { it.write(jpeg) }
        val otherBytes = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 1, 2)
        val second = store.save(photo(2, camera = "camera-b")) { it.write(otherBytes) }
        assertNotEquals(first, second)
        val duplicate = store.save(photo(3)) { it.write(jpeg) }
        assertEquals(first, duplicate)
        val otherOwner = photo(4).copy(key = "other-owner", owner = "other")
        assertNotEquals(first, store.save(otherOwner) { it.write(jpeg) })
    }

    @Test fun `crash after rename is recovered without rereading camera`() = runTest {
        val record = photo(1)
        val file = File(UploadSpool.dir(context), "camera_${record.key}.jpg").apply { writeBytes(jpeg) }
        store.dao.insert(listOf(record.copy(status = "TRANSFERRING", filePath = file.absolutePath)))
        store.recover("owner")
        assertEquals("QUEUED", store.dao.get(record.key)?.status)
        assertEquals(1, db.uploadQueueDao().getRecordsWithStatus("QUEUED").size)
        store.recover("owner")
        assertEquals(1, db.uploadQueueDao().getRecordsWithStatus("QUEUED").size)
    }

    @Test fun `missing queued file cannot discard a newly downloaded duplicate`() = runTest {
        val oldId = store.save(photo(1)) { it.write(jpeg) }
        File(requireNotNull(db.uploadQueueDao().getRecordById(oldId)).filePath).delete()
        val replacementId = store.save(photo(2)) { it.write(jpeg) }
        assertNotEquals(oldId, replacementId)
        assertArrayEquals(jpeg, File(requireNotNull(db.uploadQueueDao().getRecordById(replacementId)).filePath).readBytes())
    }

    @Test fun `version one migration preserves queued originals and validates schema`() = runTest {
        val name = "camera-migration-test"
        context.deleteDatabase(name)
        val file = context.getDatabasePath(name)
        file.parentFile?.mkdirs()
        SQLiteDatabase.openOrCreateDatabase(file, null).use { old ->
            old.execSQL("CREATE TABLE upload_queue (id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, filePath TEXT NOT NULL, eventId TEXT NOT NULL, photographerId TEXT NOT NULL, captureTimestamp INTEGER NOT NULL, uploadStatus TEXT NOT NULL, retryCount INTEGER NOT NULL, errorMessage TEXT)")
            old.execSQL("INSERT INTO upload_queue VALUES (1, '/original.jpg', 'event', 'owner', 1, 'QUEUED', 0, NULL)")
            old.version = 1
        }
        val migrated = Room.databaseBuilder(context, AppDatabase::class.java, name)
            .addMigrations(AppDatabase.MIGRATION_1_2, AppDatabase.MIGRATION_2_3, AppDatabase.MIGRATION_3_4).allowMainThreadQueries().build()
        try {
            val record = requireNotNull(migrated.uploadQueueDao().getRecordById(1))
            assertEquals("/original.jpg", record.filePath)
            assertEquals("QUEUED", record.uploadStatus)
            assertNull(record.sourceKey)
            assertNull(record.uploadedPhotoId)
            assertNull(record.uploadedBackend)
            assertNull(record.localDeletedAt)
            assertTrue(migrated.uploadQueueDao().cleanupCandidates("owner", "server").isEmpty())
            assertEquals(0, migrated.cameraTransferDao().baselineExists("scope"))
        } finally { migrated.close(); context.deleteDatabase(name) }
    }

    @Test fun `low volume space prevents a camera read and resumes after space is freed`() = runTest {
        val record = photo(1)
        store.selectForImport(listOf(record))
        ShadowStatFs.registerStats(context.filesDir.absolutePath, 4_000_000, 0, 0)
        assertEquals(UploadSpool.Level.BLOCKED, UploadSpool.space(context).level)
        try { store.save(record) { fail("Camera must not be read when storage is full") }; fail("Expected pause") }
        catch (_: LowCameraStorage) { }
        assertEquals("PENDING", store.dao.get(record.key)?.status)
        assertEquals(0, store.dao.get(record.key)?.attempts)
        ShadowStatFs.registerStats(context.filesDir.absolutePath, 4_000_000, 2_000_000, 2_000_000)
        assertEquals(UploadSpool.Level.READY, UploadSpool.space(context).level)
        val id = store.save(record) { it.write(jpeg) }
        assertArrayEquals(jpeg, File(requireNotNull(db.uploadQueueDao().getRecordById(id)).filePath).readBytes())
    }

    @Test fun `ENOSPC during camera write preserves retry intent without spending attempts`() = runTest {
        val record = photo(1)
        try {
            store.save(record) {
                it.write(jpeg, 0, 2)
                throw IOException("write failed", android.system.ErrnoException("write", android.system.OsConstants.ENOSPC))
            }
            fail("Expected storage pause")
        } catch (e: LowCameraStorage) { assertTrue(e.transferStarted) }
        assertEquals("PENDING", store.dao.get(record.key)?.status)
        assertEquals(0, store.dao.get(record.key)?.attempts)
        assertTrue(db.uploadQueueDao().getRecordsWithStatus("QUEUED").isEmpty())
        assertFalse(UploadSpool.dir(context).listFiles().orEmpty().any { it.name.endsWith(".part") })
    }

    @Test fun `capacity accounts for next file and reserve with safe unknown and overflow bounds`() {
        val reserve = UploadSpool.MIN_FREE_BYTES
        assertFalse(UploadSpool.Space().hasRoom(4))
        assertEquals(UploadSpool.Level.UNKNOWN, UploadSpool.Space().level)
        assertFalse(UploadSpool.Space(reserve + 3).hasRoom(4))
        assertTrue(UploadSpool.Space(reserve + 4).hasRoom(4))
        assertFalse(UploadSpool.Space(Long.MAX_VALUE).hasRoom(Long.MAX_VALUE))
        assertEquals(UploadSpool.Level.LOW, UploadSpool.Space(UploadSpool.LOW_FREE_BYTES - 1).level)
        assertEquals(UploadSpool.Level.READY, UploadSpool.Space(UploadSpool.LOW_FREE_BYTES).level)
    }

    @Test fun `successive temporary frames reuse a handle without losing different content`() = runTest {
        val info = PtpObjectInfo(0x3801, 4, "DSC00001.JPG")
        val first = store.saveTransient("owner", "event", "sony", 0xffffc001L, info) { it.write(jpeg) }
        val secondBytes = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 1, 2)
        val second = store.saveTransient("owner", "event", "sony", 0xffffc001L, info) { it.write(secondBytes) }
        assertNotEquals(first, second)
        assertArrayEquals(jpeg, File(requireNotNull(db.uploadQueueDao().getRecordById(first)).filePath).readBytes())
        assertArrayEquals(secondBytes, File(requireNotNull(db.uploadQueueDao().getRecordById(second)).filePath).readBytes())
        val duplicate = store.saveTransient("owner", "event", "sony", 0xffffc001L, info) { it.write(jpeg) }
        assertEquals(first, duplicate)
    }

    @Test fun `interrupted temporary handle is retained for card recovery but never automatically retried`() = runTest {
        val info = PtpObjectInfo(0x3801, 4, "DSC00001.JPG")
        try {
            store.saveTransient("owner", "event", "sony", 0xffffc001L, info) {
                it.write(jpeg, 0, 2); throw IOException("USB detached")
            }
            fail("Expected interrupted transfer")
        } catch (_: IOException) { }
        val scope = store.transfer("owner", "event", "sony", 0xffffc001L, info).scope
        val failed = store.dao.forScope(scope).single()
        assertEquals("FAILED", failed.status)
        assertTrue(failed.error.orEmpty().contains("camera card"))
        store.dao.retry("owner", "event")
        assertEquals("FAILED", store.dao.get(failed.key)?.status)
        assertTrue(db.uploadQueueDao().getRecordsWithStatus("QUEUED").isEmpty())
    }

    @Test fun `cancelled temporary transfer recovers without replaying its handle`() = runTest {
        val info = PtpObjectInfo(0x3801, 4, "DSC00001.JPG")
        try {
            store.saveTransient("owner", "event", "sony", 0xffffc001L, info) {
                it.write(jpeg, 0, 2); throw kotlinx.coroutines.CancellationException("Stop")
            }
            fail("Expected cancellation")
        } catch (_: kotlinx.coroutines.CancellationException) { }
        val interrupted = store.dao.interrupted("owner").single()
        assertTrue(interrupted.sourceKey.startsWith("transient:"))
        assertFalse(UploadSpool.dir(context).listFiles().orEmpty().any { it.name.endsWith(".part") })
        store.recover("owner")
        store.dao.retry("owner", "event")
        assertEquals("FAILED", store.dao.get(interrupted.key)?.status)
        assertTrue(store.dao.get(interrupted.key)?.error.orEmpty().contains("camera card"))
        assertTrue(db.uploadQueueDao().getRecordsWithStatus("QUEUED").isEmpty())
        // A completed rename is a different recovery case: enqueue local bytes without USB.
        File(requireNotNull(interrupted.filePath)).writeBytes(jpeg)
        store.dao.update(interrupted)
        store.recover("owner")
        assertEquals("QUEUED", store.dao.get(interrupted.key)?.status)
    }
}
