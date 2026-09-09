package com.quickpitik.mobile.data.local

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.*
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import okio.Buffer
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
class TetherStorageCleanupTest {
    private lateinit var context: Context
    private lateinit var db: AppDatabase
    private lateinit var session: SessionManager
    private lateinit var cleanup: TetherStorageCleanup
    private lateinit var server: MockWebServer
    private val original = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 1, 2, 3, 4, 0xff.toByte(), 0xd9.toByte())
    private var response: () -> MockResponse = { MockResponse().setBody(Buffer().write(original)) }

    @Before fun setup() {
        context = ApplicationProvider.getApplicationContext()
        db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).allowMainThreadQueries().build()
        session = SessionManager.getInstance(context)
        signIn("owner")
        server = MockWebServer()
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse =
                if (request.path?.endsWith("/download") == true) MockResponse().setHeader("Content-Type", "application/json")
                    .setBody("""{"success":true,"data":{"url":"${server.url("/original")}"}}""")
                else response()
        }
        server.start()
        RetrofitClient.setBaseUrl(server.url("/").toString())
        cleanup = TetherStorageCleanup(context, db)
    }

    @After fun close() {
        db.close()
        server.shutdown()
        RetrofitClient.resetBaseUrl()
    }

    private fun signIn(owner: String) = session.saveSession("token-$owner", "PHOTOGRAPHER", "Photographer", owner)

    private suspend fun record(): UploadRecord {
        val file = File(UploadSpool.dir(context), "camera_${java.util.UUID.randomUUID()}.jpg").apply { writeBytes(original) }
        val hash = MessageDigest.getInstance("SHA-256").digest(original).joinToString("") { "%02x".format(it) }
        val row = UploadRecord(filePath = file.absolutePath, photographerId = "owner", eventId = "event",
            captureTimestamp = 1, uploadStatus = "COMPLETED", contentHash = hash, sourceKey = "camera-source",
            uploadedPhotoId = "photo", uploadedBackend = RetrofitClient.BASE_URL)
        return row.copy(id = db.uploadQueueDao().insertRecord(row))
    }

    @Test fun `only a full matching original permits cleanup and persists deletion without deleting the receipt`() = runBlocking {
        val row = record()
        val review = cleanup.review()
        assertEquals(listOf(row), review.records)
        assertEquals(original.size.toLong(), review.bytes)
        assertEquals(TetherStorageCleanup.Removal(original.size.toLong(), true), cleanup.remove(review.records.single()))
        assertFalse(File(row.filePath).exists())
        val saved = requireNotNull(db.uploadQueueDao().getRecordById(row.id))
        assertEquals("COMPLETED", saved.uploadStatus)
        assertEquals("photo", saved.uploadedPhotoId)
        assertNotNull(saved.localDeletedAt)
        assertTrue(TetherStorageCleanup(context, db).review().records.isEmpty())
        assertTrue(runCatching { cleanup.remove(row) }.isFailure)
        assertEquals("Bearer token-owner", server.takeRequest().getHeader("Authorization"))
        assertNull(server.takeRequest().getHeader("Authorization")) // Never leak the API token to storage.
    }

    @Test fun `missing truncated changed and partial cloud responses all keep the original`() = runBlocking {
        for (failure in listOf(
            { MockResponse().setResponseCode(404) },
            { MockResponse().setBody(Buffer().write(original.copyOf(3))) },
            { MockResponse().setBody(Buffer().write(original.reversedArray())) },
            { MockResponse().setResponseCode(206).setBody(Buffer().write(original)) },
            { MockResponse().setBody(Buffer().write(original + byteArrayOf(0))) },
            { MockResponse().setBody(Buffer().write(original)).setSocketPolicy(okhttp3.mockwebserver.SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY) },
        )) {
            val row = record()
            response = failure
            assertTrue(runCatching { cleanup.remove(row) }.isFailure)
            assertArrayEquals(original, File(row.filePath).readBytes())
            assertEquals(row, db.uploadQueueDao().getRecordById(row.id))
        }
    }

    @Test fun `pending failed legacy foreign and outside-spool copies never become deletion candidates`() = runBlocking {
        val changes: List<(UploadRecord) -> UploadRecord> = listOf(
            { it.copy(uploadStatus = "QUEUED") }, { it.copy(uploadStatus = "UPLOADING") },
            { it.copy(uploadStatus = "FAILED") }, { it.copy(uploadedPhotoId = null) },
            { it.copy(uploadedBackend = "https://another.invalid/") }, { it.copy(photographerId = "another") },
            { it.copy(contentHash = null) }, { it.copy(sourceKey = null) },
            { it.copy(filePath = File(context.filesDir, "gallery-original.jpg").apply { writeBytes(original) }.absolutePath) },
            { it.copy(filePath = File(UploadSpool.dir(context), "credentials.jpg").apply { writeBytes(original) }.absolutePath) },
        )
        for (change in changes) {
            val row = change(record())
            db.uploadQueueDao().updateRecord(row)
            assertTrue(runCatching { cleanup.remove(row) }.isFailure)
            assertArrayEquals(original, File(row.filePath).readBytes())
        }
        assertTrue(cleanup.review().records.isEmpty())
        assertEquals(0, server.requestCount)
    }

    @Test fun `local changes and shared queue references during verification prevent deletion`() = runBlocking {
        val changed = record()
        response = {
            File(changed.filePath).writeBytes(original.reversedArray())
            MockResponse().setBody(Buffer().write(original))
        }
        assertTrue(runCatching { cleanup.remove(changed) }.isFailure)
        assertArrayEquals(original.reversedArray(), File(changed.filePath).readBytes())
        val shared = record()
        db.uploadQueueDao().insertRecord(shared.copy(id = 0, uploadStatus = "QUEUED"))
        response = { MockResponse().setBody(Buffer().write(original)) }
        assertTrue(runCatching { cleanup.remove(shared) }.isFailure)
        assertArrayEquals(original, File(shared.filePath).readBytes())
    }

    @Test fun `account or backend changing during verification keeps original bytes`() = runBlocking {
        for (change in listOf<() -> Unit>(
            { signIn("another") }, { RetrofitClient.setBaseUrl("https://another.invalid/") },
        )) {
            signIn("owner")
            RetrofitClient.setBaseUrl(server.url("/").toString())
            val row = record()
            response = { change(); MockResponse().setBody(Buffer().write(original)) }
            assertTrue(runCatching { cleanup.remove(row) }.isFailure)
            assertArrayEquals(original, File(row.filePath).readBytes())
            assertNull(db.uploadQueueDao().getRecordById(row.id)?.localDeletedAt)
        }
    }

    @Test fun `cancelling a slow original verification closes the request and preserves the file`() = runBlocking {
        val row = record()
        response = { MockResponse().setBody(Buffer().write(original)).throttleBody(1, 1, TimeUnit.SECONDS) }
        val job = launch(Dispatchers.IO) { cleanup.remove(row) }
        assertNotNull(withContext(Dispatchers.IO) { server.takeRequest(5, TimeUnit.SECONDS) })
        assertNotNull(withContext(Dispatchers.IO) { server.takeRequest(5, TimeUnit.SECONDS) })
        withTimeout(5000) { job.cancelAndJoin() }
        assertArrayEquals(original, File(row.filePath).readBytes())
        assertEquals(row, db.uploadQueueDao().getRecordById(row.id))
    }

    @Test fun `retention keeps legacy and unverified originals and only prunes removed-copy records`() = runBlocking {
        val legacy = record().copy(uploadedPhotoId = null, uploadedBackend = null)
        db.uploadQueueDao().updateRecord(legacy)
        val originalRow = record()
        val removed = record()
        cleanup.remove(removed)
        assertEquals(1, db.uploadQueueDao().deleteCompletedBefore(System.currentTimeMillis() + 1, "owner"))
        assertEquals(legacy, db.uploadQueueDao().getRecordById(legacy.id))
        assertEquals(originalRow, db.uploadQueueDao().getRecordById(originalRow.id))
        assertArrayEquals(original, File(legacy.filePath).readBytes())
        assertArrayEquals(original, File(originalRow.filePath).readBytes())
    }

    @Test fun `database failure after verified deletion reports actual removal and cannot requeue the absent file`() = runBlocking {
        val row = record()
        db.openHelper.writableDatabase.execSQL("CREATE TRIGGER fail_cleanup_history BEFORE UPDATE OF localDeletedAt ON upload_queue BEGIN SELECT RAISE(ABORT, 'disk failure'); END")
        assertEquals(TetherStorageCleanup.Removal(original.size.toLong(), false), cleanup.remove(row))
        assertFalse(File(row.filePath).exists())
        assertEquals("COMPLETED", db.uploadQueueDao().getRecordById(row.id)?.uploadStatus)
        assertTrue(cleanup.review().records.isEmpty())
        assertTrue(db.uploadQueueDao().getRecordsWithStatus("QUEUED").isEmpty())
    }
}
