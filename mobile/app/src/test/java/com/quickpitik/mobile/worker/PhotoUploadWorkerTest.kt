package com.quickpitik.mobile.worker

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.work.ListenableWorker
import androidx.work.testing.TestListenableWorkerBuilder
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.UploadRecord
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Covers the branches of [PhotoUploadWorker.doWork] that settle a record
 * WITHOUT reaching the network — the signed-out gate, the missing-file terminal
 * failure, and the stuck-row requeue a killed process leaves behind.
 *
 * The upload path itself (success, terminal 409 duplicate, the retry ladder up
 * to MAX_RETRIES) is deliberately not covered here: the worker reaches the
 * global `RetrofitClient.apiService`, built from a `const val BASE_URL`, so
 * there is no seam to point at a MockWebServer. Adding one is a production
 * refactor and is tracked separately rather than smuggled into a test change.
 */
@RunWith(RobolectricTestRunner::class)
class PhotoUploadWorkerTest {

    private lateinit var context: Context
    private lateinit var db: AppDatabase
    private lateinit var session: SessionManager

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        // The worker resolves AppDatabase + SessionManager itself, so the test
        // has to seed those exact singletons rather than inject its own.
        //
        // AppDatabase caches a file-backed instance in a static, but Robolectric
        // closes its SQLite connections between test METHODS (its sandbox is
        // per-class, not per-method). Method 2 onward would then reuse a dead
        // connection — "Illegal connection pointer". Clearing the static and
        // deleting the file gives each method a genuinely fresh database, which
        // also removes any need to clear rows.
        clearDatabaseSingleton()
        context.deleteDatabase(DB_NAME)
        db = AppDatabase.getDatabase(context)
        session = SessionManager.getInstance(context)
        session.clearSession()
    }

    @After
    fun tearDown() {
        db.close()
        clearDatabaseSingleton()
    }

    /**
     * Resets `AppDatabase.INSTANCE`. Reflection because the field is private and
     * the alternative — adding a test-only reset hook — would be a production
     * change made purely to serve a test. Kotlin lowers a companion-object
     * backing field to a static on the enclosing class, hence the lookup here.
     */
    private fun clearDatabaseSingleton() {
        AppDatabase::class.java.getDeclaredField("INSTANCE").apply {
            isAccessible = true
            set(null, null)
        }
    }

    private suspend fun enqueue(
        filePath: String = "/definitely/not/a/real/file.jpg",
        status: String = "QUEUED",
        retryCount: Int = 0,
    ): Long = db.uploadQueueDao().insertRecord(
        UploadRecord(
            filePath = filePath,
            eventId = EVENT,
            photographerId = "shooter@example.com",
            captureTimestamp = 1_700_000_000_000L,
            uploadStatus = status,
            retryCount = retryCount,
        )
    )

    private suspend fun runWorker(): ListenableWorker.Result =
        TestListenableWorkerBuilder<PhotoUploadWorker>(context).build().doWork()

    private fun signIn() =
        session.saveSession(
            token = "test-access-token",
            role = "PHOTOGRAPHER",
            name = "Test Shooter",
            email = "shooter@example.com",
        )

    @Test
    fun `signed out fails the run and leaves the queue untouched`() = runTest {
        val id = enqueue()

        val result = runWorker()

        assertTrue(result is ListenableWorker.Result.Failure)
        // Untouched matters: the frames must survive to a run made after the
        // photographer signs back in, not be marked failed on their behalf.
        assertEquals("QUEUED", db.uploadQueueDao().getRecordById(id)?.uploadStatus)
    }

    @Test
    fun `a missing local file is terminal, not a retry`() = runTest {
        signIn()
        val id = enqueue(filePath = "/cache/deleted-by-the-os.jpg")

        val result = runWorker()

        val stored = requireNotNull(db.uploadQueueDao().getRecordById(id))
        assertEquals("FAILED", stored.uploadStatus)
        assertEquals("Local file not found.", stored.errorMessage)
        // A deleted cache file never reappears, so the run must NOT ask
        // WorkManager to come back — that would spin the backoff loop forever.
        assertTrue(result is ListenableWorker.Result.Success)
        assertEquals(0, stored.retryCount)
    }

    /**
     * A process killed mid-upload leaves rows stranded in UPLOADING, which the
     * QUEUED-only drain would never look at again. The run requeues them first.
     * The bogus path makes the row settle as FAILED — an unrequeued row would
     * still read UPLOADING, so this asserts the requeue actually happened.
     */
    @Test
    fun `rows stranded in UPLOADING are requeued and then processed`() = runTest {
        signIn()
        val id = enqueue(status = "UPLOADING")

        runWorker()

        assertEquals("FAILED", db.uploadQueueDao().getRecordById(id)?.uploadStatus)
    }

    @Test
    fun `an empty queue succeeds without work`() = runTest {
        signIn()

        val result = runWorker()

        assertTrue(result is ListenableWorker.Result.Success)
        assertTrue(db.uploadQueueDao().getRecordsWithStatus("QUEUED").isEmpty())
    }

    private companion object {
        const val EVENT = "11111111-1111-1111-1111-111111111111"

        // Mirrors the name AppDatabase passes to Room.databaseBuilder.
        const val DB_NAME = "quickpitik_db"
    }
}
