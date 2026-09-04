package com.quickpitik.mobile.worker

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.work.ListenableWorker
import androidx.work.testing.TestListenableWorkerBuilder
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.test.runTest
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadows.ShadowBitmapFactory
import java.io.File

/**
 * Covers [PhotoUploadWorker.doWork] end to end.
 *
 * The pre-network branches (signed-out gate, missing file, stuck-row requeue)
 * need no server. The upload path — success, the terminal 409 duplicate, and
 * the retry ladder up to MAX_RETRIES — runs against a [MockWebServer] that
 * [RetrofitClient.setBaseUrl] points the app at. That seam landed 2026-08-16;
 * before it, `BASE_URL` was a `const val` and these five cases were untestable.
 */
@RunWith(RobolectricTestRunner::class)
class PhotoUploadWorkerTest {

    private lateinit var context: Context
    private lateinit var db: AppDatabase
    private lateinit var session: SessionManager
    private var server: MockWebServer? = null

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
        server?.shutdown()
        server = null
        // RetrofitClient is an object, so a URL left pointing at a shut-down
        // MockWebServer would leak into the next method in this class.
        // Robolectric's sandbox is per-class, not per-method.
        RetrofitClient.resetBaseUrl()
    }

    /**
     * Starts a server, points the app at it, and returns it. Each enqueued
     * response is one upload attempt, in order.
     */
    private fun startServer(vararg responses: MockResponse): MockWebServer =
        MockWebServer().apply {
            // The worker probes the direct-upload endpoint first; answer it
            // like a backend that predates it (404) so these tests keep
            // exercising the multipart path with the enqueued responses.
            val queue = ArrayDeque(responses.toList())
            dispatcher = object : okhttp3.mockwebserver.Dispatcher() {
                override fun dispatch(request: okhttp3.mockwebserver.RecordedRequest): MockResponse =
                    if (request.path?.endsWith("/photos/direct") == true) MockResponse().setResponseCode(404)
                    else queue.removeFirstOrNull() ?: MockResponse().setResponseCode(500)
            }
            start()
            server = this
            RetrofitClient.setBaseUrl(url("/").toString())
        }

    /**
     * A backend + storage pair for the direct path: begin hands out a PUT URL
     * on this same server, the PUT is accepted, commit returns the photo.
     */
    private fun startDirectServer(): MockWebServer =
        MockWebServer().apply {
            dispatcher = object : okhttp3.mockwebserver.Dispatcher() {
                override fun dispatch(request: okhttp3.mockwebserver.RecordedRequest): MockResponse = when {
                    request.path?.endsWith("/photos/direct") == true -> MockResponse()
                        .setResponseCode(200)
                        .setHeader("Content-Type", "application/json")
                        .setBody(
                            """{"success":true,"data":{"mode":"direct","photoId":"p-1",
                               "key":"events/$EVENT/photos/p-1/original.jpg",
                               "uploadUrl":"${url("/r2/events/$EVENT/photos/p-1/original.jpg")}"}}""",
                        )
                    request.method == "PUT" && request.path?.startsWith("/r2/") == true ->
                        MockResponse().setResponseCode(200)
                    request.path?.endsWith("/photos/direct/commit") == true -> ok()
                    else -> MockResponse().setResponseCode(500)
                }
            }
            start()
            server = this
            RetrofitClient.setBaseUrl(url("/").toString())
        }

    /** A real file on disk — the worker bails before the network without one. */
    private fun realJpeg(): File =
        File.createTempFile("upload", ".jpg").apply {
            writeBytes(ByteArray(64) { 0xFF.toByte() })
            deleteOnExit()
        }

    private fun ok() = MockResponse()
        .setResponseCode(200)
        .setHeader("Content-Type", "application/json")
        .setBody("""{"success":true,"data":{"id":"photo-1"}}""")

    /** The backend's dedup rejection: HTTP 409 carrying a terminal error code. */
    private fun duplicate(code: String) = MockResponse()
        .setResponseCode(409)
        .setHeader("Content-Type", "application/json")
        .setBody("""{"success":false,"errors":[{"code":"$code","message":"Already uploaded."}]}""")

    private fun serverError() = MockResponse()
        .setResponseCode(500)
        .setHeader("Content-Type", "application/json")
        .setBody("""{"success":false,"errors":[{"code":"INTERNAL_ERROR","message":"Boom."}]}""")

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

    // ---- upload path (MockWebServer) ----

    @Test
    fun `a 200 completes the record`() = runTest {
        signIn()
        val http = startServer(ok())
        val id = enqueue(filePath = realJpeg().absolutePath)

        val result = runWorker()

        val stored = requireNotNull(db.uploadQueueDao().getRecordById(id))
        assertEquals("COMPLETED", stored.uploadStatus)
        assertTrue(result is ListenableWorker.Result.Success)
        // The path carries the event, so a mis-templated URL would silently
        // upload every frame to the wrong gallery.
        // Skip the direct-upload probe; the multipart POST is the one under test.
        val request = http.takeRequest().let { first ->
            if (first.path?.endsWith("/photos/direct") == true) http.takeRequest() else first
        }
        assertEquals("POST", request.method)
        assertTrue(request.path!!.endsWith("/api/v1/me/photographer/events/$EVENT/photos"))
        assertEquals("Bearer test-access-token", request.getHeader("Authorization"))
    }

    /**
     * The 2026-06-03 regression: Retrofit throws HttpException on the 409 before
     * the success-body guard runs, so the terminal check has to happen in the
     * catch. Getting this wrong re-POSTs the same bytes forever.
     */
    @Test
    fun `direct upload PUTs the bytes to storage and commits, no multipart POST`() = runTest {
        signIn()
        val http = startDirectServer()
        val id = enqueue(filePath = realJpeg().absolutePath)

        val result = runWorker()

        assertEquals("COMPLETED", db.uploadQueueDao().getRecordById(id)?.uploadStatus)
        assertTrue(result is ListenableWorker.Result.Success)
        val paths = (1..http.requestCount).map { http.takeRequest().let { r -> "${r.method} ${r.path}" } }
        assertTrue(paths.any { it.startsWith("PUT /r2/") })
        assertTrue(paths.any { it.endsWith("/photos/direct/commit") })
        assertTrue(paths.none { it == "POST /api/v1/me/photographer/events/$EVENT/photos" })
    }

    @Test
    fun `a 409 same-event duplicate is terminal and does not drive a retry`() = runTest {
        signIn()
        val http = startServer(duplicate("PHOTO_DUPLICATE_SAME_EVENT"))
        val id = enqueue(filePath = realJpeg().absolutePath)

        val result = runWorker()

        val stored = requireNotNull(db.uploadQueueDao().getRecordById(id))
        assertEquals("FAILED", stored.uploadStatus)
        assertEquals(0, stored.retryCount)
        assertTrue(result is ListenableWorker.Result.Success)
        // 1 direct-upload probe (answered 404 by startServer) + exactly 1 multipart attempt.
        assertEquals(2, http.requestCount)
    }

    @Test
    fun `a 409 different-event duplicate is terminal too`() = runTest {
        signIn()
        startServer(duplicate("PHOTO_DUPLICATE_DIFFERENT_EVENT"))
        val id = enqueue(filePath = realJpeg().absolutePath)

        val result = runWorker()

        val stored = requireNotNull(db.uploadQueueDao().getRecordById(id))
        assertEquals("FAILED", stored.uploadStatus)
        assertTrue(result is ListenableWorker.Result.Success)
    }

    /**
     * The non-duplicate permanently-fatal codes (suspension, closed upload
     * window, missing watermark…) joined TERMINAL_UPLOAD_ERROR_CODES on
     * 2026-08-26 — before that, each burned all 5 backoff cycles per photo.
     */
    @Test
    fun `a 422 event-not-uploadable is terminal and does not drive a retry`() = runTest {
        signIn()
        val http = startServer(
            MockResponse()
                .setResponseCode(422)
                .setHeader("Content-Type", "application/json")
                .setBody("""{"success":false,"errors":[{"code":"EVENT_NOT_UPLOADABLE","message":"Upload window closed."}]}""")
        )
        val id = enqueue(filePath = realJpeg().absolutePath)

        val result = runWorker()

        val stored = requireNotNull(db.uploadQueueDao().getRecordById(id))
        assertEquals("FAILED", stored.uploadStatus)
        assertEquals(0, stored.retryCount)
        assertTrue(result is ListenableWorker.Result.Success)
        // 1 direct-upload probe (answered 404 by startServer) + exactly 1 multipart attempt.
        assertEquals(2, http.requestCount)
    }

    /**
     * A completed upload's spool copy in OUR cacheDir is shrunk in place to a
     * ~320px thumbnail (the server has the full bytes; the strip keeps showing
     * the frame); the Room row survives as the re-import dedupe ledger. Under
     * Robolectric's legacy graphics shadows, BitmapFactory "decodes" the garbage
     * test bytes into a fake bitmap and compress writes real JPEG bytes, so the
     * rewrite is observable as changed content at the same path.
     */
    @Test
    fun `a completed upload shrinks its cacheDir spool file to a thumbnail`() = runTest {
        signIn()
        startServer(ok())
        val original = ByteArray(64) { 0xFF.toByte() }
        val spooled = File(context.cacheDir, "gallery_upload_test.jpg").apply {
            writeBytes(original)
        }
        val id = enqueue(filePath = spooled.absolutePath)

        runWorker()

        assertEquals("COMPLETED", db.uploadQueueDao().getRecordById(id)?.uploadStatus)
        assertTrue(spooled.exists())
        assertTrue(spooled.length() > 0)
        assertTrue(!spooled.readBytes().contentEquals(original))
    }

    /**
     * When the spool bytes can't be decoded, the shrink falls back to the
     * pre-2026-08-28 behavior: delete the file outright so undecodable
     * originals can't hoard cacheDir. setAllowInvalidImageData(false) makes
     * the shadow decoder return null for the garbage bytes (auto-reset by
     * Robolectric between tests).
     */
    @Test
    fun `an undecodable completed upload falls back to deleting its spool file`() = runTest {
        ShadowBitmapFactory.setAllowInvalidImageData(false)
        signIn()
        startServer(ok())
        val spooled = File(context.cacheDir, "gallery_upload_test.jpg").apply {
            writeBytes(ByteArray(64) { 0xFF.toByte() })
        }
        val id = enqueue(filePath = spooled.absolutePath)

        runWorker()

        assertEquals("COMPLETED", db.uploadQueueDao().getRecordById(id)?.uploadStatus)
        assertTrue(!spooled.exists())
    }

    /**
     * A connection that never yields an HTTP response says nothing about the
     * photo, so it must requeue WITHOUT spending the retry budget — a backend
     * outage longer than five backoff rounds used to fail every queued frame.
     */
    @Test
    fun `a dropped connection requeues without consuming the retry budget`() = runTest {
        signIn()
        startServer(MockResponse().setSocketPolicy(okhttp3.mockwebserver.SocketPolicy.DISCONNECT_AT_START))
        val id = enqueue(filePath = realJpeg().absolutePath, retryCount = 4)

        val result = runWorker()

        val stored = requireNotNull(db.uploadQueueDao().getRecordById(id))
        assertEquals("QUEUED", stored.uploadStatus)
        assertEquals(4, stored.retryCount)
        assertTrue(result is ListenableWorker.Result.Retry)
    }

    @Test
    fun `a 500 requeues the record and asks WorkManager to come back`() = runTest {
        signIn()
        val http = startServer(serverError())
        val id = enqueue(filePath = realJpeg().absolutePath)

        val result = runWorker()

        val stored = requireNotNull(db.uploadQueueDao().getRecordById(id))
        // Back to QUEUED, not FAILED: the drain only picks up QUEUED, so a row
        // left FAILED would never actually be retried.
        assertEquals("QUEUED", stored.uploadStatus)
        assertEquals(1, stored.retryCount)
        assertTrue(result is ListenableWorker.Result.Retry)
        // Exactly one attempt: the `attempted` filter must stop the requeued row
        // from being picked straight back up inside the same run.
        // 1 direct-upload probe (answered 404 by startServer) + exactly 1 multipart attempt.
        assertEquals(2, http.requestCount)
    }

    /**
     * Last rung of the ladder. The record has to stop driving retries, or one
     * permanently-broken frame keeps the whole queue in backoff forever.
     */
    @Test
    fun `the retry ladder ends at MAX_RETRIES and stops asking for more runs`() = runTest {
        signIn()
        startServer(serverError())
        val id = enqueue(filePath = realJpeg().absolutePath, retryCount = MAX_RETRIES - 1)

        val result = runWorker()

        val stored = requireNotNull(db.uploadQueueDao().getRecordById(id))
        assertEquals("FAILED", stored.uploadStatus)
        assertEquals(MAX_RETRIES, stored.retryCount)
        assertTrue(result is ListenableWorker.Result.Success)
    }

    private companion object {
        const val EVENT = "11111111-1111-1111-1111-111111111111"

        // Mirrors PhotoUploadWorker's private MAX_RETRIES.
        const val MAX_RETRIES = 5

        // Mirrors the name AppDatabase passes to Room.databaseBuilder.
        const val DB_NAME = "quickpitik_db"
    }
}
