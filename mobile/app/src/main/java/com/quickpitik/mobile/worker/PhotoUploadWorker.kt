package com.quickpitik.mobile.worker

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

class PhotoUploadWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        val database = AppDatabase.getDatabase(applicationContext)
        val sessionManager = SessionManager.getInstance(applicationContext)

        // 1. Nobody logged in ⇒ nothing this job can do. The value is only a
        // gate: each upload re-reads the token, because TokenAuthenticator
        // rotates it mid-drain and a captured copy would make every subsequent
        // request spend a wasted 401 round-trip before the authenticator's
        // already-refreshed branch healed it — once per photo, on a queue that
        // can be hundreds long.
        sessionManager.getAccessToken() ?: return Result.failure()

        // 2. Requeue rows a killed process left stuck in UPLOADING. Safe: the
        // backend answers a same-event re-upload of bytes it already has with
        // an idempotent 200; a cross-event duplicate is a terminal 409 handled
        // below. Only one worker runs at a time (unique work, KEEP), so these
        // rows can't belong to a live sibling.
        database.uploadQueueDao().getRecordsWithStatus("UPLOADING")
            .forEach { database.uploadQueueDao().updateStatus(it.id, "QUEUED", null) }

        val hasFailures = AtomicBoolean(false)
        // Parallel upload gate. The 2026-05-28 diagnosis showed wall-clock per
        // photo was ~33s on demo Wi-Fi while PTP + persist combined was <0.1s —
        // the worker's serial loop was the bottleneck. 3 concurrent uploads
        // reuse a single OkHttp HTTP/2 connection (no extra TLS) and stay well
        // under Spring Boot tomcat's default 200-thread pool. A 24-photo race
        // batch drops from ~13 min to ~4-5 min. Bumping above 3 starts to
        // contend with the phone's radio and gives diminishing returns.
        val gate = Semaphore(permits = 3)

        // 3. Drain until the queue is empty. Re-querying after each batch picks
        // up rows inserted mid-run (live capture enqueues while uploads are in
        // flight) — this is what lets runSyncEngine() use ExistingWorkPolicy.KEEP
        // and still miss nothing. The `attempted` filter gives each record at
        // most ONE attempt per run, so rows that settleFailure() requeued wait
        // for the next run (scheduled by Result.retry() with backoff) and the
        // loop always terminates.
        val attempted = HashSet<Long>()
        while (true) {
            val batch = database.uploadQueueDao()
                .getRecordsWithStatus("QUEUED")
                .filter { it.id !in attempted }
            if (batch.isEmpty()) break
            batch.forEach { attempted.add(it.id) }
            coroutineScope {
                for (record in batch) {
                    launch {
                        gate.withPermit { uploadOne(record, sessionManager, database, hasFailures) }
                    }
                }
            }
        }

        return if (hasFailures.get()) Result.retry() else Result.success()
    }

    private suspend fun uploadOne(
        record: UploadRecord,
        sessionManager: SessionManager,
        database: AppDatabase,
        hasFailures: AtomicBoolean,
    ) {
        // Mark record as UPLOADING to prevent duplicate workers picking it up
        database.uploadQueueDao().updateStatus(record.id, "UPLOADING", null)

        val file = File(record.filePath)
        if (!file.exists()) {
            // Terminal: a deleted cache file never reappears, so retrying
            // can only fail the same way.
            database.uploadQueueDao().updateStatus(record.id, "FAILED", "Local file not found.")
            return
        }

        // Read per record, not per run — see the note in doWork().
        // Null means the session was cleared mid-drain (TokenAuthenticator gave
        // up on the refresh). Not terminal for the photo: requeue so the row
        // survives to a run made after the photographer signs back in.
        val token = sessionManager.getAccessToken()
        if (token == null) {
            settleFailure(record, false, "Signed out.", database, hasFailures)
            return
        }

        val uploadStart = System.currentTimeMillis()
        try {
            val requestFile = file.asRequestBody("image/jpeg".toMediaTypeOrNull())
            val body = MultipartBody.Part.createFormData(
                "file", // Must match backend @RequestPart("file") parameter name
                file.name,
                requestFile,
            )

            val responseEnvelope = RetrofitClient.apiService.uploadPhoto(
                token = "Bearer $token",
                eventId = record.eventId,
                file = body,
            )

            val uploadMs = System.currentTimeMillis() - uploadStart
            val ok = responseEnvelope.success && responseEnvelope.data != null
            android.util.Log.i(
                "QP/UPLOAD-PERF",
                "upload id=${record.id} file=${file.name} bytes=${file.length()} status=${if (ok) "OK" else "FAIL"} ms=$uploadMs",
            )

            if (ok) {
                database.uploadQueueDao().updateStatus(record.id, "COMPLETED", null)
                // The photo is on the server; the local full-res copy is pure
                // waste from here (a 500-shot race left ~1 GB in cacheDir for
                // the life of the install). The Room row is deliberately KEPT —
                // it is the card-import re-import dedupe ledger
                // (getActiveOrCompletedForEvent). Only files we spooled into
                // our own cacheDir are touched.
                if (file.absolutePath.startsWith(applicationContext.cacheDir.absolutePath)) {
                    runCatching { file.delete() }
                }
            } else {
                val errorMsg = responseEnvelope.error ?: "Upload rejected by server."
                // A duplicate rejection (backend ErrorCodes.PHOTO_DUPLICATE_*) is
                // TERMINAL — the photo is already in an event, so retrying can
                // never succeed. Everything else is retryable.
                val terminal = responseEnvelope.errors
                    ?.any { it.code in TERMINAL_UPLOAD_ERROR_CODES } == true
                settleFailure(record, terminal, errorMsg, database, hasFailures)
            }
        } catch (e: Exception) {
            val uploadMs = System.currentTimeMillis() - uploadStart
            android.util.Log.i(
                "QP/UPLOAD-PERF",
                "upload id=${record.id} file=${file.name} bytes=${file.length()} status=EXC ms=$uploadMs",
            )
            // The backend rejects a duplicate with HTTP 409, and Retrofit throws
            // HttpException for any non-2xx status BEFORE the success-path body
            // (and its TERMINAL_UPLOAD_ERROR_CODES guard above) is ever reached —
            // so the real duplicate check has to run HERE, off the thrown
            // exception's error body. A duplicate is terminal: retrying re-POSTs
            // the same bytes and can never succeed. Genuine network failures
            // (no parsed code) requeue via settleFailure.
            val apiError = RetrofitClient.parseHttpError(e)
            val terminal = apiError != null && apiError.code in TERMINAL_UPLOAD_ERROR_CODES
            val errorMsg = apiError?.message ?: e.localizedMessage ?: "Network connection timeout."
            settleFailure(record, terminal, errorMsg, database, hasFailures)
        }
    }

    /**
     * Single decision point for a failed attempt. Terminal failures end the
     * record. Non-terminal ones go BACK to QUEUED — Result.retry() re-runs
     * doWork(), which only picks up QUEUED rows, so a row left FAILED would
     * never actually be retried. retryCount caps the cycle at MAX_RETRIES
     * attempts, after which the row fails for good and stops driving retries.
     * The errorMessage is kept on a requeued row for diagnostics; the sync
     * card's lastError only reads FAILED rows, so the UI stays quiet until the
     * record truly gives up.
     */
    private suspend fun settleFailure(
        record: UploadRecord,
        terminal: Boolean,
        errorMsg: String,
        database: AppDatabase,
        hasFailures: AtomicBoolean,
    ) {
        if (terminal) {
            database.uploadQueueDao().updateStatus(record.id, "FAILED", errorMsg)
            return
        }
        database.uploadQueueDao().incrementRetryCount(record.id)
        if (record.retryCount + 1 < MAX_RETRIES) {
            database.uploadQueueDao().updateStatus(record.id, "QUEUED", errorMsg)
            hasFailures.set(true)
        } else {
            database.uploadQueueDao().updateStatus(record.id, "FAILED", errorMsg)
        }
    }

    private companion object {
        // Rejections that can NEVER succeed on retry, so they END the record
        // instead of driving the exponential-backoff loop. Beyond the two
        // dedup codes, PhotoUploadService also throws these permanently-fatal
        // codes — each used to burn all 5 retry cycles per photo before
        // settling FAILED (a suspended photographer's 200-shot card = 1000
        // doomed uploads):
        //   EVENT_NOT_UPLOADABLE     — event's upload window closed
        //   ACCOUNT_SUSPENDED        — suspension isn't lifted by retrying
        //   PHOTOGRAPHER_NOT_VERIFIED — verification is a manual admin step
        //   WATERMARK_MISSING        — settings problem, not transient
        //   UNSUPPORTED_MEDIA_TYPE   — the bytes never change
        //   USER_NOT_FOUND / EVENT_NOT_FOUND — the row references a ghost
        val TERMINAL_UPLOAD_ERROR_CODES = setOf(
            "PHOTO_DUPLICATE_DIFFERENT_EVENT",
            "PHOTO_DUPLICATE_SAME_EVENT",
            "EVENT_NOT_UPLOADABLE",
            "ACCOUNT_SUSPENDED",
            "PHOTOGRAPHER_NOT_VERIFIED",
            "WATERMARK_MISSING",
            "UNSUPPORTED_MEDIA_TYPE",
            "USER_NOT_FOUND",
            "EVENT_NOT_FOUND",
        )

        // Attempts per record across runs (tracked in UploadRecord.retryCount)
        // before a non-terminal failure is declared permanent.
        const val MAX_RETRIES = 5
    }
}
