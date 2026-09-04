package com.quickpitik.mobile.worker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.quickpitik.mobile.BuildConfig
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.local.UploadSpool
import com.quickpitik.mobile.data.remote.ApiResponseEnvelope
import com.quickpitik.mobile.data.remote.DirectUploadBeginRequest
import com.quickpitik.mobile.data.remote.DirectUploadCommitRequest
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.UploadedPhotoDto
import okhttp3.Request
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

        // 2b. Prune the COMPLETED ledger. Rows older than any event's upload
        // window can't dedupe a re-import any more (the backend refuses such
        // uploads), so drop them and their thumbnails — this is what keeps the
        // spool at MBs over a season instead of growing forever.
        val cutoff = System.currentTimeMillis() - COMPLETED_TTL_MS
        database.uploadQueueDao().getCompletedPathsBefore(cutoff)
            .forEach { runCatching { File(it).delete() } }
        database.uploadQueueDao().deleteCompletedBefore(cutoff)

        val hasFailures = AtomicBoolean(false)
        // Set on the first connection-level failure. Every record after that
        // requeues at once instead of each waiting out its own 30 s connect
        // timeout (measured: a 2-minute backend stop cost 12 × 30 s inside one
        // run, 2026-09-02). The run then ends with retry() and the 10 s linear
        // backoff, not the timeout, decides when we look again.
        val offline = AtomicBoolean(false)
        // Parallel upload gate. The 2026-05-28 diagnosis showed wall-clock per
        // photo was ~33s on demo Wi-Fi while PTP + persist combined was <0.1s —
        // the worker's serial loop was the bottleneck. 3 concurrent uploads
        // reuse a single OkHttp HTTP/2 connection (no extra TLS) and stay well
        // under Spring Boot tomcat's default 200-thread pool. A 24-photo race
        // batch drops from ~13 min to ~4-5 min. Bumping above 3 starts to
        // contend with the phone's radio and gives diminishing returns.
        // Adaptive since 2026-09-02: start at 3, climb one step after three
        // consecutive fast uploads, drop one step on a slow upload or a dropped
        // connection. The hard cap is the Semaphore; the soft level is what the
        // pacer moves. Ceiling in practice is the phone's uplink — more
        // parallelism past that just makes each upload slower.
        val gate = Semaphore(permits = Pacer.MAX_CONCURRENCY)
        val pacer = Pacer()
        val inFlight = java.util.concurrent.atomic.AtomicInteger(0)

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
                        gate.withPermit {
                            // Soft gate: wait until a slot under the current level frees.
                            while (inFlight.get() >= pacer.level.get()) kotlinx.coroutines.delay(200)
                            inFlight.incrementAndGet()
                            try {
                                uploadOne(record, sessionManager, database, hasFailures, offline, pacer)
                            } finally {
                                inFlight.decrementAndGet()
                            }
                        }
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
        offline: AtomicBoolean,
        pacer: Pacer,
    ) {
        if (offline.get()) {
            // Budget untouched, row stays QUEUED for the next run.
            hasFailures.set(true)
            return
        }
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
            // Direct-to-storage first (bytes go phone → R2, never through the
            // backend); null means "not available here", so fall back to the
            // classic multipart POST the backend has always accepted.
            val responseEnvelope = tryDirectUpload(file, token, record.eventId)
                ?: run {
                    val requestFile = file.asRequestBody("image/jpeg".toMediaTypeOrNull())
                    val body = MultipartBody.Part.createFormData(
                        "file", // Must match backend @RequestPart("file") parameter name
                        file.name,
                        requestFile,
                    )
                    RetrofitClient.apiService.uploadPhoto(
                        token = "Bearer $token",
                        eventId = record.eventId,
                        file = body,
                    )
                }

            val uploadMs = System.currentTimeMillis() - uploadStart
            val ok = responseEnvelope.success && responseEnvelope.data != null
            if (BuildConfig.DEBUG) {
                android.util.Log.i(
                    "QP/UPLOAD-PERF",
                    "upload id=${record.id} file=${file.name} bytes=${file.length()} status=${if (ok) "OK" else "FAIL"} ms=$uploadMs",
                )
            }

            pacer.observe(uploadMs, failed = !ok)
            if (ok) {
                // Status flip MUST stay before the shrink below: a crash before
                // the flip re-uploads the still-intact original (idempotent
                // same-event 200); a shrunk file must never be re-POSTed as the
                // marketplace product (different hash — dedupe wouldn't catch it).
                database.uploadQueueDao().updateStatus(record.id, "COMPLETED", null)
                // The photo is on the server; the local full-res copy is pure
                // waste from here (a 500-shot race left ~1 GB in cacheDir for
                // the life of the install). Instead of deleting it outright, it
                // is shrunk in place to a ~320px thumbnail (~20 KB) so the sync
                // strip can keep showing the frame after upload. The Room row
                // is deliberately KEPT — it is the card-import re-import dedupe
                // ledger (getActiveOrCompletedForEvent). Only files we spooled
                // into our own cacheDir are touched.
                val ours = file.absolutePath.startsWith(applicationContext.cacheDir.absolutePath) ||
                    file.absolutePath.startsWith(UploadSpool.dir(applicationContext).absolutePath)
                if (ours) {
                    if (!runCatching { shrinkToThumbnail(file) }.getOrDefault(false)) {
                        runCatching { file.delete() }
                    }
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
            if (BuildConfig.DEBUG) {
                android.util.Log.i(
                    "QP/UPLOAD-PERF",
                    "upload id=${record.id} file=${file.name} bytes=${file.length()} status=EXC ms=$uploadMs",
                )
            }
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
            // No HTTP response at all (connection refused, reset, timed out):
            // the backend never saw the photo, so this says nothing about the
            // photo. It must not consume the record's retry budget — otherwise
            // ~15 minutes of "Wi-Fi up, backend unreachable" settles every
            // queued frame FAILED for good (5 runs on exponential backoff).
            val transport = e is java.io.IOException
            if (transport) offline.set(true)
            pacer.observe(uploadMs, failed = true)
            settleFailure(record, terminal, errorMsg, database, hasFailures, transport)
        }
    }

    /**
     * Direct-to-storage path (backend 2026-09-02): hash → begin → PUT the file
     * to the presigned URL → commit. Returns the commit envelope, an envelope
     * wrapping the existing photo when the bytes were already in this event,
     * or NULL when this path isn't usable — an older backend (404), a
     * local-disk deployment ("multipart"), or a storage PUT that didn't
     * succeed — so the caller runs the multipart upload instead.
     *
     * Backend business answers (409 duplicate, 422 window closed, 403…) are
     * rethrown untouched: they mean the same thing on either path and the
     * terminal-code logic in the caller must see them.
     */
    private suspend fun tryDirectUpload(
        file: File,
        token: String,
        eventId: String,
    ): ApiResponseEnvelope<UploadedPhotoDto>? {
        val contentType = "image/jpeg"
        val hash = sha256Hex(file)
        val begin = try {
            RetrofitClient.apiService.beginDirectUpload(
                "Bearer $token",
                eventId,
                DirectUploadBeginRequest(contentHash = hash, contentType = contentType, sizeBytes = file.length()),
            )
        } catch (e: retrofit2.HttpException) {
            if (e.code() == 404) return null // backend predates the endpoint
            throw e
        }
        val plan = begin.data ?: return null
        return when (plan.mode) {
            "existing" -> ApiResponseEnvelope(success = true, data = plan.existing)
            "direct" -> {
                val url = plan.uploadUrl ?: return null
                val photoId = plan.photoId ?: return null
                val key = plan.key ?: return null
                val put = Request.Builder()
                    .url(url)
                    .put(file.asRequestBody(contentType.toMediaTypeOrNull()))
                    .build()
                val stored = try {
                    RetrofitClient.rawClient.newCall(put).execute().use { it.isSuccessful }
                } catch (e: java.io.IOException) {
                    false
                }
                if (!stored) return null
                RetrofitClient.apiService.commitDirectUpload(
                    "Bearer $token",
                    eventId,
                    DirectUploadCommitRequest(photoId = photoId, key = key, contentHash = hash, contentType = contentType),
                )
            }
            else -> null
        }
    }

    private fun sha256Hex(file: File): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(64 * 1024)
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                digest.update(buf, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
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
        transport: Boolean = false,
    ) {
        if (terminal) {
            database.uploadQueueDao().updateStatus(record.id, "FAILED", errorMsg)
            return
        }
        if (transport) {
            // Requeue with the budget intact; WorkManager's backoff paces us.
            database.uploadQueueDao().updateStatus(record.id, "QUEUED", errorMsg)
            hasFailures.set(true)
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

    /**
     * Replaces an already-uploaded spool JPEG with a ~320px thumbnail at the
     * SAME path, so UploadRecord.filePath stays renderable in the sync strip
     * forever at ~20 KB — no schema change, no capture-path cost. Returns false
     * when the bytes can't be decoded; the caller then falls back to the old
     * delete. In-place rewrite, no temp+rename: renameTo can't replace an
     * existing file on the Windows JVM the Robolectric suite runs on, and a
     * mid-write crash only costs a placeholder tile — the upload itself already
     * COMPLETED before this runs. Runs inside the Semaphore(3) permit, so at
     * most 3 bounded decodes are ever in flight.
     */
    private fun shrinkToThumbnail(file: File): Boolean {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return false
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= THUMBNAIL_LONG_EDGE_PX) {
            sample *= 2
        }
        var bitmap = BitmapFactory.decodeFile(
            file.absolutePath,
            BitmapFactory.Options().apply { inSampleSize = sample },
        ) ?: return false
        // Re-encoding drops the EXIF orientation tag, so bake the rotation into
        // the pixels — otherwise portrait frames render sideways after the swap.
        val degrees = runCatching {
            when (
                ExifInterface(file.absolutePath)
                    .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            ) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
        }.getOrDefault(0f)
        if (degrees != 0f) {
            val rotated = Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height,
                Matrix().apply { postRotate(degrees) }, true,
            )
            bitmap.recycle()
            bitmap = rotated
        }
        val encoded = file.outputStream().use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, THUMBNAIL_JPEG_QUALITY, out)
        }
        bitmap.recycle()
        return encoded
    }

    /**
     * Additive-increase / multiplicative-free concurrency pacer. Deliberately
     * dumb: three fast uploads in a row earn one more slot, any slow or failed
     * upload gives one back. Bounded by [MIN_CONCURRENCY]..[MAX_CONCURRENCY].
     */
    class Pacer {
        val level = java.util.concurrent.atomic.AtomicInteger(INITIAL_CONCURRENCY)
        private val fastStreak = java.util.concurrent.atomic.AtomicInteger(0)

        fun observe(uploadMs: Long, failed: Boolean) {
            if (failed || uploadMs > SLOW_MS) {
                fastStreak.set(0)
                level.updateAndGet { maxOf(MIN_CONCURRENCY, it - 1) }
            } else if (uploadMs < FAST_MS) {
                if (fastStreak.incrementAndGet() >= FAST_STREAK) {
                    fastStreak.set(0)
                    level.updateAndGet { minOf(MAX_CONCURRENCY, it + 1) }
                }
            } else {
                fastStreak.set(0)
            }
        }

        companion object {
            const val MIN_CONCURRENCY = 2
            const val INITIAL_CONCURRENCY = 3
            const val MAX_CONCURRENCY = 6
            const val FAST_MS = 4_000L
            const val SLOW_MS = 15_000L
            const val FAST_STREAK = 3
        }
    }

    private companion object {
        // COMPLETED rows older than this are pruned at the start of each run.
        // Seven days comfortably exceeds the backend's 4-day upload window.
        const val COMPLETED_TTL_MS = 7L * 24 * 60 * 60 * 1000

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

        // Post-upload in-place thumbnail: long edge + JPEG quality. ~20 KB per
        // synced frame keeps the sync strip renderable without hoarding
        // multi-MB originals in cacheDir.
        // Was 320px/q80 (~20 KB). The strip tile is 64dp, but the tile now opens
        // the lightbox, and a 320px frame is unreadable there. 1024px/q75 is
        // ~100 KB — a 1,000-frame race keeps ~100 MB on the phone.
        // ponytail: fixed size; prune-by-age (7 days) bounds it, add a
        // "keep thumbnails" setting if a photographer ever asks.
        const val THUMBNAIL_LONG_EDGE_PX = 1024
        const val THUMBNAIL_JPEG_QUALITY = 75
    }
}
