package com.quickpitik.mobile.worker

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.UploadQueueDao
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.data.remote.*
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.asRequestBody
import retrofit2.HttpException
import java.io.File
import java.io.IOException
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.resumeWithException

class PhotoUploadWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val started = System.nanoTime()
        var outcome = "interrupted"
        try {
            return drainQueue().also { outcome = it.javaClass.simpleName }
        } finally {
            Log.i("QP/UPLOAD", "run attempt=$runAttemptCount result=$outcome ms=${(System.nanoTime() - started) / 1_000_000}")
        }
    }

    private suspend fun drainQueue(): Result {
        val database = AppDatabase.getDatabase(applicationContext)
        val session = SessionManager.getInstance(applicationContext)
        session.getAccessToken() ?: return Result.success()
        val owner = session.getUserEmail() ?: return Result.success()
        val dao = database.uploadQueueDao()
        dao.recoverOwned(owner)
        if (dao.retryNotBefore(owner) > System.currentTimeMillis()) return Result.retry()
        val cutoff = System.currentTimeMillis() - COMPLETED_TTL_MS
        dao.deleteCompletedBefore(cutoff, owner)

        val failures = AtomicBoolean(false)
        val offline = AtomicBoolean(false)
        val unauthorized = AtomicBoolean(false)
        val completed = AtomicInteger(0)
        val pacer = Pacer()
        val started = System.nanoTime()
        var afterId = 0L
        while (true) {
            currentCoroutineContext().ensureActive()
            if (session.getUserEmail() != owner) return Result.success()
            val batch = dao.getOwnedRecords(owner, "QUEUED", afterId)
            if (batch.isEmpty()) break
            afterId = batch.last().id
            // Bounded groups need no polling of an in-flight counter. Camera transfer is independent.
            for (group in batch.chunked(pacer.level.get())) {
                if ((System.nanoTime() - started) / 1_000_000 >= runBudgetMs)
                    return endRun(owner, dao, completed, Result.retry())
                coroutineScope {
                    group.forEach { record -> launch(Dispatchers.IO) {
                        uploadOne(record, session, database, failures, offline, unauthorized, pacer, completed)
                    } }
                }
                if (unauthorized.get()) return Result.success()
                if (offline.get()) return endRun(owner, dao, completed, Result.retry())
            }
        }
        return endRun(owner, dao, completed, if (failures.get()) Result.retry() else Result.success())
    }

    private suspend fun endRun(owner: String, dao: UploadQueueDao, completed: AtomicInteger, otherwise: Result): Result {
        if (completed.get() > 0 && dao.getOwnedRecords(owner, "QUEUED", 0).isNotEmpty()) {
            PhotoUploadScheduler.enqueue(applicationContext).join()
            return Result.success()
        }
        return otherwise
    }

    private suspend fun uploadOne(
        record: UploadRecord, session: SessionManager, database: AppDatabase,
        failures: AtomicBoolean, offline: AtomicBoolean, unauthorized: AtomicBoolean, pacer: Pacer, completed: AtomicInteger,
    ) {
        currentCoroutineContext().ensureActive()
        if (session.getUserEmail() != record.photographerId) return
        if (unauthorized.get()) return
        if (offline.get() || record.retryAfter > System.currentTimeMillis()) { failures.set(true); return }
        val dao = database.uploadQueueDao()
        dao.updateStatus(record.id, "UPLOADING", null)
        val file = File(record.filePath)
        if (!file.exists()) { dao.updateStatus(record.id, "FAILED", "Local file not found."); return }
        val started = System.nanoTime()
        val backend = RetrofitClient.BASE_URL
        try {
            requireOwner(record.photographerId)
            val response = tryDirectUpload(file, record, database) ?: run {
                val body = MultipartBody.Part.createFormData("file", record.originalFilename ?: file.name,
                    file.asRequestBody("image/jpeg".toMediaTypeOrNull()))
                retryRequest(record.photographerId) {
                    logApiStep("multipart") {
                        RetrofitClient.apiService.uploadPhoto(bearer(record.photographerId), record.eventId, body)
                    }
                }
            }
            if (!response.success || response.data == null) {
                settleFailure(record, response.errors?.any { it.code in TERMINAL_UPLOAD_ERROR_CODES } == true,
                    response.error ?: "Upload rejected by server.", database, failures)
                pacer.observe(0, failed = true)
                return
            }
            requireOwner(record.photographerId)
            check(backend == RetrofitClient.BASE_URL && !response.data.id.isNullOrBlank()) {
                "Upload confirmation changed. Original kept on phone."
            }
            // Upload acknowledgement is a receipt, not a byte-integrity proof. Cleanup re-reads
            // the owner-only cloud original before deleting; never shrink or expire this file.
            dao.completeUpload(record.id, response.data.id, backend)
            completed.incrementAndGet()
            pacer.observe((System.nanoTime() - started) / 1_000_000, failed = false)
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) {
            val error = RetrofitClient.parseHttpError(e)
            val code = (e as? HttpException)?.code() ?: (e as? StorageUploadException)?.status
            if (code == 401) {
                dao.updateStatus(record.id, "QUEUED", "Sign in to resume uploads.")
                unauthorized.set(true)
                return
            }
            val transport = e is IOException || code == 408 || code == 429 || (code != null && code >= 500)
            if (e is IOException) offline.set(true)
            val retryAt = when (e) {
                is HttpException -> retryAfter(e.response()?.headers()?.get("Retry-After"))
                is StorageUploadException -> e.retryAt
                else -> 0L
            }
            if (retryAt > 0) dao.defer(record.id, retryAt)
            pacer.observe((System.nanoTime() - started) / 1_000_000, failed = true)
            settleFailure(record, error?.code in TERMINAL_UPLOAD_ERROR_CODES || code in setOf(400, 404, 413, 415, 422),
                error?.message ?: e.message ?: "Upload interrupted.", database, failures, transport)
        }
    }

    private suspend fun <T> retryRequest(owner: String, call: suspend () -> T): T {
        val session = SessionManager.getInstance(applicationContext)
        var serverAttempts = 0
        var retriedAuth = false
        while (true) {
            requireOwner(owner)
            try { return call() }
            catch (e: Exception) {
                val code = (e as? HttpException)?.code() ?: (e as? StorageUploadException)?.status
                if (code == 401 && !retriedAuth && session.getUserEmail() == owner && session.getAccessToken() != null) {
                    retriedAuth = true
                    continue // The API call rereads bearer; OkHttp owns token refresh.
                }
                if ((code == 408 || (code != null && code in 500..599)) && ++serverAttempts < 3) {
                    delay(1_000L shl (serverAttempts - 1))
                    continue
                }
                throw e
            }
        }
    }

    private suspend fun <T> logApiStep(step: String, call: suspend () -> T): T {
        val started = System.nanoTime()
        var status = 0
        try {
            return call().also { status = 200 }
        } catch (e: HttpException) {
            status = e.code()
            throw e
        } finally { logStep(step, started, status) }
    }

    private fun logStep(step: String, started: Long, status: Int) {
        Log.i("QP/UPLOAD", "$step ms=${(System.nanoTime() - started) / 1_000_000} status=$status")
    }

    /**
     * A 429 on commit means the server's verification slots are full, not that the phone is
     * offline: wait (Retry-After as the floor, else 1, 2, 4, 8, 15 s with 20 % jitter) and commit
     * again, for at most [busyWindowMs] of clock time (calls included); then the last 429 goes to the caller's catch.
     */
    private suspend fun <T> waitOutBusy(recordId: Long, database: AppDatabase, call: suspend () -> T): T {
        val start = System.currentTimeMillis()
        var step = 0
        while (true) {
            try { return call() }
            catch (e: HttpException) {
                if (e.code() != 429) throw e
                val ladder = minOf((1_000L shl minOf(step++, 4)) * (0.8 + Math.random() * 0.4), 15_000.0).toLong()
                val floor = retryAfter(e.response()?.headers()?.get("Retry-After")) - System.currentTimeMillis()
                val wait = maxOf(floor, ladder)
                if (System.currentTimeMillis() - start + wait > busyWindowMs) throw e
                database.uploadQueueDao().defer(recordId, System.currentTimeMillis() + wait)
                delay(wait)
            }
        }
    }

    private suspend fun tryDirectUpload(file: File, record: UploadRecord, database: AppDatabase): ApiResponseEnvelope<UploadedPhotoDto>? {
        val dao = database.uploadQueueDao()
        val owner = record.photographerId
        val hash = record.contentHash ?: sha256Hex(file).also { dao.setContentHash(record.id, it) }
        val backend = RetrofitClient.BASE_URL
        suspend fun commit(photo: String, key: String) = waitOutBusy(record.id, database) {
            retryRequest(owner) {
                logApiStep("commit") {
                    RetrofitClient.apiService.commitDirectUpload(
                        bearer(owner), record.eventId, DirectUploadCommitRequest(photo, key, hash, "image/jpeg", file.length()))
                }
            }
        }

        // Check the saved slot first, even after an ambiguous PUT or lost commit response.
        // No presigned URL is persisted. A confirmed missing object permits a fresh begin/PUT.
        if (record.directBackend == backend && record.directPhotoId != null && record.directKey != null) {
            try { return commit(record.directPhotoId, record.directKey) }
            catch (e: HttpException) {
                if (RetrofitClient.parseHttpError(e)?.code != "UPLOAD_OBJECT_MISSING") throw e
                dao.checkpointDirectUpload(record.id, null, null, null)
            }
        }
        val begin = try {
            waitOutBusy(record.id, database) {
                retryRequest(owner) {
                    logApiStep("begin") {
                        RetrofitClient.apiService.beginDirectUpload(bearer(owner), record.eventId,
                            DirectUploadBeginRequest(hash, "image/jpeg", file.length(), record.originalFilename ?: file.name))
                    }
                }
            }
        } catch (e: HttpException) {
            if (e.code() == 404) return null
            throw e
        }
        if (!begin.success) throw IOException(begin.error ?: "Could not prepare the upload.")
        val plan = begin.data ?: throw IOException("The server returned no upload plan.")
        return when (plan.mode) {
            "existing" -> ApiResponseEnvelope(success = true, data = plan.existing
                ?: throw IOException("The server did not identify the existing photo."))
            "multipart" -> null
            "direct" -> {
                val url = plan.uploadUrl ?: throw IOException("Missing storage upload URL.")
                val photo = plan.photoId ?: throw IOException("Missing upload photo ID.")
                val key = plan.key ?: throw IOException("Missing storage key.")
                requireOwner(owner)
                dao.checkpointDirectUpload(record.id, photo, key, backend)
                val put = Request.Builder().url(url).put(file.asRequestBody("image/jpeg".toMediaTypeOrNull())).build()
                retryRequest(owner) { putToStorage(put) }
                commit(photo, key)
            }
            else -> throw IOException("The server returned an unsupported upload plan.")
        }
    }

    @OptIn(ExperimentalCoroutinesApi::class)
    private suspend fun putToStorage(request: Request) {
        val started = System.nanoTime()
        var status = 0
        try {
            val response = suspendCancellableCoroutine<Response> { continuation ->
                val call = RetrofitClient.rawClient.newCall(request)
                continuation.invokeOnCancellation { call.cancel() }
                call.enqueue(object : Callback {
                    override fun onFailure(call: Call, e: IOException) { continuation.resumeWithException(e) }
                    override fun onResponse(call: Call, response: Response) {
                        continuation.resume(response) { response.close() }
                    }
                })
            }
            response.use {
                status = it.code
                if (!it.isSuccessful) throw StorageUploadException(it.code, retryAfter(it.header("Retry-After")))
            }
        } finally { logStep("PUT", started, status) }
    }

    private class StorageUploadException(val status: Int, val retryAt: Long) : Exception("Storage upload returned HTTP $status.")

    private fun bearer(owner: String): String {
        val session = SessionManager.getInstance(applicationContext)
        val token = session.getAccessToken() ?: throw IOException("Sign in to resume uploads.")
        if (session.authSnapshot(token)?.owner != owner) throw IOException("Account changed; upload paused.")
        return "Bearer $token"
    }

    private fun requireOwner(owner: String) {
        val session = SessionManager.getInstance(applicationContext)
        if (session.getUserEmail() != owner || session.getAccessToken() == null)
            throw IOException("Sign in to the account that captured this photo.")
    }

    private suspend fun sha256Hex(file: File): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                currentCoroutineContext().ensureActive()
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private suspend fun settleFailure(
        record: UploadRecord, terminal: Boolean, message: String, database: AppDatabase,
        failures: AtomicBoolean, transport: Boolean = false,
    ) {
        val dao = database.uploadQueueDao()
        if (terminal) { dao.updateStatus(record.id, "FAILED", message); return }
        if (!transport) dao.incrementRetryCount(record.id)
        val retry = transport || record.retryCount + 1 < MAX_RETRIES
        dao.updateStatus(record.id, if (retry) "QUEUED" else "FAILED", message)
        if (retry) failures.set(true)
    }

    internal fun retryAfter(value: String?): Long {
        if (value == null) return 0
        value.toLongOrNull()?.let {
            return System.currentTimeMillis() + it.coerceIn(0, 7 * 24 * 60 * 60L) * 1000
        }
        return runCatching { ZonedDateTime.parse(value, DateTimeFormatter.RFC_1123_DATE_TIME).toInstant().toEpochMilli() }.getOrDefault(0)
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
            // shortcut: held at 2, production's verification slot count; when UP-3 raises the slots to 4, raise MAX to 4 or add a server slot hint.
            const val INITIAL_CONCURRENCY = 2
            const val MAX_CONCURRENCY = 2
            const val FAST_MS = 4_000L
            const val SLOW_MS = 15_000L
            const val FAST_STREAK = 3
        }
    }

    internal companion object {
        // Retire metadata only after verified local deletion. Originals never expire by age.
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

        // Longest a photo waits out "verification is busy" (429) inside one run; var so a test can shorten it.
        @Volatile internal var busyWindowMs = 90_000L
        @Volatile internal var runBudgetMs = 8 * 60_000L
    }
}
