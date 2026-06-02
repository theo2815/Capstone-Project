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

        // 1. Get access token from session. If not logged in, we fail the job.
        val token = sessionManager.getAccessToken() ?: return Result.failure()

        // 2. Fetch all queued photos awaiting synchronization
        val pendingRecords = database.uploadQueueDao().getRecordsWithStatus("QUEUED")
        if (pendingRecords.isEmpty()) {
            return Result.success()
        }

        val hasFailures = AtomicBoolean(false)
        // Parallel upload gate. The 2026-05-28 diagnosis showed wall-clock per
        // photo was ~33s on demo Wi-Fi while PTP + persist combined was <0.1s —
        // the worker's serial loop was the bottleneck. 3 concurrent uploads
        // reuse a single OkHttp HTTP/2 connection (no extra TLS) and stay well
        // under Spring Boot tomcat's default 200-thread pool. A 24-photo race
        // batch drops from ~13 min to ~4-5 min. Bumping above 3 starts to
        // contend with the phone's radio and gives diminishing returns.
        val gate = Semaphore(permits = 3)

        coroutineScope {
            for (record in pendingRecords) {
                launch {
                    gate.withPermit { uploadOne(record, token, database, hasFailures) }
                }
            }
        }

        return if (hasFailures.get()) Result.retry() else Result.success()
    }

    private suspend fun uploadOne(
        record: UploadRecord,
        token: String,
        database: AppDatabase,
        hasFailures: AtomicBoolean,
    ) {
        // Mark record as UPLOADING to prevent duplicate workers picking it up
        database.uploadQueueDao().updateStatus(record.id, "UPLOADING", null)

        val file = File(record.filePath)
        if (!file.exists()) {
            database.uploadQueueDao().updateStatus(record.id, "FAILED", "Local file not found.")
            hasFailures.set(true)
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
            } else {
                val errorMsg = responseEnvelope.error ?: "Upload rejected by server."
                database.uploadQueueDao().updateStatus(record.id, "FAILED", errorMsg)
                database.uploadQueueDao().incrementRetryCount(record.id)
                hasFailures.set(true)
            }
        } catch (e: Exception) {
            val uploadMs = System.currentTimeMillis() - uploadStart
            android.util.Log.i(
                "QP/UPLOAD-PERF",
                "upload id=${record.id} file=${file.name} bytes=${file.length()} status=EXC ms=$uploadMs",
            )
            val errorMsg = e.localizedMessage ?: "Network connection timeout."
            database.uploadQueueDao().updateStatus(record.id, "FAILED", errorMsg)
            database.uploadQueueDao().incrementRetryCount(record.id)
            hasFailures.set(true)
        }
    }
}
