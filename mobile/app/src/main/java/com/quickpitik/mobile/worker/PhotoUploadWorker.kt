package com.quickpitik.mobile.worker

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.RetrofitClient
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File

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

        var hasFailures = false

        for (record in pendingRecords) {
            // Mark record as UPLOADING to prevent duplicate workers picking it up
            database.uploadQueueDao().updateStatus(record.id, "UPLOADING", null)

            val file = File(record.filePath)
            if (!file.exists()) {
                database.uploadQueueDao().updateStatus(record.id, "FAILED", "Local file not found.")
                hasFailures = true
                continue
            }

            try {
                // 3. Construct the multipart request body
                val requestFile = file.asRequestBody("image/jpeg".toMediaTypeOrNull())
                val body = MultipartBody.Part.createFormData(
                    "file", // Must match backend @RequestPart("file") parameter name
                    file.name,
                    requestFile
                )

                val bearerToken = "Bearer $token"

                // 4. Dispatch REST transaction
                val responseEnvelope = RetrofitClient.apiService.uploadPhoto(
                    token = bearerToken,
                    eventId = record.eventId,
                    file = body
                )

                if (responseEnvelope.success && responseEnvelope.data != null) {
                    // Update state to COMPLETED upon successfully landing in your S3 bucket!
                    database.uploadQueueDao().updateStatus(record.id, "COMPLETED", null)
                } else {
                    val errorMsg = responseEnvelope.error ?: "Upload rejected by server."
                    database.uploadQueueDao().updateStatus(record.id, "FAILED", errorMsg)
                    database.uploadQueueDao().incrementRetryCount(record.id)
                    hasFailures = true
                }
            } catch (e: Exception) {
                val errorMsg = e.localizedMessage ?: "Network connection timeout."
                database.uploadQueueDao().updateStatus(record.id, "FAILED", errorMsg)
                database.uploadQueueDao().incrementRetryCount(record.id)
                hasFailures = true
            }
        }

        return if (hasFailures) Result.retry() else Result.success()
    }
}
