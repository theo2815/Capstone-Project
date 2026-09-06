package com.quickpitik.mobile.worker

import android.content.Context
import androidx.work.*
import java.util.concurrent.TimeUnit

object PhotoUploadScheduler {
    fun enqueue(context: Context) {
        val request = OneTimeWorkRequestBuilder<PhotoUploadWorker>()
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
            .setBackoffCriteria(BackoffPolicy.LINEAR, 10, TimeUnit.SECONDS)
            .build()
        // Each durable enqueue has a successor even if the current worker just observed an empty queue.
        // ponytail: one lightweight successor per ingest kick; coalesce only if measured WorkManager overhead warrants it.
        WorkManager.getInstance(context).enqueueUniqueWork("photo-upload-sync", ExistingWorkPolicy.APPEND_OR_REPLACE, request)
    }
}
