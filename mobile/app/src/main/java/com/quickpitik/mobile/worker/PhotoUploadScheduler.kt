package com.quickpitik.mobile.worker

import android.content.Context
import androidx.work.*
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.TimeUnit

object PhotoUploadScheduler {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO + CoroutineExceptionHandler { _, error ->
        // Scheduling I/O no longer runs in the caller's try/catch. A later enqueue/startup
        // retries the durable queue; a scheduling failure must not crash the camera service.
        android.util.Log.e("PhotoUploadScheduler", "Could not schedule uploads; queued originals are retained.", error)
    })
    private val scheduling = Mutex()
    internal const val WORK_NAME = "photo-upload-sync"

    fun enqueue(context: Context): Job = scope.launch {
        scheduling.withLock {
            val manager = WorkManager.getInstance(context.applicationContext)
            val pending = manager.getWorkInfos(WorkQuery.Builder.fromUniqueWorkNames(listOf(WORK_NAME))
                .addStates(listOf(WorkInfo.State.ENQUEUED, WorkInfo.State.BLOCKED)).build()).get()
            // A not-yet-started drain will see this durable row. A RUNNING drain needs one
            // successor: it may already have observed an empty queue. Never append per frame.
            if (pending.any { it.runAttemptCount == 0 }) return@withLock
            val request = OneTimeWorkRequestBuilder<PhotoUploadWorker>()
                .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 10, TimeUnit.SECONDS)
                .build()
            manager.enqueueUniqueWork(WORK_NAME, ExistingWorkPolicy.APPEND_OR_REPLACE, request).result.get()
        }
    }
}
