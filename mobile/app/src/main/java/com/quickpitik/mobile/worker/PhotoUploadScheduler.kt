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

    fun enqueue(context: Context): Job = schedule(context, restart = false)

    fun restart(context: Context): Job = schedule(context, restart = true)

    private fun schedule(context: Context, restart: Boolean): Job = scope.launch {
        scheduling.withLock {
            val manager = WorkManager.getInstance(context.applicationContext)
            var pending = manager.getWorkInfos(WorkQuery.Builder.fromUniqueWorkNames(listOf(WORK_NAME))
                .addStates(listOf(WorkInfo.State.ENQUEUED, WorkInfo.State.BLOCKED, WorkInfo.State.RUNNING)).build()).get()
            if (restart && pending.none { it.state == WorkInfo.State.RUNNING } &&
                pending.any { it.state == WorkInfo.State.ENQUEUED && it.runAttemptCount > 0 }) {
                manager.cancelUniqueWork(WORK_NAME).result.get()
                pending = emptyList()
            }
            // A not-yet-started drain will see this durable row. A RUNNING drain needs one
            // successor: it may already have observed an empty queue. Never append per frame.
            if (pending.any { it.state != WorkInfo.State.RUNNING && it.runAttemptCount == 0 }) return@withLock
            val request = OneTimeWorkRequestBuilder<PhotoUploadWorker>()
                .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 10, TimeUnit.SECONDS)
                .build()
            manager.enqueueUniqueWork(WORK_NAME, ExistingWorkPolicy.APPEND_OR_REPLACE, request).result.get()
        }
    }
}
