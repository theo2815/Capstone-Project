package com.quickpitik.mobile.worker

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import androidx.work.Configuration
import androidx.work.CoroutineWorker
import androidx.work.WorkerFactory
import androidx.work.WorkerParameters
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.testing.SynchronousExecutor
import androidx.work.testing.WorkManagerTestInitHelper
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.CompletableDeferred
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertTrue

@RunWith(RobolectricTestRunner::class)
class PhotoUploadSchedulerTest {
    @Test fun `a running drain gets exactly one successor for captures arriving as it finishes`() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val entered = CountDownLatch(1)
        val release = CompletableDeferred<Unit>()
        val factory = object : WorkerFactory() {
            override fun createWorker(appContext: Context, workerClassName: String, workerParameters: WorkerParameters) =
                object : CoroutineWorker(appContext, workerParameters) {
                    override suspend fun doWork(): Result {
                        entered.countDown()
                        release.await()
                        return Result.success()
                    }
                }
        }
        WorkManagerTestInitHelper.initializeTestWorkManager(context,
            Configuration.Builder().setExecutor(SynchronousExecutor()).setWorkerFactory(factory).build())
        val manager = WorkManager.getInstance(context)
        try {
            PhotoUploadScheduler.enqueue(context).join()
            val first = manager.getWorkInfosForUniqueWork(PhotoUploadScheduler.WORK_NAME).get().single()
            WorkManagerTestInitHelper.getTestDriver(context)!!.setAllConstraintsMet(first.id)
            assertTrue(entered.await(5, TimeUnit.SECONDS))
            List(100) { PhotoUploadScheduler.enqueue(context) }.forEach { it.join() }
            val work = manager.getWorkInfosForUniqueWork(PhotoUploadScheduler.WORK_NAME).get()
            assertEquals(1, work.count { it.state == WorkInfo.State.RUNNING })
            assertEquals(1, work.count { it.state == WorkInfo.State.BLOCKED })
            assertEquals(2, work.size)
        } finally {
            manager.cancelAllWork().result.get()
            release.complete(Unit)
            WorkManagerTestInitHelper.closeWorkDatabase()
        }
    }

    @Test fun `a burst without connectivity schedules one durable drain`() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Context>()
        WorkManagerTestInitHelper.initializeTestWorkManager(context,
            Configuration.Builder().setExecutor(SynchronousExecutor()).build())
        try {
            val manager = WorkManager.getInstance(context)
            List(100) { PhotoUploadScheduler.enqueue(context) }.forEach { it.join() }
            val work = manager.getWorkInfosForUniqueWork(PhotoUploadScheduler.WORK_NAME).get()
            assertEquals(1, work.size)
            assertEquals(0, work.single().runAttemptCount)
        } finally { WorkManagerTestInitHelper.closeWorkDatabase() }
    }
}
