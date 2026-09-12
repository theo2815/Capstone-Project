package com.quickpitik.mobile.support

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.AppDatabase
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.TestCoroutineScheduler
import kotlinx.coroutines.test.TestDispatcher
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.withTimeout
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.rules.ExternalResource
import org.junit.rules.TestWatcher
import org.junit.runner.Description
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.TimeUnit

/**
 * Swaps Dispatchers.Main for a test dispatcher for one test. Unconfined by
 * default: viewModelScope launches run eagerly and network resumptions complete
 * on OkHttp's threads, so a test just awaits the StateFlow. Pass a
 * StandardTestDispatcher instead when the code under test delays: its
 * [scheduler] then skips those delays in virtual time.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class MainDispatcherRule(private val dispatcher: TestDispatcher = UnconfinedTestDispatcher()) : TestWatcher() {
    val scheduler: TestCoroutineScheduler get() = dispatcher.scheduler
    override fun starting(description: Description) = Dispatchers.setMain(dispatcher)
    override fun finished(description: Description) = Dispatchers.resetMain()
}

/**
 * A MockWebServer the app's real HTTP stack talks to for one test, through the
 * [pointAppAt] seam. Reset in after() because that seam is a process-wide
 * static: a leaked URL would poison the next method.
 */
class MockBackend : ExternalResource() {
    val server = MockWebServer()

    override fun before() {
        server.start()
        pointAppAt(server.url("/").toString())
    }

    override fun after() {
        server.shutdown()
        RetrofitClient.resetBaseUrl()
    }

    /** Queues [responses] in order; returns the server so a test can takeRequest(). */
    fun enqueue(vararg responses: MockResponse): MockWebServer =
        server.also { s -> responses.forEach(s::enqueue) }
}

/**
 * Points the app's HTTP stack at [url], failing closed. RetrofitClient.setBaseUrl
 * returns false and changes nothing outside a debug build, so unchecked the suite
 * would keep the compiled-in production origin and every test that thinks it is
 * talking to a MockWebServer would dial the real backend instead. Every test that
 * moves the app off its default origin goes through here.
 */
fun pointAppAt(url: String) {
    check(RetrofitClient.setBaseUrl(url)) {
        "The backend seam needs the debug variant (RetrofitClient.setBaseUrl is a no-op elsewhere): run testDebugUnitTest"
    }
}

/** The backend's success envelope around [data] (a JSON literal). */
fun ok(data: String): MockResponse = MockResponse().setBody("""{"success":true,"data":$data}""")

/** Signs in a fresh session as [role], the starting state of most ViewModel tests. */
fun signInAs(role: String): SessionManager =
    SessionManager.getInstance(ApplicationProvider.getApplicationContext()).apply {
        clearSession()
        saveSession("t", role, "Juan", "juan@x.com")
    }

/** Blocks until the flow emits a value [accept]s; fails the test after [timeoutMs]. */
fun <T> Flow<T>.awaitFirst(timeoutMs: Long = 5_000, accept: (T) -> Boolean): T =
    runBlocking { withTimeout(timeoutMs) { first(accept) } }

/**
 * Blocks until every query already handed to Room's IO pool has finished. A
 * ViewModel's Room flows can still be mid-query when its test ends; closing the
 * database under them throws "Illegal connection pointer" into the next
 * coroutine test. Clear the ViewModels first (so nothing new starts), call
 * this, then close the database.
 */
fun awaitRoomIdle(application: Application) {
    // Room's default executor is androidx.arch.core's fixed pool of 4 threads
    // (arch_disk_io_*, core-runtime 2.2.0). One barrier task per thread: once
    // all four run at the same time, everything queued before them is done.
    // ponytail: pool size read off core-runtime 2.2.0 with javap; re-check on
    // upgrade (a bigger pool would drain early, a smaller one times out).
    val barrier = CyclicBarrier(ROOM_IO_THREADS + 1)
    val io = AppDatabase.getDatabase(application).queryExecutor
    repeat(ROOM_IO_THREADS) { io.execute { barrier.await(5, TimeUnit.SECONDS) } }
    barrier.await(5, TimeUnit.SECONDS)
}

private const val ROOM_IO_THREADS = 4

/**
 * Ends a ViewModel test's use of Room: drains the IO pool, closes the database
 * and forgets the singleton. The forget is the part a bare close() misses -
 * AppDatabase caches INSTANCE and never nulls it, and Robolectric shares one
 * sandbox (and the app statics in it) across test classes with the same config,
 * so the next class would inherit a closed database. Clear the ViewModels first.
 */
fun closeRoom(application: Application) {
    awaitRoomIdle(application)
    AppDatabase.getDatabase(application).close()
    forgetRoomSingleton()
}

/**
 * Nulls AppDatabase.INSTANCE. Reflection because the field is private and the
 * alternative - a test-only reset hook - would be a production change made
 * purely to serve a test. Kotlin lowers a companion-object backing field to a
 * static on the enclosing class, hence the lookup here.
 */
fun forgetRoomSingleton() {
    AppDatabase::class.java.getDeclaredField("INSTANCE").apply {
        isAccessible = true
        set(null, null)
    }
}
