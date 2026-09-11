package com.quickpitik.mobile.support

import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import kotlinx.coroutines.withTimeout
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.rules.ExternalResource
import org.junit.rules.TestWatcher
import org.junit.runner.Description

/**
 * Swaps Dispatchers.Main for an Unconfined test dispatcher for one test:
 * viewModelScope launches run eagerly and network resumptions complete on
 * OkHttp's threads, so a test just awaits the StateFlow. Replaces the
 * setMain/resetMain pair that used to be hand-written in every ViewModel test.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class MainDispatcherRule : TestWatcher() {
    override fun starting(description: Description) = Dispatchers.setMain(UnconfinedTestDispatcher())
    override fun finished(description: Description) = Dispatchers.resetMain()
}

/**
 * A MockWebServer the app's real HTTP stack talks to for one test, through the
 * RetrofitClient.setBaseUrl seam (debug builds only, which is what
 * testDebugUnitTest runs). Reset in after() because Robolectric's sandbox is
 * per-class, not per-method: a leaked URL would poison the next method.
 */
class MockBackend : ExternalResource() {
    val server = MockWebServer()

    override fun before() {
        server.start()
        RetrofitClient.setBaseUrl(server.url("/").toString())
    }

    override fun after() {
        server.shutdown()
        RetrofitClient.resetBaseUrl()
    }

    /** Queues [responses] in order; returns the server so a test can takeRequest(). */
    fun enqueue(vararg responses: MockResponse): MockWebServer =
        server.also { s -> responses.forEach(s::enqueue) }
}

/** The backend's success envelope around [data] (a JSON literal). */
fun ok(data: String): MockResponse = MockResponse().setBody("""{"success":true,"data":$data}""")

/** Blocks until the flow emits a value [accept]s; fails the test after [timeoutMs]. */
fun <T> Flow<T>.awaitFirst(timeoutMs: Long = 5_000, accept: (T) -> Boolean): T =
    runBlocking { withTimeout(timeoutMs) { first(accept) } }
