package com.quickpitik.mobile

import android.app.Application
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.runComposeUiTest
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import com.quickpitik.mobile.ui.photographer.PhotographerDashboardViewModel
import com.quickpitik.mobile.ui.photographer.PhotographerPurchasesViewModel
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CopyOnWriteArrayList

@RunWith(RobolectricTestRunner::class)
class StudioShareDestinationTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    private lateinit var application: Application
    private lateinit var dashboard: PhotographerDashboardViewModel
    private lateinit var purchases: PhotographerPurchasesViewModel
    private val paths = CopyOnWriteArrayList<String>()
    private var failDetail = false

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                paths += request.path.orEmpty()
                return when {
                    request.path.orEmpty().contains("/events/event-a?") && failDetail -> MockResponse().setResponseCode(503)
                    request.path.orEmpty().contains("/events/event-a?") -> ok("""{"id":"event-a","slug":"run","name":"Restored Run","date":"2026-10-01","location":"Cebu","state":"past","photoCount":1,"salesCount":0,"revenueKept":0,"deletedAt":null}""")
                    request.path.orEmpty().contains("purchased-photos") -> ok("""{"items":[],"total":0,"offset":0,"limit":50}""")
                    else -> MockResponse().setResponseCode(404)
                }
            }
        }
        dashboard = PhotographerDashboardViewModel(application)
        purchases = PhotographerPurchasesViewModel(application)
    }

    @After fun tearDown() { SessionManager.getInstance(application).clearSession() }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `restored share route fetches detail by id without cached event membership`() {
        var backs = 0
        runComposeUiTest {
            setContent { StudioShareDestination("event-a", "purchased", dashboard, purchases, onBack = { backs++ }, onOpenEvent = {}) }
            waitUntil(timeoutMillis = 5_000) { paths.any { it.contains("purchased-photos") } }
            onNodeWithText("Restored Run").assertExists()
            onNodeWithText("No completed purchases yet.").assertExists()
            runOnIdle {
                assertEquals(0, backs)
                assertTrue(paths.any { it.contains("includeRemoved=true") })
            }
        }
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `event outage remains mounted with retry and back`() {
        failDetail = true
        var backs = 0
        runComposeUiTest {
            setContent { StudioShareDestination("event-a", "purchased", dashboard, purchases, onBack = { backs++ }, onOpenEvent = {}) }
            waitUntil(timeoutMillis = 5_000) { purchases.eventState.value.error != null }
            onNodeWithText("Try again").assertExists()
            onNodeWithText("Back").assertExists()
            runOnIdle { failDetail = false }
            onNodeWithText("Try again").performClick()
            waitUntil(timeoutMillis = 5_000) { paths.count { it.contains("includeRemoved=true") } >= 2 }
            onNodeWithText("Restored Run").assertExists()
            runOnIdle { assertEquals(0, backs) }
        }
    }
}
