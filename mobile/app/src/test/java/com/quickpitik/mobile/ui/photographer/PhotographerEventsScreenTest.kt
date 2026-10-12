package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.filter
import androidx.compose.ui.test.onAncestors
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.runComposeUiTest
import androidx.compose.ui.unit.Density
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.closeRoom
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
class PhotographerEventsScreenTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    private lateinit var application: Application

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse =
                if (request.path.orEmpty().contains("/events")) ok("""{"items":[],"total":0,"offset":0,"limit":200}""")
                else MockResponse().setResponseCode(404)
        }
    }

    @After fun tearDown() {
        closeRoom(application)
        SessionManager.getInstance(application).clearSession()
    }

    @OptIn(ExperimentalTestApi::class)
    @Config(qualifiers = "w360dp-h800dp")
    @Test fun `status chips stay whole by scrolling sideways at large font scale`() {
        val dashboard = PhotographerDashboardViewModel(application)
        val tether = TetherViewModel(application, dashboard.eventsState)
        dashboard.fetchEvents()
        dashboard.eventsState.awaitFirst { it is EventsState.Success }
        dashboard.fetchPublicEvents()
        dashboard.publicEventsState.awaitFirst { it is EventsState.Success }
        runComposeUiTest {
            setContent {
                val density = LocalDensity.current
                CompositionLocalProvider(LocalDensity provides Density(density.density, 1.45f)) {
                    PhotographerEventsScreen(viewModel = dashboard, tetherViewModel = tether)
                }
            }
            for (label in listOf("UPLOADING · 0", "UPCOMING · 0", "COMPLETED · 0")) {
                val scrolls = onNodeWithText(label, useUnmergedTree = true).onAncestors()
                    .filter(SemanticsMatcher.keyIsDefined(SemanticsProperties.HorizontalScrollAxisRange))
                assertTrue("$label sits in a row that cannot scroll sideways", scrolls.fetchSemanticsNodes().isNotEmpty())
            }
        }
    }
}
