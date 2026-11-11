package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.runComposeUiTest
import androidx.compose.ui.semantics.SemanticsActions
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.ProvinceDto
import com.quickpitik.mobile.data.remote.RegionDto
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowToast
import java.util.concurrent.atomic.AtomicInteger

@RunWith(RobolectricTestRunner::class)
class PhotographerRaceFormScreenTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()
    private val models = ViewModelStore()
    private val application: Application get() = ApplicationProvider.getApplicationContext()

    @Before fun setUp() {
        signInAs("PHOTOGRAPHER")
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse = when (request.path?.substringBefore('?')) {
                "/api/v1/race-types" -> ok("""{"selfServeEnabled":true,"types":[{"code":"ROAD_RACE","label":"Road race","hint":"On roads","shared":true},{"code":"OTHER","label":"Other","hint":"Your own event","shared":false}]}""")
                "/api/v1/me/photographer/verification" -> ok("""{"status":"approved","missing":[],"missingCodes":[]}""")
                "/api/v1/me/photographer/races" -> ok("""{"outcome":"match","candidates":[{"id":"race","name":"Cebu Run","date":"2026-11-11","location":"Cebu","reviewStatus":"pending"}],"checkReasons":[]}""")
                "/api/v1/me/photographer/races/race/join" -> ok("""{"outcome":"joined","event":{"id":"race","name":"Cebu Run","reviewStatus":"pending"},"candidates":[],"checkReasons":[]}""")
                else -> ok("""{"items":[],"total":0,"offset":0,"limit":100}""")
            }
        }
    }

    @After fun tearDown() {
        models.clear()
        SessionManager.getInstance(application).clearSession()
    }

    @OptIn(ExperimentalTestApi::class)
    @Config(qualifiers = "w360dp-h800dp")
    @Test fun `race creation requires its location codes and pending join tells the photographer to wait`() {
        val vm = PhotographerDashboardViewModel(application).also { models.put("dashboard", it) }
        val saved = AtomicInteger()
        val regions = listOf(RegionDto("07", "Central Visayas", "VII", "Visayas", listOf(ProvinceDto("0722", "Cebu"))))
        runComposeUiTest {
            setContent { PhotographerEventFormScreen(vm, null, {}, { saved.incrementAndGet() }, regions) }
            waitUntil(timeoutMillis = 5_000) { vm.raceTypes.value?.selfServeEnabled == true }
            onNodeWithText("Pick a race type, or Other").performClick()
            onNodeWithText("Road race").performClick()
            onNodeWithText("Paksit Sunrise Run").performTextInput("Cebu Run")
            onNodeWithText("SRP Boulevard, Cebu City").performTextInput("Cebu City")
            onNodeWithText("2026-09-20").performTextInput("2026-11-11")
            onNodeWithText("Create race").performScrollTo().assertIsNotEnabled()
            onNodeWithText("Pick a region").performScrollTo().performClick()
            onNodeWithText("Central Visayas").performClick()
            onNodeWithText("Pick a province").performScrollTo().performClick()
            onNodeWithText("Cebu", substring = false).performClick()
            onNodeWithText("Create race").performScrollTo().performClick()
            waitUntil(timeoutMillis = 5_000) { vm.raceFormState.value is RaceFormState.Duplicate }
            onNodeWithText(RACE_PENDING_JOIN_LINE).assertExists()
            // Invoke the sheet window's accessibility click; Robolectric's touch injection misses it.
            onNodeWithText("Join Cebu Run").performScrollTo().assertIsEnabled()
                .performSemanticsAction(SemanticsActions.OnClick) { it() }
            waitUntil(timeoutMillis = 5_000) { vm.raceFormState.value is RaceFormState.PendingReview }
            waitUntil(timeoutMillis = 5_000) { saved.get() == 1 }
            assertEquals(RACE_PENDING_JOIN_LINE, ShadowToast.getTextOfLatestToast())
        }
    }
}
