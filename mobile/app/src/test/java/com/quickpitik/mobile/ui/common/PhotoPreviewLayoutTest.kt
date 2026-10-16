package com.quickpitik.mobile.ui.common

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Text
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTouchInput
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.runComposeUiTest
import androidx.compose.ui.unit.Density
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
class PhotoPreviewLayoutTest {
    private val photo = PhotoPreviewData("photo-a", 50.0, null, "Cebu Run", metadataAccess = "owner")

    @Test fun `lifting both fingers skips the zoom transform`() {
        assertNull(
            zoomOffsetAfterGesture(Offset(120f, -30f), Offset.Unspecified, Offset(300f, 300f), 2f, 2f, Offset.Zero),
        )
    }

    @OptIn(ExperimentalTestApi::class)
    private fun checkDetails(fontScale: Float = 1f) = runComposeUiTest {
        setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale)) {
                PhotoPreview(
                    photos = listOf(photo), currentIndex = 0, onClose = {}, onIndexChange = {},
                    mode = PhotoPreviewMode.OwnerReview,
                    details = {
                        Column {
                            Text("Buyer")
                            Text("Aira S.")
                            Text("Reference")
                            Text("A1B2C3D4")
                            repeat(24) { Text("Purchase fact $it") }
                        }
                    },
                )
            }
        }
        onNodeWithContentDescription("Close preview").assertIsDisplayed()
        assertTrue(onNodeWithText("QUICKPITIK PREVIEW").fetchSemanticsNode().boundsInRoot.height > 0f)
        onNodeWithText("Aira S.", useUnmergedTree = true).performScrollTo().assertIsDisplayed()
        onNodeWithText("A1B2C3D4", useUnmergedTree = true).performScrollTo().assertIsDisplayed()
        onNodeWithText("Show Metadata", useUnmergedTree = true).performScrollTo().assertIsDisplayed()
        onNodeWithContentDescription("Close preview").assertIsDisplayed()
    }

    @Test @Config(qualifiers = "w411dp-h914dp") fun `details fit portrait`() = checkDetails()
    @Test @Config(qualifiers = "w914dp-h411dp") fun `details fit landscape`() = checkDetails()
    @Test @Config(qualifiers = "w640dp-h360dp") fun `details fit large font short landscape`() = checkDetails(1.5f)
    @Test @Config(qualifiers = "w768dp-h1024dp") fun `details fit tablet`() = checkDetails()

    @OptIn(ExperimentalTestApi::class)
    private fun checkChromeApart(fontScale: Float = 1f, withDetails: Boolean = false) = runComposeUiTest {
        val photos = (0 until 3).map { PhotoPreviewData("photo-$it", 50.0, null, "Cebu Run") }
        setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, fontScale)) {
                PhotoPreview(
                    photos, 1, {}, {}, mode = PhotoPreviewMode.OwnerReview,
                    details = if (withDetails) ({ Column { repeat(24) { Text("Purchase fact $it") } } }) else null,
                )
            }
        }
        val dp = density.density
        val chrome = listOf(
            "Close preview" to onNodeWithContentDescription("Close preview").fetchSemanticsNode().boundsInRoot,
            "Previous" to onNodeWithContentDescription("Previous").fetchSemanticsNode().boundsInRoot,
            "Next" to onNodeWithContentDescription("Next").fetchSemanticsNode().boundsInRoot,
            "Counter" to onNodeWithText("2 / 3", useUnmergedTree = true).fetchSemanticsNode().boundsInRoot.let {
                // The pill is 48dp high with 16dp side padding around its text.
                Rect(it.left - 16f * dp, it.center.y - 24f * dp, it.right + 16f * dp, it.center.y + 24f * dp)
            },
        )
        chrome.take(3).forEach { (name, r) ->
            assertTrue("$name is ${r.width / dp} x ${r.height / dp} dp", r.width >= 48f * dp - 0.5f && r.height >= 48f * dp - 0.5f)
        }
        for (i in chrome.indices) for (j in i + 1 until chrome.size) {
            assertTrue("${chrome[i].first} ${chrome[i].second} overlaps ${chrome[j].first} ${chrome[j].second}", !chrome[i].second.overlaps(chrome[j].second))
        }
    }

    @Test @Config(qualifiers = "w411dp-h914dp") fun `chrome apart portrait`() = checkChromeApart()
    @Test @Config(qualifiers = "w914dp-h411dp") fun `chrome apart landscape`() = checkChromeApart()
    @Test @Config(qualifiers = "w640dp-h360dp") fun `chrome apart short landscape`() = checkChromeApart()
    @Test @Config(qualifiers = "w640dp-h360dp") fun `chrome apart short landscape large font`() = checkChromeApart(1.5f)
    @Test @Config(qualifiers = "w768dp-h1024dp") fun `chrome apart tablet`() = checkChromeApart()
    @Test @Config(qualifiers = "w411dp-h914dp") fun `chrome apart portrait with details`() = checkChromeApart(withDetails = true)
    @Test @Config(qualifiers = "w914dp-h411dp") fun `chrome apart landscape with details`() = checkChromeApart(withDetails = true)
    @Test @Config(qualifiers = "w640dp-h360dp") fun `chrome apart short landscape with details`() = checkChromeApart(withDetails = true)
    @Test @Config(qualifiers = "w640dp-h360dp") fun `chrome apart short landscape large font with details`() = checkChromeApart(1.5f, true)
    @Test @Config(qualifiers = "w768dp-h1024dp") fun `chrome apart tablet with details`() = checkChromeApart(withDetails = true)

    @OptIn(ExperimentalTestApi::class)
    @Config(qualifiers = "w411dp-h914dp")
    @Test fun `counter shows the whole set when only part of it is loaded`() = runComposeUiTest {
        val photos = (0 until 5).map { PhotoPreviewData("photo-$it", 50.0, null, "Cebu Run") }
        setContent { PhotoPreview(photos, 1, {}, {}, mode = PhotoPreviewMode.OwnerReview, totalCount = 210) }
        onNodeWithText("2 / 210").assertExists()
    }

    @OptIn(ExperimentalTestApi::class)
    @Config(qualifiers = "w411dp-h914dp")
    @Test fun `tapping next during the page animation moves forward without bouncing back`() = runComposeUiTest {
        val photos = (0 until 20).map { PhotoPreviewData("photo-$it", 50.0, null, "Cebu Run") }
        var index by mutableStateOf(10)
        val reported = mutableListOf<Int>()
        setContent { PhotoPreview(photos, index, {}, { reported += it; index = it }, mode = PhotoPreviewMode.OwnerReview) }
        mainClock.autoAdvance = false
        onNodeWithContentDescription("Next").performClick()
        // Second tap must land before the 10 → 11 slide reaches page 11.
        mainClock.advanceTimeBy(48)
        onNodeWithContentDescription("Next").performClick()
        mainClock.advanceTimeBy(5_000)
        mainClock.autoAdvance = true
        waitForIdle()
        assertEquals(listOf(11, 12), reported)
        onNodeWithText("13 / 20").assertExists()
    }

    @OptIn(ExperimentalTestApi::class)
    @Config(qualifiers = "w411dp-h914dp")
    @Test fun `grabbing the pager mid-slide reports the page the user settles on`() = runComposeUiTest {
        val photos = (0 until 20).map { PhotoPreviewData("photo-$it", 50.0, null, "Cebu Run") }
        var index by mutableStateOf(10)
        val reported = mutableListOf<Int>()
        setContent { PhotoPreview(photos, index, {}, { reported += it; index = it }, mode = PhotoPreviewMode.OwnerReview) }
        mainClock.autoAdvance = false
        onNodeWithContentDescription("Next").performClick()
        mainClock.advanceTimeBy(48)
        onNodeWithText("QUICKPITIK PREVIEW").performTouchInput { down(center); moveBy(Offset(5f, 0f)); up() }
        mainClock.advanceTimeBy(5_000)
        mainClock.autoAdvance = true
        waitForIdle()
        assertEquals(listOf(11, 10), reported)
        onNodeWithText("11 / 20").assertExists()
    }

    @OptIn(ExperimentalTestApi::class)
    @Config(qualifiers = "w411dp-h914dp")
    @Test fun `defaulted reload controls leave sibling modes unchanged`() = runComposeUiTest {
        var mode by mutableStateOf(PhotoPreviewMode.Browse)
        setContent { PhotoPreview(listOf(photo), 0, {}, {}, mode = mode) }
        onNodeWithText("Buy now").assertExists()
        onNodeWithText("Refresh image").assertDoesNotExist()
        runOnIdle { mode = PhotoPreviewMode.Owned }
        onNodeWithText("Download photo ↓").assertExists()
        runOnIdle { mode = PhotoPreviewMode.OwnerReview }
        onNodeWithText("SOLD").assertExists()
        runOnIdle { mode = PhotoPreviewMode.LocalReview }
        onNodeWithText("Close").assertExists()
        onNodeWithText("Refresh image").assertDoesNotExist()
    }
}
