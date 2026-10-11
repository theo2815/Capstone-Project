package com.quickpitik.mobile.ui.common

import androidx.compose.foundation.layout.Column
import androidx.compose.material3.Text
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
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
    @Config(qualifiers = "w411dp-h914dp")
    @Test fun `tapping next during the page animation moves forward without bouncing back`() = runComposeUiTest {
        val photos = (0 until 20).map { PhotoPreviewData("photo-$it", 50.0, null, "Cebu Run") }
        var index by mutableStateOf(10)
        val reported = mutableListOf<Int>()
        setContent { PhotoPreview(photos, index, {}, { reported += it; index = it }, mode = PhotoPreviewMode.OwnerReview) }
        mainClock.autoAdvance = false
        onNodeWithContentDescription("Next").performClick()
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
