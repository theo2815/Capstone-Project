package com.quickpitik.mobile.ui.photographer

import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.SemanticsActions
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.runComposeUiTest
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.unit.Density
import org.junit.Assert.assertTrue
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
class StudioShellTest {
    @Test fun `the banner title names the server wait only while busyUntil is in the future`() {
        assertEquals("Server busy, retrying in 7 s", uploadBannerTitle(1, 2, 17_000, 10_000))
        assertEquals("Uploading photos", uploadBannerTitle(1, 2, 9_000, 10_000))
        assertEquals("Uploading photos", uploadBannerTitle(1, 0, 0, 10_000))
        assertEquals("Photos queued", uploadBannerTitle(0, 2, 0, 10_000))
    }

    @OptIn(ExperimentalTestApi::class)
    @Config(qualifiers = "w360dp-h800dp")
    @Test fun `bottom nav labels do not outgrow their fixed-width slots at large font scale`() = runComposeUiTest {
        setContent {
            val density = LocalDensity.current
            CompositionLocalProvider(LocalDensity provides Density(density.density, 1.45f)) {
                PhotographerFloatingBottomNav(currentRoute = "studio/home", showSettingsBadge = false, onNavigate = {})
            }
        }
        for (label in listOf("Home", "Capture", "Events", "Earnings", "Settings")) {
            val layouts = mutableListOf<TextLayoutResult>()
            onNodeWithText(label, useUnmergedTree = true).fetchSemanticsNode()
                .config[SemanticsActions.GetTextLayoutResult].action!!.invoke(layouts)
            // Robolectric measures every glyph at 1px, so a cut-off label cannot be
            // seen here; the cap on the label's own scale is what keeps it whole.
            val input = layouts.single().layoutInput
            assertTrue("$label scales ${input.density.fontScale}x", input.density.fontScale <= 1.15f)
        }
    }
}
