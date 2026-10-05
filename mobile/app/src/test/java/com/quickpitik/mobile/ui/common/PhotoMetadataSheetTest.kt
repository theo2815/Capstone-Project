package com.quickpitik.mobile.ui.common

import android.app.Application
import android.view.WindowManager
import androidx.test.core.app.ApplicationProvider
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performSemanticsAction
import androidx.compose.ui.semantics.SemanticsActions
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.local.LocalPhotoMetadata
import com.quickpitik.mobile.data.remote.PhotoMetadataDto
import com.quickpitik.mobile.data.remote.PhotoMetadataFieldsDto
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.atomic.AtomicInteger
import okhttp3.ResponseBody.Companion.toResponseBody
import retrofit2.HttpException
import retrofit2.Response
import org.robolectric.shadows.ShadowDialog

@RunWith(RobolectricTestRunner::class)
class PhotoMetadataSheetTest {
    @get:Rule val compose = createComposeRule()

    @Test fun `preview access follows owner mode or server hint only`() {
        val photo = PhotoPreviewData(id = "remote-uuid", price = 0.0, imageUrl = null, eventName = null)
        assertEquals("owner", metadataAccessFor(PhotoPreviewMode.OwnerReview, photo))
        assertEquals(null, metadataAccessFor(PhotoPreviewMode.Browse, photo))
        assertEquals(null, metadataAccessFor(PhotoPreviewMode.Owned, photo))
        assertEquals(null, metadataAccessFor(PhotoPreviewMode.LocalReview, photo))
        assertEquals("public", metadataAccessFor(PhotoPreviewMode.Owned, photo.copy(metadataAccess = "public")))
        assertEquals("owner", metadataAccessFor(PhotoPreviewMode.Browse, photo.copy(metadataAccess = "owner")))
        assertEquals(null, metadataAccessFor(PhotoPreviewMode.Browse, photo.copy(metadataAccess = "unexpected")))
        val remote = PhotoDto(
            id = "server-id", bib = null, km = null, time = "12:00", tone = 0, span = "portrait",
            price = 0.0, imageUrl = null, alt = null, metadataAccess = "public",
        )
        assertEquals("public", remote.toPreviewData(null).metadataAccess)
    }

    @Test fun `preview fetches only on tap and refetches after sheet close`() {
        val count = AtomicInteger()
        val app = ApplicationProvider.getApplicationContext<Application>()
        val vm = PhotoMetadataViewModel(app) { id, _ ->
            count.incrementAndGet()
            PhotoMetadataDto(id, "ready", "$id.jpg", 1024, null)
        }
        val index = androidx.compose.runtime.mutableIntStateOf(0)
        compose.setContent {
            PhotoPreview(
                photos = listOf(
                    PhotoPreviewData("remote-uuid", 0.0, null, null, metadataAccess = "public"),
                    PhotoPreviewData("other-uuid", 0.0, null, null, metadataAccess = "public"),
                ),
                currentIndex = index.intValue, onClose = {}, onIndexChange = {}, metadataViewModel = vm,
            )
        }
        compose.runOnIdle { assertEquals(0, count.get()) }
        compose.onNode(hasText("Show Metadata") and hasClickAction()).performSemanticsAction(SemanticsActions.OnClick)
        compose.waitUntil(5000) { count.get() == 1 }
        compose.onNode(hasText("Close metadata") and hasClickAction()).performSemanticsAction(SemanticsActions.OnClick)
        compose.onNodeWithText("Photo metadata").assertDoesNotExist()
        compose.onNodeWithContentDescription("Close preview").assertExists()
        compose.onNode(hasText("Show Metadata") and hasClickAction()).performSemanticsAction(SemanticsActions.OnClick)
        compose.waitUntil(5000) { count.get() == 2 }
        compose.runOnIdle { index.intValue = 1 }
        compose.waitForIdle()
        compose.waitUntil(5000) { vm.state.value == PhotoMetadataState.Closed }
        compose.onNodeWithText("Photo metadata").assertDoesNotExist()
        compose.runOnIdle { assertEquals(2, count.get()) }
    }

    @Test fun `public denial remains hidden after visiting another photo`() {
        val index = androidx.compose.runtime.mutableIntStateOf(0)
        val app = ApplicationProvider.getApplicationContext<Application>()
        val vm = PhotoMetadataViewModel(app) { id, _ ->
            if (id == "a") throw HttpException(Response.error<PhotoMetadataDto>(404, "".toResponseBody()))
            PhotoMetadataDto(id, "ready", "$id.jpg", 1024, null)
        }
        compose.setContent {
            PhotoPreview(
                photos = listOf(
                    PhotoPreviewData("a", 0.0, null, null, metadataAccess = "public"),
                    PhotoPreviewData("b", 0.0, null, null, metadataAccess = "public"),
                ),
                currentIndex = index.intValue, onClose = {}, onIndexChange = {}, metadataViewModel = vm,
            )
        }
        compose.onNode(hasText("Show Metadata") and hasClickAction()).performSemanticsAction(SemanticsActions.OnClick)
        compose.waitUntil(5000) { vm.state.value == PhotoMetadataState.Denied }
        compose.onNode(hasText("Close metadata") and hasClickAction()).performSemanticsAction(SemanticsActions.OnClick)
        compose.runOnIdle { index.intValue = 1 }
        compose.waitForIdle()
        compose.onNodeWithText("Show Metadata").assertExists()
        compose.runOnIdle { index.intValue = 0 }
        compose.waitForIdle()
        compose.onNodeWithText("Show Metadata").assertDoesNotExist()
    }

    @Test fun `Back dismisses only metadata and both Browse windows are secure`() {
        val app = ApplicationProvider.getApplicationContext<Application>()
        val vm = PhotoMetadataViewModel(app) { id, _ -> PhotoMetadataDto(id, "ready", null, null, null) }
        compose.setContent {
            PhotoPreview(
                photos = listOf(PhotoPreviewData("a", 0.0, null, null, metadataAccess = "public")),
                currentIndex = 0, onClose = {}, onIndexChange = {}, metadataViewModel = vm,
            )
        }
        compose.waitForIdle()
        val preview = ShadowDialog.getShownDialogs().last { it.isShowing }
        assertTrue("preview=${preview.javaClass.name}, window=${System.identityHashCode(preview.window)}, flags=${preview.window!!.attributes.flags}, shown=${ShadowDialog.getShownDialogs().map { it.javaClass.name to it.window?.attributes?.flags }}",
            preview.window!!.attributes.flags and WindowManager.LayoutParams.FLAG_SECURE != 0)
        compose.onNode(hasText("Show Metadata") and hasClickAction()).performSemanticsAction(SemanticsActions.OnClick)
        compose.onNodeWithText("Photo metadata").assertExists()
        compose.waitForIdle()
        val sheet = ShadowDialog.getShownDialogs().last { it.isShowing }
        assertFalse(sheet === preview)
        assertTrue(sheet.window!!.attributes.flags and WindowManager.LayoutParams.FLAG_SECURE != 0)
        compose.runOnIdle { sheet.onBackPressed() }
        compose.waitForIdle()
        compose.onNodeWithText("Photo metadata").assertDoesNotExist()
        compose.onNodeWithContentDescription("Close preview").assertExists()
        assertTrue(preview.isShowing)
        assertTrue(preview.window!!.attributes.flags and WindowManager.LayoutParams.FLAG_SECURE != 0)
    }

    @Test fun `ready sheet matches field order and formats camera date and size`() {
        compose.setContent {
            PhotoMetadataSheet(
                state = PhotoMetadataState.Data(metadata("ready", PhotoMetadataFieldsDto(
                    cameraMake = "Canon", cameraModel = "Canon EOS R6", lens = "A very long lens name that should wrap across the phone width",
                    aperture = 2.8, exposureTime = "1/500 s", iso = 400, flash = null,
                    exposureMode = "Manual", whiteBalance = "Auto", author = "Ana",
                    copyright = "Copyright Ana", takenAtLocal = "2026-09-28T07:45:12",
                    takenAtOffset = "+08:00",
                ))),
                onClose = {}, onRefresh = {}, secure = false,
            )
        }
        compose.onNodeWithText("Canon EOS R6").assertIsDisplayed()
        compose.onNodeWithText("Canon Canon EOS R6").assertDoesNotExist()
        compose.onNodeWithText("A very long lens name that should wrap across the phone width").assertExists()
        compose.onNodeWithText("f/2.8").assertExists()
        compose.onNodeWithText("1/500 s").assertExists()
        compose.onNodeWithText("Not recorded").assertExists()
        compose.onNodeWithText("2026-09-28 07:45:12 +08:00").assertExists()
        compose.onNodeWithText("1.5 MiB").assertExists()
        compose.onNodeWithContentDescription("Original size, 1.5 MiB, 1,572,864 bytes").assertExists()
    }

    @Test fun `pending and server failed metadata still show file facts`() {
        val state = androidx.compose.runtime.mutableStateOf<PhotoMetadataState>(PhotoMetadataState.Pending(metadata("pending")))
        compose.setContent { PhotoMetadataSheet(state.value, onClose = {}, onRefresh = {}, secure = false) }
        compose.onNodeWithText("race.jpg").assertExists()
        compose.runOnIdle { state.value = PhotoMetadataState.Failed("Could not prepare metadata. Try again.", metadata("failed")) }
        compose.onNodeWithText("Metadata could not be loaded.").assertIsDisplayed()
        compose.onNodeWithText("race.jpg").assertExists()
    }

    @Test fun `pending and failure offer only explicit refresh or retry`() {
        var refreshes = 0
        val state = androidx.compose.runtime.mutableStateOf<PhotoMetadataState>(PhotoMetadataState.Pending(metadata("pending")))
        compose.setContent {
            PhotoMetadataSheet(state.value, onClose = {}, onRefresh = { refreshes++ }, secure = false)
        }
        compose.onNodeWithText("Metadata is being prepared.").assertIsDisplayed()
        compose.onNode(hasText("Refresh") and hasClickAction()).performSemanticsAction(SemanticsActions.OnClick)
        compose.runOnIdle { assertEquals(1, refreshes); state.value = PhotoMetadataState.Failed("offline") }
        compose.onNodeWithText("Metadata could not be loaded.").assertIsDisplayed()
        compose.onNode(hasText("Retry") and hasClickAction()).performSemanticsAction(SemanticsActions.OnClick)
        compose.runOnIdle { assertEquals(2, refreshes); state.value = PhotoMetadataState.Denied }
        compose.onNodeWithText("Metadata is no longer available").assertIsDisplayed()
        compose.onNodeWithText("Retry").assertDoesNotExist()
    }

    @Test fun `empty and unavailable copy differs while empty keeps file facts`() {
        val state = androidx.compose.runtime.mutableStateOf<PhotoMetadataState>(PhotoMetadataState.Empty(metadata("empty")))
        compose.setContent {
            PhotoMetadataSheet(state.value, onClose = {}, onRefresh = {}, secure = false)
        }
        compose.onNodeWithText("No camera metadata was found.").assertIsDisplayed()
        compose.onNodeWithText("race.jpg").assertExists()
        compose.runOnIdle { state.value = PhotoMetadataState.Unavailable(metadata("unavailable")) }
        compose.onNodeWithText("Metadata is unavailable for this photo.").assertIsDisplayed()
        compose.onNodeWithText("race.jpg").assertExists()
    }

    @Test fun `local empty camera tags still identify the phone original`() {
        compose.setContent {
            PhotoMetadataSheet(
                state = PhotoMetadataState.LocalEmpty(LocalPhotoMetadata("empty", null, "race.jpg", 1_572_864)),
                onClose = {}, onRefresh = {}, secure = false,
            )
        }
        compose.onNodeWithText("From phone original").assertIsDisplayed()
        compose.onNodeWithText("No camera metadata was found.").assertIsDisplayed()
        compose.onNodeWithText("race.jpg").assertExists()
    }

    private fun metadata(status: String, fields: PhotoMetadataFieldsDto? = null) = PhotoMetadataDto(
        photoId = "server-id", status = status, originalFilename = "race.jpg",
        fileSizeBytes = 1_572_864, fields = fields,
    )
}
