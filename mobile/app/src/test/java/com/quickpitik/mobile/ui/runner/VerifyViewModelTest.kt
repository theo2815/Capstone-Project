package com.quickpitik.mobile.ui.runner

import android.app.Application
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.remote.PhotoVerifyResultDto
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import okhttp3.mockwebserver.MockResponse
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import java.io.ByteArrayInputStream

/**
 * "Whose photo is this?": an image over the backend's 10 MB cap is refused on
 * the phone, the upload is public (no bearer), and both the answer and a
 * refusal reach the screen as they came.
 */
@RunWith(RobolectricTestRunner::class)
class VerifyViewModelTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private fun picked(name: String, size: Int): Uri = Uri.parse("content://test/$name").also {
        val app = ApplicationProvider.getApplicationContext<Application>()
        Shadows.shadowOf(app.contentResolver).registerInputStream(it, ByteArrayInputStream(ByteArray(size)))
    }

    @Test fun `oversized picks stay on the phone, answers and refusals reach the screen`() {
        val vm = VerifyViewModel(ApplicationProvider.getApplicationContext())
        vm.verify(picked("huge.jpg", 10 * 1024 * 1024 + 1))
        assertEquals(VerifyState.Error("That image is over 10 MB. Pick a smaller one."), vm.state.awaitFirst { it is VerifyState.Error })
        assertEquals(0, backend.server.requestCount)

        val server = backend.enqueue(
            ok("""{"matched":true,"confidence":"high","photographerName":"Ana","photographerHandle":"ana","eventName":"Cebu Run","eventDate":"2026-08-01"}"""),
            MockResponse().setResponseCode(415)
                .setBody("""{"success":false,"errors":[{"code":"UNSUPPORTED_MEDIA","message":"Only JPEG or PNG photos."}]}"""),
        )
        vm.verify(picked("a.jpg", 16))
        assertEquals(
            VerifyState.Result(PhotoVerifyResultDto(true, "high", "Ana", "ana", "Cebu Run", "2026-08-01")),
            vm.state.awaitFirst { it is VerifyState.Result },
        )
        val request = server.takeRequest()
        assertEquals("/api/v1/public/photos/verify", request.path)
        assertNull(request.getHeader("Authorization"))
        assertTrue(request.body.readUtf8().contains("""filename="photo.jpg""""))

        vm.verify(picked("b.jpg", 16))
        assertEquals(VerifyState.Error("Only JPEG or PNG photos."), vm.state.awaitFirst { it is VerifyState.Error })
    }
}
