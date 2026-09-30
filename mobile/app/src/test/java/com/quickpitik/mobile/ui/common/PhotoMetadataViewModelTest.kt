package com.quickpitik.mobile.ui.common

import android.app.Application
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.PhotoMetadataDto
import com.quickpitik.mobile.data.remote.PhotoMetadataFieldsDto
import com.quickpitik.mobile.data.local.LocalPhotoMetadata
import com.quickpitik.mobile.data.local.UploadRecord
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.signInAs
import kotlin.coroutines.Continuation
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference
import okhttp3.mockwebserver.MockResponse
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
class PhotoMetadataViewModelTest {
    private val application = ApplicationProvider.getApplicationContext<Application>().also(RetrofitClient::init)
    private val main = MainDispatcherRule()
    private val backend = MockBackend()
    @get:Rule val rules: RuleChain = RuleChain.outerRule(main).around(backend)

    private fun viewModel() = PhotoMetadataViewModel(application)
    private fun metadata(id: String, status: String = "ready", model: String = id) =
        """{"photoId":"$id","status":"$status","originalFilename":"$id.jpg","fileSizeBytes":1024,"fields":{"cameraMake":"Canon","cameraModel":"$model","lens":null,"aperture":null,"exposureTime":null,"iso":null,"author":null,"copyright":null,"takenAtLocal":null,"takenAtOffset":null,"flash":null,"exposureMode":null,"whiteBalance":null}}"""

    @Test fun `local record shows its own metadata without using Room ID as photo ID`() {
        signInAs("PHOTOGRAPHER")
        val row = UploadRecord(id = 7, filePath = "ignored", eventId = "event", photographerId = "juan@x.com", captureTimestamp = 1)
        val vm = PhotoMetadataViewModel(application,
            getRecord = { if (it == 7L) row else null },
            readLocal = { LocalPhotoMetadata("ready", PhotoMetadataFieldsDto("Canon", "EOS R6", null, null, null, null, null, null, null, null, null, null, null), "race.jpg", 1024) })
        vm.selectLocal(7)
        val state = vm.state.awaitFirst { it is PhotoMetadataState.LocalData } as PhotoMetadataState.LocalData
        assertEquals("EOS R6", state.metadata.fields?.cameraModel)
        assertEquals("race.jpg", state.metadata.originalFilename)
        assertEquals(0, backend.server.requestCount)
    }

    @Test fun `cleaned matching receipt requests owner metadata by uploaded photo ID`() {
        signInAs("PHOTOGRAPHER")
        val row = UploadRecord(id = 7, filePath = "gone", eventId = "event", photographerId = "juan@x.com",
            captureTimestamp = 1, uploadStatus = "COMPLETED", uploadedPhotoId = "cloud-photo",
            uploadedBackend = RetrofitClient.BASE_URL, localDeletedAt = 1)
        backend.enqueue(ok(metadata("cloud-photo")))
        val vm = PhotoMetadataViewModel(application, getRecord = { row },
            readLocal = { LocalPhotoMetadata("unavailable", null, null, null) })
        vm.selectLocal(7)
        val state = vm.state.awaitFirst { it is PhotoMetadataState.Data } as PhotoMetadataState.Data
        assertEquals("cloud-photo", state.metadata.photoId)
        assertTrue(backend.server.takeRequest().path!!.contains("cloud-photo"))
    }

    @Test fun `foreign backend receipt and foreign account never call owner API`() {
        signInAs("PHOTOGRAPHER")
        val row = UploadRecord(id = 7, filePath = "gone", eventId = "event", photographerId = "juan@x.com",
            captureTimestamp = 1, uploadStatus = "COMPLETED", uploadedPhotoId = "cloud-photo",
            uploadedBackend = "https://other.invalid/", localDeletedAt = 1)
        val vm = PhotoMetadataViewModel(application, getRecord = { row },
            readLocal = { LocalPhotoMetadata("unavailable", null, null, null) })
        vm.selectLocal(7)
        vm.state.awaitFirst { it is PhotoMetadataState.LocalUnavailable }
        assertEquals(0, backend.server.requestCount)
        val foreign = PhotoMetadataViewModel(application, getRecord = { row.copy(photographerId = "other") },
            readLocal = { error("foreign record was opened") })
        foreign.selectLocal(7)
        foreign.state.awaitFirst { it is PhotoMetadataState.LocalUnavailable }
        assertEquals(0, backend.server.requestCount)
    }

    @Test fun `late local parse cannot repaint a newly selected upload row`() {
        signInAs("PHOTOGRAPHER")
        val old = AtomicReference<Continuation<LocalPhotoMetadata>>()
        val started = CountDownLatch(1)
        val finished = CountDownLatch(1)
        val vm = PhotoMetadataViewModel(application,
            getRecord = { id -> UploadRecord(id = id, filePath = "unused", eventId = "event",
                photographerId = "juan@x.com", captureTimestamp = 1) },
            readLocal = { row ->
                if (row.id == 1L) suspendCoroutine<LocalPhotoMetadata> { old.set(it); started.countDown() }
                    .also { finished.countDown() }
                else LocalPhotoMetadata("ready", PhotoMetadataFieldsDto("Canon", "new", null, null, null,
                    null, null, null, null, null, null, null, null), "new.jpg", 4)
            })
        vm.selectLocal(1)
        assertTrue(started.await(5, TimeUnit.SECONDS))
        vm.selectLocal(2)
        vm.state.awaitFirst { it is PhotoMetadataState.LocalData && it.metadata.originalFilename == "new.jpg" }
        old.get().resume(LocalPhotoMetadata("ready", null, "old.jpg", 4))
        assertTrue(finished.await(5, TimeUnit.SECONDS))
        Thread.sleep(100)
        assertEquals("new.jpg", (vm.state.value as PhotoMetadataState.LocalData).metadata.originalFilename)
    }

    @Test fun `backend switch while local file disappears cannot send receipt to new backend`() {
        signInAs("PHOTOGRAPHER")
        val calls = java.util.concurrent.atomic.AtomicInteger()
        val row = UploadRecord(id = 7, filePath = "gone", eventId = "event", photographerId = "juan@x.com",
            captureTimestamp = 1, uploadStatus = "COMPLETED", uploadedPhotoId = "cloud-photo",
            uploadedBackend = RetrofitClient.BASE_URL)
        val started = CountDownLatch(1)
        val finished = CountDownLatch(1)
        val suspended = AtomicReference<Continuation<LocalPhotoMetadata>>()
        val vm = PhotoMetadataViewModel(application, getRecord = { row }, readLocal = {
            suspendCoroutine<LocalPhotoMetadata> { suspended.set(it); started.countDown() }
                .also { finished.countDown() }
        }, load = { _, _ -> calls.incrementAndGet(); PhotoMetadataDto("cloud-photo", "ready", null, null, null) })
        vm.selectLocal(7)
        assertTrue(started.await(5, TimeUnit.SECONDS))
        assertTrue(RetrofitClient.setBaseUrl("http://127.0.0.1:12345/"))
        vm.state.awaitFirst { it == PhotoMetadataState.Closed }
        suspended.get().resume(LocalPhotoMetadata("unavailable", null, null, null))
        assertTrue(finished.await(5, TimeUnit.SECONDS))
        Thread.sleep(100)
        assertEquals(0, calls.get())
    }

    @Test fun `one fetch on open and explicit refresh for pending`() {
        signInAs("RUNNER")
        backend.enqueue(ok(metadata("a", "pending")), ok(metadata("a", "ready")))
        val vm = viewModel()
        vm.selectRemote("a", "public")
        vm.state.awaitFirst { it is PhotoMetadataState.Pending }
        assertEquals(1, backend.server.requestCount)
        vm.refresh()
        assertEquals("a", vm.state.awaitFirst { it is PhotoMetadataState.Data }.let { (it as PhotoMetadataState.Data).metadata.photoId })
        assertEquals(2, backend.server.requestCount)
    }

    @Test fun `unavailable preserves file facts without camera fields`() {
        signInAs("RUNNER")
        backend.enqueue(ok("""{"photoId":"a","status":"unavailable","originalFilename":"race.jpg","fileSizeBytes":1572864,"fields":null}"""))
        val vm = viewModel()
        vm.selectRemote("a", "public")
        val state = vm.state.awaitFirst { it is PhotoMetadataState.Unavailable } as PhotoMetadataState.Unavailable
        assertEquals("race.jpg", state.metadata.originalFilename)
        assertNull(state.metadata.fields)
    }

    @Test fun `old photo response cannot replace newly selected photo`() {
        signInAs("RUNNER")
        backend.enqueue(ok(metadata("a", model = "old")).setBodyDelay(2, TimeUnit.SECONDS), ok(metadata("b", model = "new")))
        val vm = viewModel()
        vm.selectRemote("a", "public")
        backend.server.takeRequest(5, TimeUnit.SECONDS)
        vm.selectRemote("b", "public")
        assertEquals("new", (vm.state.awaitFirst { it is PhotoMetadataState.Data } as PhotoMetadataState.Data).metadata.fields?.cameraModel)
        Thread.sleep(2200)
        assertEquals("b", (vm.state.value as PhotoMetadataState.Data).metadata.photoId)
    }

    @Test fun `clear and invalid access prevent old or unauthorized fields`() {
        signInAs("RUNNER")
        backend.enqueue(ok(metadata("a")).setBodyDelay(2, TimeUnit.SECONDS))
        val vm = viewModel()
        vm.selectRemote("a", "public")
        backend.server.takeRequest(5, TimeUnit.SECONDS)
        vm.clear()
        Thread.sleep(2200)
        assertEquals(PhotoMetadataState.Closed, vm.state.value)
        vm.selectRemote("a", "owned")
        assertEquals(PhotoMetadataState.Closed, vm.state.value)
        assertEquals(1, backend.server.requestCount)
    }

    @Test fun `public denial clears prior fields and retry recovers`() {
        signInAs("RUNNER")
        backend.enqueue(ok(metadata("a")), MockResponse().setResponseCode(404), MockResponse().setResponseCode(500), ok(metadata("a")))
        val vm = viewModel()
        vm.selectRemote("a", "public")
        vm.state.awaitFirst { it is PhotoMetadataState.Data }
        vm.refresh()
        vm.state.awaitFirst { it is PhotoMetadataState.Denied }
        vm.refresh()
        vm.state.awaitFirst { it is PhotoMetadataState.Failed }
        vm.refresh()
        vm.state.awaitFirst { it is PhotoMetadataState.Data }
        assertEquals(4, backend.server.requestCount)
    }

    @Test fun `logout and backend change fence late responses`() {
        val session = signInAs("RUNNER")
        backend.enqueue(ok(metadata("a")).setBodyDelay(2, TimeUnit.SECONDS), ok(metadata("b")).setBodyDelay(2, TimeUnit.SECONDS))
        val vm = viewModel()
        vm.selectRemote("a", "public")
        backend.server.takeRequest(5, TimeUnit.SECONDS)
        session.clearSession()
        Thread.sleep(2200)
        assertEquals(PhotoMetadataState.Closed, vm.state.value)
        vm.selectRemote("b", "public")
        backend.server.takeRequest(5, TimeUnit.SECONDS)
        assertTrue(RetrofitClient.setBaseUrl("http://127.0.0.1:12345/"))
        Thread.sleep(2200)
        assertEquals(PhotoMetadataState.Closed, vm.state.value)
    }

    @Test fun `logout clears metadata already displayed`() {
        val session = signInAs("RUNNER")
        backend.enqueue(ok(metadata("a")))
        val vm = viewModel()
        vm.selectRemote("a", "public")
        vm.state.awaitFirst { it is PhotoMetadataState.Data }
        session.clearSession()
        assertEquals(PhotoMetadataState.Closed, vm.state.awaitFirst { it == PhotoMetadataState.Closed })
    }

    @Test fun `backend switch clears metadata already displayed`() {
        signInAs("RUNNER")
        backend.enqueue(ok(metadata("a")))
        val vm = viewModel()
        vm.selectRemote("a", "public")
        vm.state.awaitFirst { it is PhotoMetadataState.Data }
        assertTrue(RetrofitClient.setBaseUrl("http://127.0.0.1:12345/"))
        assertEquals(PhotoMetadataState.Closed, vm.state.awaitFirst { it == PhotoMetadataState.Closed })
    }

    @Test fun `noncooperative canceled response cannot repaint a new photo`() {
        signInAs("RUNNER")
        val started = CountDownLatch(1)
        val completed = CountDownLatch(1)
        val old = AtomicReference<Continuation<PhotoMetadataDto>>()
        fun dto(id: String) = PhotoMetadataDto(id, "ready", "$id.jpg", 1024,
            PhotoMetadataFieldsDto("Canon", id, null, null, null, null, null, null, null, null, null, null, null))
        val vm = PhotoMetadataViewModel(application) { id, _ ->
            if (id == "a") suspendCoroutine<PhotoMetadataDto> { continuation -> old.set(continuation); started.countDown() }
                .also { completed.countDown() }
            else dto(id)
        }
        vm.selectRemote("a", "public")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        vm.selectRemote("b", "public")
        vm.state.awaitFirst { it is PhotoMetadataState.Data && it.metadata.photoId == "b" }
        old.get().resume(dto("a"))
        assertTrue(completed.await(5, TimeUnit.SECONDS))
        Thread.sleep(100)
        assertEquals("b", (vm.state.value as PhotoMetadataState.Data).metadata.photoId)
    }
}
