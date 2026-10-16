package com.quickpitik.mobile.ui.photographer

import android.app.Application
import androidx.lifecycle.ViewModelStore
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.remote.ApiResponseEnvelope
import com.quickpitik.mobile.data.remote.PaginatedResponse
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.support.pointAppAt
import com.quickpitik.mobile.support.signInAs
import kotlinx.coroutines.CompletableDeferred
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlin.coroutines.Continuation
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine
import org.junit.Assert.*
import org.junit.After
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class PublicPhotographerViewModelTest {
    private val application = ApplicationProvider.getApplicationContext<Application>().also(RetrofitClient::init)
    private val main = MainDispatcherRule()
    private val backend = MockBackend()
    @get:Rule val rules: RuleChain = RuleChain.outerRule(main).around(backend)
    private val models = mutableListOf<PublicPhotographerViewModel>()
    private fun track(vm: PublicPhotographerViewModel) = vm.also(models::add)
    @After fun closeModels() {
        models.forEach { ViewModelStore().apply { put("gallery", it); clear() } }
    }

    private fun photo(hint: String) = """{"id":"123e4567-e89b-12d3-a456-426614174000","bib":null,"km":null,"time":"06:30","tone":0,"span":"portrait","price":100.0,"imageUrl":"/photo.jpg","alt":"Photo","metadataAccess":"$hint"}"""
    private fun page(hint: String) = """{"items":[${photo(hint)}],"total":1,"offset":0,"limit":100}"""
    private fun hint(vm: PublicPhotographerViewModel) =
        (vm.profileEventPhotosState.awaitFirst { it is ProfileEventPhotosState.Success } as ProfileEventPhotosState.Success)
            .photos.single().metadataAccess

    @Test fun `guest gallery has no bearer and profile stays public`() {
        com.quickpitik.mobile.data.local.SessionManager.getInstance(application).clearSession()
        backend.enqueue(ok("{}"), ok(page("public")))
        val vm = track(PublicPhotographerViewModel(application))
        vm.fetchPublicProfile("ana")
        vm.publicProfileState.awaitFirst { it is PublicProfileState.Success }
        vm.fetchProfileEventPhotos("ana", "race")
        assertEquals("public", hint(vm))
        val profile = backend.server.takeRequest(5, TimeUnit.SECONDS)!!
        val gallery = backend.server.takeRequest(5, TimeUnit.SECONDS)!!
        assertEquals("/api/v1/public/photographers/ana", profile.path)
        assertNull(profile.getHeader("Authorization"))
        assertEquals("/api/v1/public/photographers/ana/events/race/photos?offset=0&limit=100", gallery.path)
        assertNull(gallery.getHeader("Authorization"))
    }

    @Test fun `signed in gallery sends bearer and receives owner hint while profile stays public`() {
        signInAs("PHOTOGRAPHER")
        backend.enqueue(ok("{}"), ok(page("owner")))
        val vm = track(PublicPhotographerViewModel(application))
        vm.fetchPublicProfile("ana")
        vm.publicProfileState.awaitFirst { it is PublicProfileState.Success }
        vm.fetchProfileEventPhotos("ana", "race")
        assertEquals("owner", hint(vm))
        assertNull(backend.server.takeRequest(5, TimeUnit.SECONDS)!!.getHeader("Authorization"))
        assertEquals("Bearer t", backend.server.takeRequest(5, TimeUnit.SECONDS)!!.getHeader("Authorization"))
    }

    @Test fun `same photo loses owner hint and refetches for changed account`() {
        val session = signInAs("PHOTOGRAPHER")
        backend.enqueue(ok(page("owner")), ok(page("public")))
        val vm = track(PublicPhotographerViewModel(application))
        vm.fetchProfileEventPhotos("ana", "race")
        assertEquals("owner", hint(vm))
        session.saveSession("other", "RUNNER", "Other", "other@x.com")
        assertEquals("public", hint(vm))
        assertEquals("Bearer t", backend.server.takeRequest(5, TimeUnit.SECONDS)!!.getHeader("Authorization"))
        assertEquals("Bearer other", backend.server.takeRequest(5, TimeUnit.SECONDS)!!.getHeader("Authorization"))
        assertEquals(2, backend.server.requestCount)
    }

    @Test fun `reload clears old list and late response cannot restore it`() {
        signInAs("PHOTOGRAPHER")
        val old = AtomicReference<Continuation<ApiResponseEnvelope<PaginatedResponse<PhotoDto>>>>()
        val started = CountDownLatch(1)
        val releaseNew = CompletableDeferred<Unit>()
        val vm = track(PublicPhotographerViewModel(application) { _, slug, _ ->
            if (slug == "old") suspendCoroutine { continuation -> old.set(continuation); started.countDown() }
            else { releaseNew.await(); response("public") }
        })
        vm.fetchProfileEventPhotos("ana", "old")
        assertTrue(started.await(5, TimeUnit.SECONDS))
        vm.fetchProfileEventPhotos("bob", "new")
        assertEquals(ProfileEventPhotosState.Loading, vm.profileEventPhotosState.value)
        releaseNew.complete(Unit)
        assertEquals("public", hint(vm))
        old.get().resume(response("owner"))
        Thread.sleep(100)
        assertEquals("public", (vm.profileEventPhotosState.value as ProfileEventPhotosState.Success).photos.single().metadataAccess)
    }

    @Test fun `reloading the same gallery removes its old owner hint immediately`() {
        signInAs("PHOTOGRAPHER")
        var calls = 0
        val vm = track(PublicPhotographerViewModel(application) { _, _, _ ->
            if (++calls == 1) response("owner") else kotlinx.coroutines.awaitCancellation()
        })
        vm.fetchProfileEventPhotos("ana", "race")
        assertEquals("owner", hint(vm))
        vm.fetchProfileEventPhotos("ana", "race")
        assertEquals(ProfileEventPhotosState.Loading, vm.profileEventPhotosState.value)
    }

    @Test fun `backend switch refetches same photo without carrying owner hint`() {
        signInAs("PHOTOGRAPHER")
        var calls = 0
        val vm = track(PublicPhotographerViewModel(application) { _, _, _ ->
            response(if (++calls == 1) "owner" else "public")
        })
        vm.fetchProfileEventPhotos("ana", "race")
        assertEquals("owner", hint(vm))
        pointAppAt("http://127.0.0.1:12345/")
        assertEquals("public", hint(vm))
        assertEquals(2, calls)
    }

    @Test fun `logout and backend change clear visible hints and fence late responses`() {
        val session = signInAs("PHOTOGRAPHER")
        val old = AtomicReference<Continuation<ApiResponseEnvelope<PaginatedResponse<PhotoDto>>>>()
        val started = CountDownLatch(1)
        var calls = 0
        val vm = track(PublicPhotographerViewModel(application) { _, _, _ ->
            if (++calls == 1) response("owner")
            else if (calls == 2) suspendCoroutine { continuation -> old.set(continuation); started.countDown() }
            else kotlinx.coroutines.awaitCancellation()
        })
        vm.fetchProfileEventPhotos("ana", "race")
        assertEquals("owner", hint(vm))
        session.clearSession()
        assertTrue(started.await(5, TimeUnit.SECONDS))
        assertEquals(ProfileEventPhotosState.Loading, vm.profileEventPhotosState.value)
        pointAppAt("http://127.0.0.1:12345/")
        old.get().resume(response("owner"))
        Thread.sleep(100)
        assertFalse(vm.profileEventPhotosState.value is ProfileEventPhotosState.Success)
    }

    private fun response(hint: String): ApiResponseEnvelope<PaginatedResponse<PhotoDto>> =
        ApiResponseEnvelope(true, PaginatedResponse(listOf(PhotoDto(
            "123e4567-e89b-12d3-a456-426614174000", null, null, "06:30", 0,
            "portrait", 100.0, "/photo.jpg", "Photo", metadataAccess = hint,
        )), 1, 0, 100))
}
