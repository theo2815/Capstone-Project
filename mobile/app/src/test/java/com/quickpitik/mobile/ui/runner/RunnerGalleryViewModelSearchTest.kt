package com.quickpitik.mobile.ui.runner

import android.app.Application
import android.net.Uri
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.MAX_UPLOAD_BYTES
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import java.io.ByteArrayInputStream
import java.time.LocalDate
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * The event cockpit's search wall: bib search and paging over a frozen
 * snapshot, the stored-selfie and fresh-selfie face searches with their copy,
 * the photo-alert switch's revert, and the rule that a late response from a
 * search the runner has already left never lands on the new screen.
 */
@RunWith(RobolectricTestRunner::class)
class RunnerGalleryViewModelSearchTest {
    @get:Rule val main = MainDispatcherRule()
    @get:Rule val backend = MockBackend()

    private lateinit var application: Application

    /** One recorded request, with the query params the wall cares about. */
    private data class Seen(
        val method: String,
        val path: String,
        val bib: String?,
        val offset: String?,
        val snapshotAt: String?,
        val type: String?,
        val body: String,
    )
    private val seen = CopyOnWriteArrayList<Seen>()

    @Volatile private var selfies = "[]"
    @Volatile private var selfieUpload: MockResponse = ok(selfie("s9", primary = false))
    @Volatile private var alertToggle: MockResponse = ok("""{"registered":true,"selfieId":"s1"}""")
    @Volatile private var failNextPage = false
    @Volatile private var holdNextPage: CountDownLatch? = null

    @Before fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        SessionManager.getInstance(application).apply {
            clearSession()
            saveSession("t", "RUNNER", "Juan", "juan@x.com")
        }
        backend.server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                val url = request.requestUrl!!
                val path = url.encodedPath
                val r = Seen(
                    request.method.orEmpty(), path, url.queryParameter("bib"), url.queryParameter("offset"),
                    url.queryParameter("snapshotAt"), request.getHeader("Content-Type"), request.body.readUtf8(),
                )
                seen += r
                return when {
                    path.endsWith("/photos/search-by-face") -> faceSearch(r)
                    path.endsWith("/photos") -> page(r)
                    path.endsWith("/photo-alert") && r.method == "GET" -> ok("""{"registered":false,"selfieId":null}""")
                    path.endsWith("/photo-alert") -> alertToggle
                    path == "/api/v1/me/selfies" && r.method == "GET" -> ok(selfies)
                    path == "/api/v1/me/selfies" -> selfieUpload
                    else -> MockResponse().setResponseCode(404) // event detail is best-effort
                }
            }
        }
    }

    @After fun tearDown() {
        holdNextPage?.countDown()
        SessionManager.getInstance(application).clearSession()
    }

    // Browse: 4 photos, served 2 + (1 repeat, 1 new). Bib: 2 photos, served 1 + (1 repeat, 1 new).
    private fun page(r: Seen): MockResponse {
        val offset = r.offset?.toInt() ?: 0
        if (offset > 0) {
            holdNextPage?.let { latch -> holdNextPage = null; latch.await(5, TimeUnit.SECONDS) }
            if (failNextPage) {
                failNextPage = false
                return MockResponse().setResponseCode(500)
            }
        }
        return when {
            r.bib == null && offset == 0 -> photos(listOf("p1", "p2"), total = 4, snapshotAt = "S1")
            r.bib == null -> photos(listOf("p2", "p3"), total = 4)
            offset == 0 -> photos(listOf("b1"), total = 2, snapshotAt = "S2")
            else -> photos(listOf("b1", "b2"), total = 2)
        }
    }

    // A selfie id spelled as an error code is refused with that code, the way
    // the backend passes ai-api codes through; any other id matches.
    private fun faceSearch(r: Seen): MockResponse {
        val id = Regex(""""selfieId":"([^"]+)"""").find(r.body)?.groupValues?.get(1)
        val code = id?.takeIf { it.all { c -> c.isUpperCase() || c == '_' } }
            ?: return photos(listOf("f1", "f2"), total = 2)
        // LOW_CONFIDENCE and NO_FACES arrive without a message: the copy must not depend on one.
        val message = if (code == "LOW_CONFIDENCE" || code == "NO_FACES") "" else ""","message":"server: $code""""
        return MockResponse().setResponseCode(422).setBody("""{"success":false,"errors":[{"code":"$code"$message}]}""")
    }

    private fun photos(ids: List<String>, total: Int, snapshotAt: String? = null): MockResponse {
        val items = ids.joinToString(",") {
            """{"id":"$it","bib":null,"km":null,"time":"08:00","tone":0,"span":"portrait","price":100,"imageUrl":null,"alt":null}"""
        }
        val snap = snapshotAt?.let { ""","snapshotAt":"$it"""" } ?: ""
        return ok("""{"items":[$items],"total":$total,"offset":0,"limit":100$snap}""")
    }

    private fun event(slug: String, date: LocalDate = LocalDate.now().minusDays(10)) =
        EventDto("id-$slug", slug, "Cebu Run", date.toString(), "Cebu", null, 0, 0, "COMPLETED")

    private fun pages() = seen.filter { it.path.endsWith("/photos") }
    private fun faceSearches() = seen.filter { it.path.endsWith("/photos/search-by-face") }
    private fun selfieUploads() = seen.count { it.method == "POST" && it.path == "/api/v1/me/selfies" }
    private fun ids(state: PhotosSearchState) = (state as PhotosSearchState.Success).photos.map { it.id }

    private fun RunnerGalleryViewModel.awaitPhotos(vararg ids: String) = searchState.awaitFirst {
        it is PhotosSearchState.Success && !it.loadingMore && it.photos.map { p -> p.id } == ids.toList()
    }

    private fun RunnerGalleryViewModel.awaitError() =
        (searchState.awaitFirst { it is PhotosSearchState.Error } as PhotosSearchState.Error).message

    private fun awaitUntil(check: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 5_000
        while (!check() && System.currentTimeMillis() < deadline) Thread.sleep(5)
        assertTrue(check())
    }

    private fun picked(name: String, size: Int): Uri = Uri.parse("content://test/$name").also {
        Shadows.shadowOf(application.contentResolver).registerInputStream(it, ByteArrayInputStream(ByteArray(size)))
    }

    @Test fun `browse and bib pages - snapshot only for browse, repeats dropped, a failed page keeps the grid`() {
        val vm = RunnerGalleryViewModel(application)
        vm.selectEvent(event("ev"))
        vm.awaitPhotos("p1", "p2")
        assertFalse(vm.isFiltered.value)

        failNextPage = true
        vm.loadMorePhotos()
        awaitUntil { pages().size == 2 && (vm.searchState.value as? PhotosSearchState.Success)?.loadingMore == false }
        assertEquals(listOf("p1", "p2"), ids(vm.searchState.value)) // kept, and the button is live again

        vm.loadMorePhotos()
        vm.awaitPhotos("p1", "p2", "p3")

        vm.searchByBib("  A12 ")
        vm.awaitPhotos("b1")
        assertTrue(vm.isFiltered.value)
        vm.loadMorePhotos()
        vm.awaitPhotos("b1", "b2")
        vm.loadMorePhotos() // everything is shown: nothing to ask for

        vm.clearFilter()
        vm.awaitPhotos("p1", "p2")
        assertFalse(vm.isFiltered.value)

        assertEquals(
            listOf(
                Triple(null, "0", null),
                Triple(null, "2", "S1"), // the failed page
                Triple(null, "2", "S1"),
                Triple("A12", "0", null),
                Triple("A12", "1", null), // bib pages stay live-ranked: no snapshot
                Triple(null, "0", null),
            ),
            pages().map { Triple(it.bib, it.offset, it.snapshotAt) },
        )
    }

    @Test fun `a late page from a screen the runner has left never lands on the new one`() {
        val vm = RunnerGalleryViewModel(application)
        vm.selectEvent(event("ev"))
        vm.awaitPhotos("p1", "p2")

        // Load more is still out when the runner opens a race that hasn't happened yet.
        val held = CountDownLatch(1).also { holdNextPage = it }
        vm.loadMorePhotos()
        awaitUntil { pages().size == 2 }
        vm.selectEvent(event("soon", LocalDate.now().plusDays(7)))
        held.countDown()
        Thread.sleep(500)
        assertEquals(PhotosSearchState.Idle, vm.searchState.value)

        // Load more is still out when the runner submits a bib.
        vm.selectEvent(event("ev"))
        vm.awaitPhotos("p1", "p2")
        val held2 = CountDownLatch(1).also { holdNextPage = it }
        vm.loadMorePhotos()
        awaitUntil { pages().size == 4 }
        vm.searchByBib("A12")
        vm.awaitPhotos("b1")
        held2.countDown()
        Thread.sleep(500)
        assertEquals(listOf("b1"), ids(vm.searchState.value))
        assertTrue(vm.isFiltered.value)
    }

    @Test fun `stored-selfie search uses the whole library, falls back to the primary, and says when there is none`() {
        selfies = "[${selfie("s1", primary = false)}]"
        val noPrimary = RunnerGalleryViewModel(application)
        noPrimary.selectEvent(event("ev"))
        noPrimary.awaitPhotos("p1", "p2")
        noPrimary.searchByStoredSelfie()
        assertEquals("No primary selfie set. Please upload a selfie and set it as primary first.", noPrimary.awaitError())
        assertTrue(faceSearches().isEmpty())

        selfies = "[${selfie("s1", primary = false)},${selfie("s2", primary = true)}]"
        val vm = RunnerGalleryViewModel(application)
        vm.selectEvent(event("ev"))
        vm.awaitPhotos("p1", "p2")
        vm.searchByStoredSelfie() // nothing cached yet: re-fetch, then match with the primary
        vm.awaitPhotos("f1", "f2")
        assertTrue(faceSearches().last().body.contains(""""selfieId":"s2""""))
        assertTrue(vm.isFiltered.value)

        vm.searchByStoredSelfie() // the library is cached now: match with all of it
        awaitUntil { faceSearches().size == 2 }
        vm.awaitPhotos("f1", "f2")
        assertTrue(faceSearches().last().body.contains(""""allSelfies":true"""))

        val before = pages().size
        vm.loadMorePhotos() // face results are one-shot, never paged
        Thread.sleep(100)
        assertEquals(before, pages().size)
    }

    @Test fun `face-search refusals read as targeted copy, falling back to the server's words`() {
        val vm = RunnerGalleryViewModel(application)
        vm.selectEvent(event("ev"))
        vm.awaitPhotos("p1", "p2")
        mapOf(
            "LOW_CONFIDENCE" to "We didn't find your face in this event yet. Try another shot.",
            "SELFIE_REJECTED" to "server: SELFIE_REJECTED",
            "NO_FACES" to "Selfie rejected — try a clearer, frontal shot.",
            "AI_API_UNAVAILABLE" to "Face search is offline right now. Try again in a few minutes.",
            "SELFIE_NOT_FOUND" to
                "This saved selfie's photo is missing. Remove it in your profile and add a new one, or take a selfie now.",
            "RATE_LIMITED" to "server: RATE_LIMITED",
        ).forEach { (code, copy) ->
            vm.searchBySelfieId(code)
            assertEquals(code, copy, vm.awaitError())
        }
    }

    @Test fun `a fresh selfie is saved then matched with the library, or matched alone when the save is refused or off`() {
        val vm = RunnerGalleryViewModel(application)
        vm.selectEvent(event("ev"))
        vm.awaitPhotos("p1", "p2")

        vm.searchBySelfieUri(picked("huge.jpg", MAX_UPLOAD_BYTES + 1))
        assertEquals("Couldn't read that photo. Use an image up to 8 MB and try again.", vm.awaitError())
        assertEquals(0, selfieUploads())
        assertTrue(faceSearches().isEmpty())

        vm.searchBySelfieUri(picked("a.jpg", 16))
        vm.awaitPhotos("f1", "f2")
        assertEquals("Saved to your selfie library.", vm.selfieSaveNotice.value)
        assertTrue(faceSearches().single().body.contains(""""allSelfies":true"""))

        selfieUpload = MockResponse().setResponseCode(422)
            .setBody("""{"success":false,"errors":[{"code":"SELFIE_REJECTED","message":"No face found"}]}""")
        vm.searchBySelfieUri(picked("b.jpg", 16))
        awaitUntil { faceSearches().size == 2 }
        vm.awaitPhotos("f1", "f2")
        assertEquals("Not saved to your library (No face found) — searching with it anyway.", vm.selfieSaveNotice.value)
        assertTrue(faceSearches().last().type!!.startsWith("multipart/"))

        val uploads = selfieUploads()
        vm.setSaveSelfieToLibrary(false)
        vm.searchBySelfieUri(picked("c.jpg", 16))
        awaitUntil { faceSearches().size == 3 }
        vm.awaitPhotos("f1", "f2")
        assertNull(vm.selfieSaveNotice.value)
        assertEquals(uploads, selfieUploads())
        assertTrue(faceSearches().last().type!!.startsWith("multipart/"))
    }

    @Test fun `a refused photo-alert switch snaps back with the reason, or asks for a selfie when it vanished`() {
        selfies = "[${selfie("s1", primary = true)}]"
        val vm = RunnerGalleryViewModel(application)
        vm.loadGalleryMetadata("ev")
        vm.photoAlert.awaitFirst { it == PhotoAlertUiState.Ready(registered = false) }

        alertToggle = MockResponse().setResponseCode(409).setBody(
            """{"success":false,"errors":[{"code":"EVENT_NOT_UPLOADABLE","message":"Photo alerts are closed for this event."}]}""",
        )
        vm.togglePhotoAlert("ev", register = true)
        assertEquals(
            PhotoAlertUiState.Ready(registered = false, message = "Photo alerts are closed for this event."),
            vm.photoAlert.awaitFirst { it is PhotoAlertUiState.Ready && !it.updating },
        )

        alertToggle = MockResponse().setResponseCode(422)
            .setBody("""{"success":false,"errors":[{"code":"SELFIE_REQUIRED","message":"Add a selfie first."}]}""")
        vm.togglePhotoAlert("ev", register = true)
        assertEquals(PhotoAlertUiState.NeedsSelfie(), vm.photoAlert.awaitFirst { it is PhotoAlertUiState.NeedsSelfie })
    }

    private fun selfie(id: String, primary: Boolean) =
        """{"id":"$id","dataUrl":"","uploadedAt":"","isPrimary":$primary,"qualityScore":0,"qualityTestStatus":"untested"}"""
}
