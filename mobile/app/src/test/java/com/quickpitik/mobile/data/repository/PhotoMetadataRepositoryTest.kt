package com.quickpitik.mobile.data.repository

import androidx.test.core.app.ApplicationProvider
import com.google.gson.Gson
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.ok
import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import retrofit2.HttpException

@RunWith(RobolectricTestRunner::class)
class PhotoMetadataRepositoryTest {
    @get:Rule val backend = MockBackend()
    private val session = SessionManager.getInstance(ApplicationProvider.getApplicationContext())
    private val repo = PhotoMetadataRepository(session)

    @After fun signOut() = session.clearSession()

    @Test fun `settings and event visibility use exact authenticated routes and bodies`() = runBlocking {
        session.saveSession("t", "PHOTOGRAPHER", "Juan", "juan@x.com")
        backend.enqueue(
            ok("""{"metadataPublicByDefault":false}"""),
            ok("""{"metadataPublicByDefault":true}"""),
            ok("""{"metadataPublic":false}"""),
            ok("""{"metadataPublic":true}"""),
        )
        assertEquals(false, repo.getMetadataSettings().metadataPublicByDefault)
        assertEquals(true, repo.setMetadataSettings(true).metadataPublicByDefault)
        assertEquals(false, repo.getEventVisibility("event-1").metadataPublic)
        assertEquals(true, repo.setEventVisibility("event-1", true).metadataPublic)

        val paths = listOf(
            "/api/v1/me/photographer/metadata-settings",
            "/api/v1/me/photographer/metadata-settings",
            "/api/v1/me/photographer/events/event-1/metadata-visibility",
            "/api/v1/me/photographer/events/event-1/metadata-visibility",
        )
        paths.forEachIndexed { i, path ->
            val request = backend.server.takeRequest()
            assertEquals(path, request.path)
            assertEquals(if (i % 2 == 0) "GET" else "PUT", request.method)
            assertEquals("Bearer t", request.getHeader("Authorization"))
            if (i == 1) assertEquals("""{"metadataPublicByDefault":true}""", request.body.readUtf8())
            if (i == 3) assertEquals("""{"metadataPublic":true}""", request.body.readUtf8())
        }
    }

    @Test fun `owner and public metadata use separate routes with nullable fields`() = runBlocking {
        session.saveSession("t", "PHOTOGRAPHER", "Juan", "juan@x.com")
        backend.enqueue(ok(METADATA), ok(METADATA))
        val owned = repo.getPhotoMetadata("photo-1", owner = true)
        assertEquals("photo-1", owned.photoId)
        assertEquals("ready", owned.status)
        assertNull(owned.originalFilename)
        assertNull(owned.fields?.lens)
        repo.getPhotoMetadata("photo-1", owner = false)
        assertEquals("/api/v1/me/photographer/photos/photo-1/metadata", backend.server.takeRequest().path)
        assertEquals("/api/v1/public/photos/photo-1/metadata", backend.server.takeRequest().path)
    }

    @Test fun `public metadata works without a session and reports 404`() = runBlocking {
        session.clearSession()
        backend.enqueue(ok(METADATA), MockResponse().setResponseCode(404))
        assertEquals("ready", repo.getPhotoMetadata("photo-1", owner = false).status)
        assertNull(backend.server.takeRequest().getHeader("Authorization"))
        assertThrows(HttpException::class.java) { runBlocking { repo.getPhotoMetadata("missing", owner = false) } }
        Unit
    }

    @Test fun `owner routes reject missing session before network`() {
        session.clearSession()
        assertThrows(IllegalStateException::class.java) { runBlocking { repo.getMetadataSettings() } }
        assertEquals(0, backend.server.requestCount)
    }

    @Test fun `owner metadata reports forbidden without exposing another photographer's data`() = runBlocking {
        session.saveSession("t", "PHOTOGRAPHER", "Juan", "juan@x.com")
        backend.enqueue(MockResponse().setResponseCode(403))
        assertThrows(HttpException::class.java) { runBlocking { repo.getPhotoMetadata("other", owner = true) } }
        Unit
    }

    @Test fun `public photographer photos accept an optional bearer`() = runBlocking {
        session.saveSession("t", "RUNNER", "Juan", "juan@x.com")
        backend.enqueue(ok("""{"items":[],"total":0,"offset":0,"limit":100}"""),
            ok("""{"items":[],"total":0,"offset":0,"limit":100}"""))
        val api = RetrofitClient.apiService
        api.getPublicPhotographerEventPhotos("handle", "race")
        api.getPublicPhotographerEventPhotos("handle", "race", token = "Bearer t")
        val guest = backend.server.takeRequest()
        val runner = backend.server.takeRequest()
        assertEquals("/api/v1/public/photographers/handle/events/race/photos?offset=0&limit=100", guest.path)
        assertNull(guest.getHeader("Authorization"))
        assertEquals("Bearer t", runner.getHeader("Authorization"))
    }

    @Test fun `public metadata propagates an unauthorized response`() = runBlocking {
        session.clearSession()
        backend.enqueue(MockResponse().setResponseCode(401))
        val error = assertThrows(HttpException::class.java) {
            runBlocking { repo.getPhotoMetadata("private", owner = false) }
        }
        assertEquals(401, error.code())
    }

    @Test fun `older photo JSON without access field remains readable`() {
        val photo = Gson().fromJson(
            """{"id":"photo-1","time":"10:00","tone":0,"span":"portrait","price":0}""",
            PhotoDto::class.java,
        )
        assertNull(photo.metadataAccess)
    }

    private companion object {
        const val METADATA = """{"photoId":"photo-1","status":"ready","originalFilename":null,"fileSizeBytes":null,"fields":{"cameraMake":"Canon","cameraModel":"R6","lens":null,"aperture":2.8,"exposureTime":"1/500","iso":400,"author":null,"copyright":null,"takenAtLocal":null,"takenAtOffset":null,"flash":null,"exposureMode":null,"whiteBalance":null}}"""
    }
}
