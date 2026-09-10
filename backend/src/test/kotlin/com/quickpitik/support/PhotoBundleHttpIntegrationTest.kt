package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotographerBundleTokenService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.mock.mockito.MockBean
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.jdbc.core.JdbcTemplate
import com.quickpitik.common.PaginationParams
import com.quickpitik.service.photographer.PhotographerEventService
import java.io.ByteArrayInputStream
import java.io.IOException
import java.math.BigDecimal
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.time.LocalDate
import java.util.UUID
import java.util.zip.ZipInputStream
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PhotoBundleHttpIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var tokens: PhotographerBundleTokenService
    @LocalServerPort var port: Int = 0
    @MockBean lateinit var storage: StorageService
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var photographerEvents: PhotographerEventService
    private fun <T> anyArg(): T = Mockito.any()

    @Test fun `all 3028 originals download despite a 60 photo page and selected POST stays scoped`() {
        val owner = users.save(User(email = "large-bundle-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Bundle", role = Role.PHOTOGRAPHER))
        val foreign = users.save(User(email = "foreign-bundle-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Other", role = Role.PHOTOGRAPHER))
        val event = events.save(Event(slug = "large-bundle-${UUID.randomUUID()}", name = "Large Bundle", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        jdbc.update("""INSERT INTO photos (event_id, photographer_id, s3_key, price_php)
            SELECT ?, ?, 'large-bundle/' || n || '.jpg', 125 FROM generate_series(1, 3028) n""", event.id, owner.id)
        val foreignPhoto = photos.saveAndFlush(Photo(eventId = event.id, photographerId = foreign.id, s3Key = "foreign.jpg", pricePhp = BigDecimal("125")))
        photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "trashed.jpg", pricePhp = BigDecimal("125"), deletedAt = java.time.OffsetDateTime.now()))
        val page = photographerEvents.listPhotos(owner.id, event.id, null, pagination = PaginationParams(offset = 0, limit = 60))
        assertEquals(60, page.items.size)
        assertEquals(3028L, page.total)

        Mockito.`when`(storage.open(Mockito.anyString())).thenAnswer { call ->
            ByteArrayInputStream(call.getArgument<String>(0).toByteArray())
        }
        val grant = tokens.issue(owner.id, event.id)
        val uri = URI("http://localhost:$port/api/v1/photographer-bundles/${event.id}?token=${grant.token}")
        val client = HttpClient.newHttpClient()
        fun download(ids: List<UUID>? = null): HttpResponse<ByteArray> {
            val request = HttpRequest.newBuilder(uri).timeout(Duration.ofSeconds(60))
            if (ids == null) request.GET() else request
                .header("Content-Type", "application/x-www-form-urlencoded")
                .POST(HttpRequest.BodyPublishers.ofString("photoIds=" + ids.joinToString(",")))
            return client.send(request.build(), HttpResponse.BodyHandlers.ofByteArray())
        }
        fun originals(response: HttpResponse<ByteArray>): Set<String> {
            assertEquals(200, response.statusCode())
            assertTrue(response.headers().firstValue("Content-Disposition").orElse("").startsWith("attachment;"))
            val contents = mutableSetOf<String>()
            ZipInputStream(ByteArrayInputStream(response.body())).use { zip ->
                while (zip.nextEntry != null) contents += zip.readAllBytes().toString(Charsets.UTF_8)
            }
            return contents
        }
        assertEquals((1..3028).map { "large-bundle/$it.jpg" }.toSet(), originals(download()))
        val selected = page.items.take(2).map { it.id }
        assertEquals(photos.findAllById(selected).map { it.s3Key }.toSet(), originals(download(selected)))
        // Even the entire event can be submitted as a selection without URL
        // length/parameter-count limits; the form has a single field.
        val allIds = photos.findActiveBundleRows(event.id, owner.id).map { it.id }
        assertEquals(3028, originals(download(allIds)).size)
        assertEquals(404, download(selected + foreignPhoto.id).statusCode())
        assertEquals(404, download(emptyList()).statusCode())
        val noToken = HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/photographer-bundles/${event.id}"))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .POST(HttpRequest.BodyPublishers.ofString("photoIds=" + selected.joinToString(","))).build()
        assertEquals(404, client.send(noToken, HttpResponse.BodyHandlers.ofString()).statusCode())

        // The single-photo endpoint must use the attachment signer, with the
        // original key, and keep its existing ownership/deletion checks.
        val photo = photos.findById(selected.first()).orElseThrow()
        Mockito.`when`(storage.presignedDownloadUrl(anyArg(), anyArg(), anyArg()))
            .thenReturn("https://storage.test/attachment")
        assertEquals("https://storage.test/attachment", photographerEvents.getDownload(owner.id, photo.id).url)
        Mockito.verify(storage).presignedDownloadUrl(photo.s3Key, Duration.ofMinutes(5), "quickpitik-${photo.id}.jpg")
        kotlin.test.assertFailsWith<com.quickpitik.exception.NotFoundException> { photographerEvents.getDownload(foreign.id, photo.id) }
    }

    @Test fun `storage failure aborts an already committed HTTP download`() {
        val owner = users.save(User(email = "bundle-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Bundle", role = Role.PHOTOGRAPHER))
        val event = events.save(Event(slug = "bundle-${UUID.randomUUID()}", name = "Bundle", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        repeat(2) { photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "failed-bundle/$it.jpg", pricePhp = BigDecimal("125"))) }
        // Incompressible bytes commit the response before the second read fails.
        val bytes = ByteArray(64000).also { java.util.Random(1).nextBytes(it) }
        var reads = 0
        Mockito.`when`(storage.open(Mockito.anyString())).thenAnswer {
            if (++reads == 2) throw IOException("storage unavailable")
            ByteArrayInputStream(bytes)
        }
        val grant = tokens.issue(owner.id, event.id)
        val request = HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/photographer-bundles/${event.id}?token=${grant.token}"))
            .timeout(Duration.ofSeconds(15)).GET().build()
        val error = assertFailsWith<IOException> {
            HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofByteArray())
        }
        assertFalse(error is java.net.http.HttpTimeoutException, "The server must abort the response, not leave it hanging")
    }

    @Test fun `an event ZIP that takes longer than thirty seconds must finish`() {
        val owner = users.save(User(email = "bundle-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Bundle", role = Role.PHOTOGRAPHER))
        val event = events.save(Event(slug = "bundle-${UUID.randomUUID()}", name = "Bundle", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        repeat(2) { photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "bundle/$it.jpg", pricePhp = BigDecimal("125"))) }
        val randomBytes = ByteArray(64000).also { java.util.Random(1).nextBytes(it) }
        var reads = 0
        Mockito.`when`(storage.open(Mockito.anyString())).thenAnswer {
            if (++reads == 2) Thread.sleep(35000)
            ByteArrayInputStream(randomBytes)
        }
        val grant = tokens.issue(owner.id, event.id)
        val request = HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/photographer-bundles/${event.id}?token=${grant.token}"))
            .timeout(Duration.ofSeconds(60)).GET().build()
        val result = HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.ofByteArray())
        assertEquals(200, result.statusCode())
        var entries = 0
        ZipInputStream(ByteArrayInputStream(result.body())).use { zip ->
            while (zip.nextEntry != null) { zip.readAllBytes(); entries++ }
        }
        assertEquals(2, entries, "A successful Download all response must contain both originals")
    }
}
