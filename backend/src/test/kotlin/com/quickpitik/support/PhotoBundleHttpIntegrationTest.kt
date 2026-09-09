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

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PhotoBundleHttpIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var tokens: PhotographerBundleTokenService
    @LocalServerPort var port: Int = 0
    @MockBean lateinit var storage: StorageService

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
