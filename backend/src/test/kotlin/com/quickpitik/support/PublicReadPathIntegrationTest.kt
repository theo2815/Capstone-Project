package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.mock.mockito.MockBean
import org.springframework.boot.test.web.server.LocalServerPort
import java.math.BigDecimal
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// RD-7: three public side doors the runner-deletion audit found. A deleted
// free event's photos still downloaded; a photographer's profile listed their
// DRAFT and Private events; the photographer gallery opened a DRAFT event.
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PublicReadPathIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var settings: PhotographerSettingsRepository
    @Autowired lateinit var coverage: EventPhotographerRepository
    @MockBean lateinit var storage: StorageService
    @LocalServerPort var port: Int = 0

    private val client = HttpClient.newHttpClient()
    private val json = ObjectMapper()

    private class Seeded(val handle: String, val ownerId: UUID)

    private fun photographer(): Seeded {
        val owner = users.save(User(email = "rd7-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Shooter", role = Role.PHOTOGRAPHER))
        val handle = "rd7-${UUID.randomUUID().toString().take(8)}"
        settings.save(PhotographerSettings(userId = owner.id, handle = handle, verificationStatus = VerificationStatus.APPROVED))
        return Seeded(handle, owner.id)
    }

    private fun event(
        status: EventStatus = EventStatus.ACTIVE,
        visibility: EventVisibility = EventVisibility.PUBLIC,
        pricing: EventPricingMode = EventPricingMode.PAID,
        deleted: Boolean = false,
    ) = events.save(
        Event(
            slug = "rd7-${UUID.randomUUID()}", name = "RD7 Race", date = LocalDate.now(), location = "Cebu", status = status,
            visibility = visibility, pricingMode = pricing,
            deletedAt = if (deleted) OffsetDateTime.now() else null,
        ),
    )

    private fun cover(owner: Seeded, event: Event): Photo {
        coverage.save(EventPhotographer(id = EventPhotographerId(event.id, owner.ownerId), photoCount = 1, lastUploadAt = OffsetDateTime.now()))
        return photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.ownerId, s3Key = "rd7/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("125")))
    }

    private fun get(path: String): HttpResponse<String> =
        client.send(HttpRequest.newBuilder(URI("http://localhost:$port$path")).GET().build(), HttpResponse.BodyHandlers.ofString())

    @Test
    fun `free download of a deleted free event is 404`() {
        val owner = photographer()
        val event = event(pricing = EventPricingMode.FREE)
        val photo = cover(owner, event)
        Mockito.`when`(storage.open(Mockito.anyString())).thenAnswer { java.io.ByteArrayInputStream(ByteArray(4)) }
        // Control: while the event is live the route reaches storage, so the 404 below is the deletion.
        assertTrue(photos.isFreeDeliveryAvailable(event.id, photo.id))
        assertEquals(200, get("/api/v1/events/${event.id}/photos/${photo.id}/download").statusCode())

        events.saveAndFlush(event.also { it.deletedAt = OffsetDateTime.now() })

        assertFalse(photos.isFreeDeliveryAvailable(event.id, photo.id))
        assertEquals(404, get("/api/v1/events/${event.id}/photos/${photo.id}/download").statusCode())
    }

    @Test
    fun `the public profile lists neither DRAFT nor Private events`() {
        val owner = photographer()
        val live = event()
        val draft = event(status = EventStatus.DRAFT)
        val privateEvent = event(visibility = EventVisibility.UNLISTED)
        listOf(live, draft, privateEvent).forEach { cover(owner, it) }

        val response = get("/api/v1/public/photographers/${owner.handle}")

        assertEquals(200, response.statusCode())
        val slugs = json.readTree(response.body()).path("data").path("events").map { it.path("eventSlug").asText() }
        assertEquals(listOf(live.slug), slugs)
    }

    @Test
    fun `the photographer gallery of a DRAFT event is 404`() {
        val owner = photographer()
        val draft = event(status = EventStatus.DRAFT)
        cover(owner, draft)
        val live = event()
        cover(owner, live)

        assertEquals(200, get("/api/v1/public/photographers/${owner.handle}/events/${live.slug}/photos").statusCode())
        assertEquals(404, get("/api/v1/public/photographers/${owner.handle}/events/${draft.slug}/photos").statusCode())
    }

    @Test
    fun `an anonymous gallery JSON response carries Cache-Control no-store`() {
        val owner = photographer()
        val live = event()
        cover(owner, live)

        val response = get("/api/v1/public/photographers/${owner.handle}/events/${live.slug}/photos")

        assertEquals(200, response.statusCode())
        assertTrue("no-store" in response.headers().firstValue("Cache-Control").orElse(""))
    }
}
