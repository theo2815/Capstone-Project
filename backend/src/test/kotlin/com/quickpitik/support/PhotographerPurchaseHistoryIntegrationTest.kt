package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.JwtTokenProvider
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
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

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PhotographerPurchaseHistoryIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var orders: OrderRepository
    @Autowired private lateinit var items: OrderItemRepository
    @Autowired private lateinit var tokens: JwtTokenProvider
    @LocalServerPort private var port: Int = 0
    private val client = HttpClient.newHttpClient()
    private val json = ObjectMapper()

    @Test
    fun `removed event is visible only through an authorized history read`() {
        val owner = seller()
        val foreign = seller()
        val legacySeller = seller()
        val removedAt = OffsetDateTime.parse("2026-09-04T21:36:27Z")
        val event = event(EventStatus.ARCHIVED, owner.id, removedAt)
        coverage.save(EventPhotographer(EventPhotographerId(event.id, owner.id), photoCount = 1))
        photos.save(Photo(eventId = event.id, photographerId = owner.id, s3Key = "history/owner.jpg", pricePhp = BigDecimal("100.00")))
        val legacyPhoto = photos.save(Photo(eventId = event.id, photographerId = legacySeller.id, s3Key = "history/legacy.jpg", pricePhp = BigDecimal("100.00")))
        val order = orders.save(Order(eventId = event.id, recipientEmail = "buyer@test.local", paymentMethodWire = "card", status = OrderStatus.FULFILLED, totalPhp = BigDecimal("100.00")))
        items.save(OrderItem(OrderItemId(order.id, legacyPhoto.id), BigDecimal("100.00")))

        val path = "/api/v1/me/photographer/events/${event.id}"
        val ownerToken = tokens.createAccessToken(owner)
        val foreignToken = tokens.createAccessToken(foreign)
        val legacyToken = tokens.createAccessToken(legacySeller)
        val history = get("$path?includeRemoved=true", ownerToken)
        assertEquals(200, history.statusCode())
        assertEquals(removedAt.toString(), json.readTree(history.body()).path("data").path("deletedAt").asText())
        assertEquals(404, get(path, ownerToken).statusCode())
        assertEquals(404, get("$path?includeRemoved=true", foreignToken).statusCode())
        assertEquals(200, get("$path?includeRemoved=true", legacyToken).statusCode())
        assertEquals(404, get(path, legacyToken).statusCode())
        assertEquals(404, get("/api/v1/me/photographer/events/${UUID.randomUUID()}?includeRemoved=true", ownerToken).statusCode())
        assertEquals(404, get("/api/v1/events/${event.slug}").statusCode())
        assertEquals(404, get("$path/photos", ownerToken).statusCode())
        assertEquals(404, get("$path/photos?deleted=true", ownerToken).statusCode())
    }

    @Test
    fun `removed event rejects photo mutations while active and archived details remain readable`() {
        val owner = seller()
        val token = tokens.createAccessToken(owner)
        for (status in listOf(EventStatus.ACTIVE, EventStatus.ARCHIVED)) {
            val available = event(status, owner.id)
            assertEquals(200, get("/api/v1/me/photographer/events/${available.id}", token).statusCode())
        }
        val removed = event(EventStatus.ACTIVE, owner.id, OffsetDateTime.parse("2026-09-04T21:36:27Z"))
        val photo = photos.save(Photo(eventId = removed.id, photographerId = owner.id, s3Key = "history/mutation.jpg", pricePhp = BigDecimal("100.00")))
        val body = """{"photoIds":["${photo.id}"]}"""
        val base = "/api/v1/me/photographer/events/${removed.id}/photos"
        for (action in listOf("trash", "restore", "permanent-delete")) {
            assertEquals(404, post("$base/$action", body, token).statusCode(), action)
        }
    }

    private fun seller() = users.save(User(email = "history-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Seller", role = Role.PHOTOGRAPHER))

    private fun event(status: EventStatus, owner: UUID, deletedAt: OffsetDateTime? = null) = events.save(
        Event(slug = "history-${UUID.randomUUID()}", name = "History", date = LocalDate.of(2026, 9, 1), location = "Cebu", status = status, createdBy = owner, deletedAt = deletedAt),
    )

    private fun get(path: String, token: String? = null): HttpResponse<String> = send(HttpRequest.newBuilder(uri(path)).GET(), token)

    private fun post(path: String, body: String, token: String): HttpResponse<String> = send(
        HttpRequest.newBuilder(uri(path)).header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)), token,
    )

    private fun send(builder: HttpRequest.Builder, token: String?): HttpResponse<String> {
        token?.let { builder.header("Authorization", "Bearer $it") }
        return client.send(builder.build(), HttpResponse.BodyHandlers.ofString())
    }

    private fun uri(path: String) = URI("http://localhost:$port$path")
}
