package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.security.JwtTokenProvider
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.jdbc.core.JdbcTemplate
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

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class PhotoMetadataAccessIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var orders: OrderRepository
    @Autowired private lateinit var items: OrderItemRepository
    @Autowired private lateinit var grants: DownloadGrantRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tokens: JwtTokenProvider
    @LocalServerPort private var port: Int = 0
    private val http = HttpClient.newHttpClient()
    private val json = ObjectMapper()

    @Test
    fun `owner reads retained processing hidden and trash but nobody else does`() {
        val owner = user(Role.PHOTOGRAPHER)
        val other = user(Role.PHOTOGRAPHER)
        val runner = user(Role.RUNNER)
        val event = event()
        val photo = photo(event, owner)
        val path = "/api/v1/me/photographer/photos/${photo.id}/metadata"
        metadata(photo, "private camera")

        for (status in listOf("PROCESSING", "HIDDEN", "LIVE")) {
            jdbc.update("UPDATE photos SET status = ?, deleted_at = ? WHERE id = ?", status, if (status == "LIVE") OffsetDateTime.now() else null, photo.id)
            val response = get(path, owner)
            assertEquals(200, response.statusCode(), status)
            assertEquals("private camera", data(response).path("fields").path("cameraMake").asText())
            assertNoStore(response)
        }
        for (caller in listOf(other, runner)) {
            val response = get(path, caller)
            assertEquals(if (caller.role == Role.RUNNER) 403 else 404, response.statusCode())
            assertFalse(response.body().contains("private camera"))
            assertNoStore(response)
        }
        assertEquals(401, get(path).statusCode())
        jdbc.update("UPDATE photos SET deletion_started_at = now() WHERE id = ?", photo.id)
        assertEquals(404, get(path, owner).statusCode())
        assertEquals(404, get("/api/v1/me/photographer/photos/${UUID.randomUUID()}/metadata", owner).statusCode())
    }

    @Test
    fun `public read requires this photographers current sharing and every public gate`() {
        val a = user(Role.PHOTOGRAPHER)
        val b = user(Role.PHOTOGRAPHER)
        val event = event()
        val public = photo(event, a)
        val private = photo(event, b)
        metadata(public, "shared camera")
        metadata(private, "private camera")
        approve(a)
        approve(b)
        coverage(event, a, true)
        coverage(event, b, false)
        val publicPath = "/api/v1/public/photos/${public.id}/metadata"
        val privatePath = "/api/v1/public/photos/${private.id}/metadata"

        assertEquals(200, get(publicPath).statusCode())
        assertEquals("shared camera", data(get(publicPath)).path("fields").path("cameraMake").asText())
        denied(privatePath)
        // Even a signed-in buyer with an active original-download grant receives the public policy.
        val buyer = user(Role.RUNNER)
        val order = orders.save(
            Order(
                userId = buyer.id,
                eventId = event.id,
                recipientEmail = buyer.email,
                paymentMethodWire = "gcash",
                status = OrderStatus.FULFILLED,
                totalPhp = private.pricePhp,
            ),
        )
        items.save(OrderItem(OrderItemId(order.id, private.id), private.pricePhp))
        grants.save(DownloadGrant(DownloadGrantId(order.id, private.id), OffsetDateTime.now().plusDays(1)))
        denied(privatePath, buyer)

        jdbc.update("UPDATE event_photographer SET metadata_public = false WHERE event_id = ? AND photographer_id = ?", event.id, a.id)
        denied(publicPath)
        jdbc.update("UPDATE event_photographer SET metadata_public = true WHERE event_id = ? AND photographer_id = ?", event.id, a.id)
        jdbc.update("UPDATE photos SET deleted_at = now(), deletion_started_at = now() WHERE id = ?", public.id)
        denied(publicPath)
        jdbc.update("UPDATE photos SET deleted_at = null, deletion_started_at = null WHERE id = ?", public.id)
        for (status in listOf("PROCESSING", "HIDDEN")) {
            jdbc.update("UPDATE photos SET status = ? WHERE id = ?", status, public.id)
            denied(publicPath)
        }
        jdbc.update("UPDATE photos SET status = 'LIVE', deleted_at = now() WHERE id = ?", public.id)
        denied(publicPath)
        jdbc.update("UPDATE photos SET deleted_at = null WHERE id = ?", public.id)
        jdbc.update("UPDATE events SET status = 'DRAFT' WHERE id = ?", event.id)
        denied(publicPath)
        jdbc.update("UPDATE events SET status = 'ARCHIVED', visibility = 'UNLISTED' WHERE id = ?", event.id)
        assertEquals(200, get(publicPath).statusCode())
        jdbc.update("UPDATE events SET deleted_at = now() WHERE id = ?", event.id)
        denied(publicPath)
        jdbc.update("UPDATE events SET deleted_at = null WHERE id = ?", event.id)
        jdbc.update("UPDATE users SET suspended_at = now() WHERE id = ?", a.id)
        denied(publicPath)
        jdbc.update("UPDATE users SET suspended_at = null WHERE id = ?", a.id)
        jdbc.update("UPDATE photographer_settings SET verification_status = 'INCOMPLETE' WHERE user_id = ?", a.id)
        denied(publicPath)
        jdbc.update("DELETE FROM photographer_settings WHERE user_id = ?", a.id)
        denied(publicPath)
        jdbc.update("DELETE FROM event_photographer WHERE event_id = ? AND photographer_id = ?", event.id, a.id)
        denied(publicPath)

        val ownerless = photos.save(
            Photo(eventId = event.id, photographerId = null, s3Key = "m4/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE),
        )
        metadata(ownerless, "ownerless camera")
        denied("/api/v1/public/photos/${ownerless.id}/metadata")
    }

    @Test
    fun `galleries expose only access hints and owner requires the actual requester`() {
        val owner = user(Role.PHOTOGRAPHER)
        val runner = user(Role.RUNNER)
        val event = event()
        val photo = photo(event, owner)
        metadata(photo, "never in gallery")
        approve(owner)
        coverage(event, owner, false)
        val gallery = "/api/v1/events/${event.slug}/photos?snapshotAt=2027-01-01T00:00:00Z"
        val profile = "/api/v1/public/photographers/${handle(owner)}/events/${event.slug}/photos"

        assertHint(gallery, null, "none")
        assertHint(gallery, owner, "owner")
        assertHint(gallery, runner, "none")
        assertHint(profile, null, "none")
        assertHint(profile, owner, "owner")
        assertHint(profile, runner, "none")
        jdbc.update("UPDATE event_photographer SET metadata_public = true WHERE event_id = ? AND photographer_id = ?", event.id, owner.id)
        assertHint(gallery, null, "public")
        assertHint(profile, null, "public")
        assertHint(gallery, owner, "owner")
    }

    private fun assertHint(path: String, caller: User?, expected: String) {
        val response = get(path, caller)
        assertEquals(200, response.statusCode(), path)
        val item = data(response).path("items").first()
        assertEquals(expected, item.path("metadataAccess").asText(), path)
        assertFalse(response.body().contains("never in gallery"), path)
        if (caller != null) assertNoStore(response)
    }

    private fun denied(path: String, caller: User? = null) {
        val response = get(path, caller)
        assertEquals(404, response.statusCode(), path)
        assertFalse(response.body().contains("camera"), path)
        assertNoStore(response)
    }

    private fun user(role: Role) = users.save(User(email = "m4-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "M4", role = role))
    private fun event() = events.save(Event(slug = "m4-${UUID.randomUUID()}", name = "M4", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
    private fun photo(event: Event, owner: User) = photos.save(Photo(eventId = event.id, photographerId = owner.id, s3Key = "m4/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE))
    private fun metadata(photo: Photo, camera: String) {
        jdbc.update("INSERT INTO photo_metadata (photo_id, state, extractor_version, fields) VALUES (?, 'READY', 1, ?::jsonb)", photo.id, "{\"cameraMake\":\"$camera\"}")
    }
    private fun approve(user: User) {
        jdbc.update("INSERT INTO photographer_settings (user_id, handle, verification_status) VALUES (?, ?, 'APPROVED')", user.id, handle(user))
    }
    private fun handle(user: User) = "m4-${user.id.toString().take(12)}"
    private fun coverage(event: Event, user: User, shared: Boolean) {
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, metadata_public) VALUES (?, ?, ?)", event.id, user.id, shared)
    }
    private fun get(path: String, caller: User? = null): HttpResponse<String> {
        val builder = HttpRequest.newBuilder(URI("http://localhost:$port$path"))
        caller?.let { builder.header("Authorization", "Bearer ${tokens.createAccessToken(it)}") }
        return http.send(builder.GET().build(), HttpResponse.BodyHandlers.ofString())
    }
    private fun data(response: HttpResponse<String>) = json.readTree(response.body()).path("data")
    private fun assertNoStore(response: HttpResponse<String>) {
        assertTrue("no-store" in response.headers().firstValue("Cache-Control").orElse(""))
    }
}
