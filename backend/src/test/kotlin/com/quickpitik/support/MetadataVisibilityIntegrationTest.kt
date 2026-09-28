package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoMetadata
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.dto.photos.toDto
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.dto.photographer.CreateMyEventRequest
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotoMetadataRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotoMetadataSettingsService
import com.quickpitik.service.photographer.PhotographerOwnedEventService
import com.quickpitik.security.JwtTokenProvider
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.LocalDateTime
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class MetadataVisibilityIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var photoMetadata: PhotoMetadataRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var transactions: TransactionTemplate
    @Autowired private lateinit var metadataSettings: PhotoMetadataSettingsService
    @Autowired private lateinit var ownedEvents: PhotographerOwnedEventService
    @Autowired private lateinit var tokens: JwtTokenProvider
    @LocalServerPort private var port: Int = 0
    private val http = HttpClient.newHttpClient()
    private val json = ObjectMapper()

    @Test
    fun `metadata settings reject non boolean JSON without changing private choices`() {
        val owner = photographer()
        val event = event()
        transactions.executeWithoutResult { coverage.upsertOnUpload(event.id, owner.id, OffsetDateTime.now()) }
        val defaultsPath = "/api/v1/me/photographer/metadata-settings"
        val eventPath = "/api/v1/me/photographer/events/${event.id}/metadata-visibility"

        assertEquals(400, request("PUT", defaultsPath, owner, """{"metadataPublicByDefault":1}""").statusCode())
        assertEquals(false, metadataSettings.getDefault(owner.id).metadataPublicByDefault)
        assertEquals(400, request("PUT", eventPath, owner, """{"metadataPublic":"true"}""").statusCode())
        assertEquals(false, metadataSettings.getEvent(owner.id, event.id).metadataPublic)
        assertEquals(200, request("PUT", defaultsPath, owner, """{"metadataPublicByDefault":false}""").statusCode())
        assertEquals(200, request("PUT", eventPath, owner, """{"metadataPublic":false}""").statusCode())
    }

    @Test
    fun `settings routes return typed envelopes and refuse foreign or non photographer callers`() {
        val a = photographer()
        val b = photographer()
        val runner = users.save(User(email = "runner-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Runner", role = Role.RUNNER))
        val event = event()
        transactions.executeWithoutResult { coverage.upsertOnUpload(event.id, a.id, OffsetDateTime.now()) }

        val defaults = request("GET", "/api/v1/me/photographer/metadata-settings", a)
        assertEquals(200, defaults.statusCode())
        assertEquals(false, json.readTree(defaults.body()).path("data").path("metadataPublicByDefault").booleanValue())
        val updated = request("PUT", "/api/v1/me/photographer/metadata-settings", a, """{"metadataPublicByDefault":true}""")
        assertEquals(200, updated.statusCode())
        assertEquals(true, json.readTree(updated.body()).path("data").path("metadataPublicByDefault").booleanValue())
        assertEquals(400, request("PUT", "/api/v1/me/photographer/metadata-settings", a, "{}").statusCode())
        assertEquals(400, request("PUT", "/api/v1/me/photographer/metadata-settings", a, """{"metadataPublicByDefault":null}""").statusCode())

        val path = "/api/v1/me/photographer/events/${event.id}/metadata-visibility"
        assertEquals(404, request("GET", path, b).statusCode())
        assertEquals(404, request("PUT", path, b, """{"metadataPublic":true}""").statusCode())
        assertEquals(403, request("PUT", path, runner, """{"metadataPublic":true}""").statusCode())
        val visible = request("PUT", path, a, """{"metadataPublic":true}""")
        assertEquals(200, visible.statusCode())
        assertEquals(true, json.readTree(visible.body()).path("data").path("metadataPublic").booleanValue())
        assertEquals(400, request("PUT", path, a, "{}").statusCode())
        assertEquals(400, request("PUT", path, a, """{"metadataPublic":null}""").statusCode())
        assertEquals(true, json.readTree(request("GET", path, a).body()).path("data").path("metadataPublic").booleanValue())
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM event_photographer WHERE event_id = ? AND photographer_id = ?", Int::class.java, event.id, b.id))
    }

    @Test
    fun `default changes affect new owned events but leave earlier choices and another photographer alone`() {
        val a = photographer()
        val b = photographer()
        jdbc.update("INSERT INTO photographer_settings (user_id, verification_status) VALUES (?, 'APPROVED')", a.id)

        assertEquals(false, metadataSettings.getDefault(a.id).metadataPublicByDefault)
        val e1 = ownedEvents.create(a.id, ownedRequest("First"), null)
        assertEquals(false, metadataSettings.getEvent(a.id, e1.id).metadataPublic)

        assertEquals(true, metadataSettings.setDefault(a.id, true).metadataPublicByDefault)
        assertEquals(false, metadataSettings.getEvent(a.id, e1.id).metadataPublic)
        val e2 = ownedEvents.create(a.id, ownedRequest("Second"), null)
        assertEquals(true, metadataSettings.getEvent(a.id, e2.id).metadataPublic)
        assertEquals(false, metadataSettings.setEvent(a.id, e2.id, false).metadataPublic)
        assertEquals(false, metadataSettings.getDefault(b.id).metadataPublicByDefault)
        assertEquals(true, metadataSettings.getDefault(a.id).metadataPublicByDefault)
    }

    @Test
    fun `event preference is limited to existing coverage or a legacy owned event`() {
        val a = photographer()
        val b = photographer()
        val foreign = event(createdBy = b.id)
        val legacyOwn = event(createdBy = a.id)

        assertFailsWith<NotFoundException> { metadataSettings.getEvent(a.id, foreign.id) }
        assertFailsWith<NotFoundException> { metadataSettings.setEvent(a.id, foreign.id, true) }
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM event_photographer WHERE event_id = ? AND photographer_id = ?", Int::class.java, foreign.id, a.id))

        metadataSettings.setDefault(a.id, true)
        assertEquals(false, metadataSettings.getEvent(a.id, legacyOwn.id).metadataPublic)
        assertEquals(false, metadataSettings.getEvent(a.id, legacyOwn.id).metadataPublic)
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM event_photographer WHERE event_id = ? AND photographer_id = ?", Int::class.java, legacyOwn.id, a.id))
        assertEquals(true, metadataSettings.setEvent(a.id, legacyOwn.id, true).metadataPublic)
        assertEquals(true, metadataSettings.getEvent(a.id, legacyOwn.id).metadataPublic)
    }

    @Test
    fun `old coverage stays private while a later upload takes the saved default only on insert`() {
        val photographer = photographer()
        val oldEvent = event()
        val newEvent = event()
        val oldKey = EventPhotographerId(oldEvent.id, photographer.id)
        val newKey = EventPhotographerId(newEvent.id, photographer.id)

        transactions.executeWithoutResult { coverage.upsertOnUpload(oldEvent.id, photographer.id, OffsetDateTime.now()) }
        assertEquals(false, visibility(oldKey))

        jdbc.update("INSERT INTO photographer_settings (user_id, metadata_public_by_default) VALUES (?, true)", photographer.id)
        transactions.executeWithoutResult { coverage.upsertOnUpload(newEvent.id, photographer.id, OffsetDateTime.now()) }
        assertEquals(false, visibility(oldKey))
        assertEquals(true, visibility(newKey))

        jdbc.update("UPDATE event_photographer SET metadata_public = false WHERE event_id = ? AND photographer_id = ?", newEvent.id, photographer.id)
        transactions.executeWithoutResult { coverage.upsertOnUpload(newEvent.id, photographer.id, OffsetDateTime.now()) }
        assertEquals(false, visibility(newKey))
        assertEquals(2, coverage.findById(newKey).orElseThrow().photoCount)
    }

    @Test
    fun `first upload keeps legacy owned event private after the default becomes public`() {
        val owner = photographer()
        val legacyOwned = event(createdBy = owner.id)
        val shared = event()
        val ownedKey = EventPhotographerId(legacyOwned.id, owner.id)
        val sharedKey = EventPhotographerId(shared.id, owner.id)

        metadataSettings.setDefault(owner.id, true)
        assertEquals(false, metadataSettings.getEvent(owner.id, legacyOwned.id).metadataPublic)
        transactions.executeWithoutResult { coverage.upsertOnUpload(legacyOwned.id, owner.id, OffsetDateTime.now()) }
        assertEquals(false, visibility(ownedKey))
        assertEquals(false, metadataSettings.getEvent(owner.id, legacyOwned.id).metadataPublic)

        transactions.executeWithoutResult { coverage.upsertOnUpload(shared.id, owner.id, OffsetDateTime.now()) }
        assertEquals(true, visibility(sharedKey))
    }

    @Test
    fun `concurrent upload counter and preference writes preserve both values and other photographers choices`() {
        val a = photographer()
        val b = photographer()
        val shared = event()
        transactions.executeWithoutResult {
            coverage.upsertOnUpload(shared.id, a.id, OffsetDateTime.now())
            coverage.upsertOnUpload(shared.id, b.id, OffsetDateTime.now())
        }
        metadataSettings.setEvent(b.id, shared.id, true)

        val start = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        try {
            val upload = pool.submit {
                start.await()
                transactions.executeWithoutResult { coverage.upsertOnUpload(shared.id, a.id, OffsetDateTime.now()) }
            }
            val preference = pool.submit {
                start.await()
                metadataSettings.setEvent(a.id, shared.id, true)
            }
            start.countDown()
            upload.get(10, TimeUnit.SECONDS)
            preference.get(10, TimeUnit.SECONDS)
        } finally {
            pool.shutdownNow()
        }

        val aRow = coverage.findById(EventPhotographerId(shared.id, a.id)).orElseThrow()
        assertEquals(2, aRow.photoCount)
        assertEquals(true, aRow.metadataPublic)
        assertEquals(true, coverage.findById(EventPhotographerId(shared.id, b.id)).orElseThrow().metadataPublic)
        assertEquals(false, metadataSettings.getDefault(a.id).metadataPublicByDefault)
    }

    @Test
    fun `sales insert uses private database default and metadata is deleted with its photo`() {
        val photographer = photographer()
        val salesEvent = event()
        val photoEvent = event()
        jdbc.update("INSERT INTO photographer_settings (user_id, metadata_public_by_default) VALUES (?, true)", photographer.id)

        transactions.executeWithoutResult {
            coverage.adjustSalesCounters(salesEvent.id, photographer.id, 1, BigDecimal("10.00"))
        }
        assertEquals(false, visibility(EventPhotographerId(salesEvent.id, photographer.id)))

        val photo = photos.save(Photo(eventId = photoEvent.id, photographerId = photographer.id, s3Key = "metadata-test/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE))
        jdbc.update("INSERT INTO photo_metadata (photo_id, state, extractor_version) VALUES (?, 'PENDING', 1)", photo.id)
        photos.delete(photo)
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM photo_metadata WHERE photo_id = ?", Int::class.java, photo.id))
    }

    @Test
    fun `typed JSON metadata round trips and rejects nonpositive original size`() {
        val owner = photographer()
        val photo = photos.save(Photo(eventId = event().id, photographerId = owner.id, s3Key = "metadata-test/${UUID.randomUUID()}", pricePhp = BigDecimal.ONE))
        val fields = PhotoMetadataFields(cameraMake = "Canon", iso = 400, takenAtLocal = LocalDateTime.of(2026, 9, 28, 6, 35, 20))
        photoMetadata.saveAndFlush(PhotoMetadata(photoId = photo.id, originalSizeBytes = 7421000, fields = fields, state = PhotoMetadataState.READY))
        assertEquals(fields, photoMetadata.findById(photo.id).orElseThrow().fields)
        assertFailsWith<org.springframework.dao.DataIntegrityViolationException> {
            jdbc.update("UPDATE photo_metadata SET original_size_bytes = 0 WHERE photo_id = ?", photo.id)
        }
    }

    @Test
    fun `metadata DTO keeps file facts but hides fields outside ready and empty states`() {
        val id = UUID.randomUUID()
        val row = PhotoMetadata(photoId = id, originalFilename = "IMG.JPG", originalSizeBytes = 7421000, fields = PhotoMetadataFields(cameraMake = "Canon"))
        assertEquals("unavailable", (null as PhotoMetadata?).toDto(id).status)
        assertEquals("pending", row.toDto().status)
        assertEquals(null, row.toDto().fields)
        row.state = PhotoMetadataState.PROCESSING
        assertEquals("pending", row.toDto().status)
        row.state = PhotoMetadataState.FAILED
        assertEquals(null, row.toDto().fields)
        row.state = PhotoMetadataState.EMPTY
        assertEquals(PhotoMetadataFields(), row.toDto().fields)
        row.state = PhotoMetadataState.READY
        assertEquals("Canon", row.toDto().fields?.cameraMake)
        assertEquals("IMG.JPG", row.toDto().originalFilename)
        assertEquals(7421000, row.toDto().fileSizeBytes)
    }

    @Test
    fun `normalized numeric fields reject nonpositive and nonfinite values`() {
        assertFailsWith<IllegalArgumentException> { PhotoMetadataFields(aperture = 0.0) }
        assertFailsWith<IllegalArgumentException> { PhotoMetadataFields(aperture = Double.NaN) }
        assertFailsWith<IllegalArgumentException> { PhotoMetadataFields(iso = -1) }
    }

    private fun photographer(): User = users.save(
        User(email = "metadata-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Metadata", role = Role.PHOTOGRAPHER),
    )

    private fun event(createdBy: UUID? = null): Event = events.save(
        Event(slug = "metadata-${UUID.randomUUID()}", name = "Metadata", date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE, createdBy = createdBy),
    )

    private fun ownedRequest(title: String) = CreateMyEventRequest(
        title = title,
        date = "2026-10-10",
        location = "Cebu",
        pricingMode = "free",
    )

    private fun visibility(key: EventPhotographerId): Boolean = jdbc.queryForObject(
        "SELECT metadata_public FROM event_photographer WHERE event_id = ? AND photographer_id = ?",
        Boolean::class.java,
        key.eventId,
        key.photographerId,
    )!!

    private fun request(method: String, path: String, user: User, body: String = ""): HttpResponse<String> {
        val builder = HttpRequest.newBuilder(URI("http://localhost:$port$path"))
            .header("Authorization", "Bearer ${tokens.createAccessToken(user)}")
            .header("Content-Type", "application/json")
        val request = if (method == "GET") builder.GET() else builder.PUT(HttpRequest.BodyPublishers.ofString(body))
        return http.send(request.build(), HttpResponse.BodyHandlers.ofString())
    }
}
