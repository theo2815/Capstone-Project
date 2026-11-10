package com.quickpitik.support

import com.quickpitik.dto.admin.UpdateAdminEventRequest
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.ai.RekognitionIdentityClient
import com.quickpitik.service.moderation.ImageModerationService
import com.quickpitik.service.moderation.WordListChecker
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.JwtTokenProvider
import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.test.context.bean.override.mockito.MockitoBean
import software.amazon.awssdk.services.rekognition.RekognitionClient
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.assertEquals

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class RacePriceConcurrencyIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var adminEvents: AdminEventService
    @Autowired private lateinit var tx: TransactionTemplate
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var tokens: JwtTokenProvider
    @LocalServerPort private var port: Int = 0
    @MockitoBean(name = "rekognitionClient") private lateinit var rekognition: RekognitionClient
    @MockitoBean(name = "livenessRekognitionClient") private lateinit var liveness: RekognitionClient
    @MockitoBean private lateinit var identity: RekognitionIdentityClient
    @MockitoBean private lateinit var images: ImageModerationService
    @MockitoBean private lateinit var words: WordListChecker

    private fun id(sql: String, vararg args: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *args)!!

    private fun user(role: String = "PHOTOGRAPHER") = id(
        "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Price Test', ?)",
        "clamp-${UUID.randomUUID()}@test.local", role)

    private fun race() = id("INSERT INTO events (slug, name, date, location, status, price_per_photo, event_type) " +
        "VALUES (?, 'Price Run', current_date + 10, 'Cebu', 'ACTIVE', 125, 'ROAD_RACE')", "clamp-${UUID.randomUUID()}")

    private fun cover(race: UUID, photographer: UUID, custom: Int? = null) {
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, price_php) VALUES (?, ?, ?)", race, photographer, custom)
    }

    private fun photo(race: UUID, photographer: UUID?, price: Int = 125) = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php) VALUES (?, ?, ?, ?)",
        race, photographer, "clamp/${UUID.randomUUID()}", price)

    private fun price(photo: UUID): BigDecimal = jdbc.queryForObject("SELECT price_php FROM photos WHERE id = ?", BigDecimal::class.java, photo)!!

    @Test
    fun `floor change keeps a raise inside the new bounds`() {
        val race = race()
        val photographer = user()
        cover(race, photographer, 180)
        val photo = photo(race, photographer, 180)

        adminEvents.update(user("ADMIN"), race, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("150")))

        assertEquals(BigDecimal("180.00"), price(photo), "a valid custom price must survive an admin floor change")
    }

    @Test
    fun `clamps high and low raises and follows the floor without a coverage row`() {
        val race = race()
        val raised = user()
        val follows = user()
        cover(race, raised, 240)
        cover(race, follows)
        val raisedPhoto = photo(race, raised, 240)
        val followsPhoto = photo(race, follows)
        val noCoveragePhoto = photo(race, user())
        val anonymousPhoto = photo(race, null)
        val otherRacePhoto = photo(race(), raised)
        val admin = user("ADMIN")

        adminEvents.update(admin, race, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("80")))
        assertEquals(BigDecimal("160.00"), price(raisedPhoto))
        assertEquals(BigDecimal("80.00"), price(followsPhoto))
        assertEquals(BigDecimal("80.00"), price(noCoveragePhoto))
        assertEquals(BigDecimal("80.00"), price(anonymousPhoto))
        assertEquals(BigDecimal("125.00"), price(otherRacePhoto))

        adminEvents.update(admin, race, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("200")))
        assertEquals(BigDecimal("200.00"), price(raisedPhoto))
        assertEquals(BigDecimal("200.00"), price(followsPhoto))
        assertEquals(null, jdbc.queryForObject("SELECT price_php FROM event_photographer WHERE event_id = ? AND photographer_id = ?",
            BigDecimal::class.java, race, raised))
        assertEquals(2, jdbc.queryForObject("SELECT count(*) FROM race_price_changes WHERE event_id = ? AND reason = 'CLAMP' AND changed_by = ?",
            Int::class.java, race, admin))
    }

    @Test
    fun `admin list and reset keep the cooldown and return only this race history`() {
        val race = race()
        val photographer = user()
        cover(race, photographer, 180)
        jdbc.update("INSERT INTO photographer_settings (user_id, handle) VALUES (?, ?)", photographer, "price-${UUID.randomUUID().toString().take(8)}")
        jdbc.update("UPDATE event_photographer SET price_changed_at = now() WHERE event_id = ?", race)
        val photo = photo(race, photographer, 180)
        val admin = user("ADMIN")
        val route = "/api/v1/admin/events/$race/photographer-prices"

        assertEquals(401, request(route).statusCode())
        assertEquals(403, request(route, actor = user("RUNNER")).statusCode())
        assertEquals(403, request("$route/$photographer/reset", post = true, actor = photographer).statusCode())
        val listed = request(route, actor = admin)
        assertEquals(200, listed.statusCode(), listed.body())
        val priceRow = ObjectMapper().readTree(listed.body())["data"]["prices"][0]
        assertEquals(photographer.toString(), priceRow["photographerId"].asText())
        assertEquals(180, priceRow["effectivePhp"].asInt())
        val changedAt = priceRow["priceChangedAt"].asText()
        val reset = request("$route/$photographer/reset", post = true, actor = admin)
        assertEquals(200, reset.statusCode(), reset.body())
        val data = ObjectMapper().readTree(reset.body())["data"]
        assertEquals(125, data["prices"][0]["effectivePhp"].asInt())
        assertEquals(false, data["prices"][0]["custom"].asBoolean())
        assertEquals(changedAt, data["prices"][0]["priceChangedAt"].asText())
        assertEquals("RESET", data["history"][0]["reason"].asText())
        assertEquals(admin.toString(), data["history"][0]["changedBy"].asText())
        assertEquals(BigDecimal("125.00"), price(photo))
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM admin_decision_log WHERE target_event_id = ? AND decision = 'race_price_reset'", Int::class.java, race))
        assertEquals(404, request("$route/${UUID.randomUUID()}/reset", post = true, actor = admin).statusCode())
        assertEquals(404, request("/api/v1/admin/events/${UUID.randomUUID()}/photographer-prices", actor = admin).statusCode())
    }

    private fun request(path: String, post: Boolean = false, actor: UUID? = null): HttpResponse<String> {
        val request = HttpRequest.newBuilder(URI("http://localhost:$port$path"))
        if (actor != null) request.header("Authorization", "Bearer ${tokens.createAccessToken(users.findById(actor).orElseThrow())}")
        if (post) request.POST(HttpRequest.BodyPublishers.noBody()) else request.GET()
        return HttpClient.newHttpClient().send(request.build(), HttpResponse.BodyHandlers.ofString())
    }

    @Test
    fun `admin list returns the newest fifty changes only from this race`() {
        val race = race()
        val otherRace = race()
        val photographer = user()
        cover(race, photographer)
        for (n in 0..55) {
            jdbc.update("INSERT INTO race_price_changes (event_id, photographer_id, changed_by, from_php, to_php, reason, changed_at) " +
                "VALUES (?, ?, ?, ?, 125, 'SET', ?)", if (n == 55) otherRace else race, photographer, photographer, 125 + n,
                OffsetDateTime.parse("2026-11-01T00:00:00Z").plusSeconds(n.toLong()))
        }
        val response = request("/api/v1/admin/events/$race/photographer-prices", actor = user("ADMIN"))
        assertEquals(200, response.statusCode(), response.body())
        val data = ObjectMapper().readTree(response.body())["data"]
        assertEquals(50, data["history"].size())
        assertEquals(179, data["history"][0]["fromPhp"].asInt())
        assertEquals(130, data["history"][49]["fromPhp"].asInt())
        assertEquals(125, data["prices"][0]["effectivePhp"].asInt())
        assertEquals(false, data["prices"][0]["custom"].asBoolean())
    }

    @Test
    fun `coverage locks follow photographer UUID order`() {
        val race = race()
        val photographers = List(3) { user() }
        photographers.forEach { cover(race, it) }
        tx.execute {
            assertEquals(photographers.sortedBy { it.toString() }, coverage.lockByEventIdForUpdate(race).map { it.id.photographerId })
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = [false, true])
    fun `floor change and an upload finish without deadlock`(keyed: Boolean) {
        val race = race()
        val photographer = user()
        cover(race, photographer, 180)
        photo(race, photographer, 180)
        val admin = user("ADMIN")
        val holdsCoverage = CountDownLatch(1)
        val releaseUpload = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        try {
            val upload = pool.submit {
                tx.execute {
                    coverage.upsertOnUpload(race, photographer, OffsetDateTime.now())
                    holdsCoverage.countDown()
                    check(releaseUpload.await(15, TimeUnit.SECONDS)) { "upload was not released" }
                    events.incrementPhotoCount(race)
                }
            }
            check(holdsCoverage.await(10, TimeUnit.SECONDS)) { "upload did not lock coverage" }
            val patch = pool.submit {
                adminEvents.update(admin, race, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("150")),
                    key = if (keyed) IdempotencyKey(UUID.randomUUID().toString()) else null)
            }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
            while (jdbc.queryForObject("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() " +
                    "AND wait_event_type = 'Lock' AND query LIKE '%event_photographer%'", Int::class.java) == 0) {
                check(System.nanoTime() < deadline) { "admin PATCH never waited for the coverage lock" }
                Thread.sleep(20)
            }
            releaseUpload.countDown()
            upload.get(10, TimeUnit.SECONDS)
            patch.get(10, TimeUnit.SECONDS)
        } finally {
            releaseUpload.countDown()
            pool.shutdownNow()
        }
        assertEquals(BigDecimal("150.00"), jdbc.queryForObject("SELECT price_per_photo FROM events WHERE id = ?", BigDecimal::class.java, race))
    }
}
