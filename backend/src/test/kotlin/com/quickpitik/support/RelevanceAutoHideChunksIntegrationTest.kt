package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.LegalProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.entity.Order
import com.quickpitik.service.orders.OrderCapability
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Payment
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PaymentRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.orders.OrderAccessTokenService
import com.quickpitik.service.orders.PaymongoClient
import com.quickpitik.service.orders.PaymongoWebhookService
import com.quickpitik.service.relevance.AnthropicClient
import com.quickpitik.service.relevance.RelevanceCheckService
import com.quickpitik.service.relevance.RelevanceMeasurer
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.context.ApplicationEventPublisher
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionTemplate
import java.io.ByteArrayInputStream
import java.lang.reflect.InvocationTargetException
import java.math.BigDecimal
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Clock
import java.time.Duration
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// RV-2.2b (plan part 2, A.2): a current auto-hide case with target HIDE_ALL hides its whole batch in chunks, each chunk
// its own transaction, and later uploads join the hide. The web server runs for the paid-download check (G18). The
// container is shared, so fixtures sit in 2007 and assertions look only at this class's own rows.
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class RelevanceAutoHideChunksIntegrationTest : PostgresIntegrationTest() {
    @LocalServerPort var port = 0
    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var alerts: RelevanceAlertRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var decisions: AdminDecisionLogService
    @Autowired private lateinit var orders: OrderRepository
    @Autowired private lateinit var orderItems: OrderItemRepository
    @Autowired private lateinit var payments: PaymentRepository
    @Autowired private lateinit var grants: DownloadGrantRepository
    @Autowired private lateinit var webhook: PaymongoWebhookService
    @Autowired private lateinit var tokens: OrderAccessTokenService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var resend: com.quickpitik.service.email.ResendClient
    @MockitoBean lateinit var paymongo: PaymongoClient
    private val bytes = byteArrayOf(4, 5, 6)

    @BeforeEach fun stubs() {
        Mockito.`when`(storage.open(anyArg())).thenAnswer { ByteArrayInputStream(bytes) }
        Mockito.`when`(storage.exists(anyArg())).thenReturn(true)
        Mockito.`when`(storage.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://synthetic.test/image")
        Mockito.`when`(resend.send(anyArg())).thenReturn(com.quickpitik.dto.email.ResendSendEmailResponse("re_synthetic"))
    }
    private fun <T> anyArg(): T = Mockito.any()

    @Test
    fun `with the mode off nothing runs`() {
        val b = batch(photos = 3)
        val c = case(b, "JUNK_DUMP")

        service(mode = RelevanceMode.OFF).tick()

        assertEquals(listOf("LIVE"), statuses(b).distinct())
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
        assertEquals(null, row(c, "hide_finished_at").single())
    }

    @Test
    fun `in record-only no photo, hold, flag, message or e-mail changes`() {
        val b = batch(photos = 3)
        case(b, "JUNK_DUMP")
        val before = untouched()

        service(mode = RelevanceMode.RECORD_ONLY).tick()

        assertEquals(before, untouched())
    }

    @Test
    fun `a case applied in record-only is never acted on`() {
        val b = batch(photos = 3)
        val c = case(b, "JUNK_DUMP", mode = "RECORD_ONLY")

        service().reconcileHolds()

        assertEquals(listOf("LIVE"), statuses(b).distinct())
        assertEquals(0L, holds.countLiveHolds(b.id, "AUTO"))
        assertEquals(null, row(c, "hide_finished_at").single())
    }

    @Test
    fun `a junk dump verdict hides the whole batch in chunks of the chunk size`() {
        val b = batch(photos = 7)
        val c = case(b, "JUNK_DUMP")
        val service = service(chunkSize = 3, chunksPerTick = 2)

        service.reconcileHolds()
        assertEquals(6, statuses(b).count { it == "HIDDEN" }, "two chunks of three in the first tick")
        assertEquals(null, row(c, "hide_finished_at").single())

        service.reconcileHolds()
        assertEquals(listOf("HIDDEN"), statuses(b).distinct())
        assertEquals(listOf(7), row(c, "hidden_photos"))
    }

    @Test
    fun `no statement binds more ids than the chunk size`() {
        val b = batch(photos = 7)
        case(b, "COPIED")
        val bound = mutableListOf<Int>()
        val counted = Mockito.mock(PhotoRepository::class.java) { inv ->
            if (inv.method.name == "lockIdsForUpdate") bound += (inv.arguments[0] as Collection<*>).size
            try { inv.method.invoke(photos, *inv.arguments) } catch (e: InvocationTargetException) { throw e.targetException }
        }
        val countedHolds = Mockito.mock(RelevanceHoldRepository::class.java) { inv ->
            if (inv.method.name == "hidePhotos") bound += (inv.arguments[0] as Collection<*>).size
            try { inv.method.invoke(holds, *inv.arguments) } catch (e: InvocationTargetException) { throw e.targetException }
        }

        val service = service(chunkSize = 3, chunksPerTick = 10, photoRepo = counted, holdRepo = countedHolds)
        service.reconcileHolds()

        assertEquals(listOf("HIDDEN"), statuses(b).distinct())
        assertTrue(bound.isNotEmpty() && bound.all { it <= 3 }, "bound $bound")
    }

    @Test
    fun `a photo uploaded after the hide is hidden on the next pass`() {
        val b = batch(photos = 2)
        val c = case(b, "WRONG_EVENT")
        val service = service()
        service.reconcileHolds()
        assertEquals(listOf(2), row(c, "hidden_photos"))

        val late = photo(b)
        service.reconcileHolds()

        assertEquals("HIDDEN", status(late))
        assertEquals(3L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `a photo an admin restored under this case is not hidden again`() {
        val b = batch(photos = 2)
        val c = case(b, "JUNK_DUMP")
        val restored = photo(b)
        jdbc.update(
            "INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at, restored_at, restored_by) " +
                "VALUES (?, ?, 'AUTO', ?, ?, ?, 'ADMIN')", b.id, restored, c, NOW, NOW,
        )

        service().reconcileHolds()

        assertEquals("LIVE", status(restored))
        assertEquals(2L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `the event going private after the hide lifts nothing`() {
        val b = batch(photos = 2)
        case(b, "JUNK_DUMP")
        val service = service()
        service.reconcileHolds()
        jdbc.update("UPDATE events SET visibility = 'UNLISTED' WHERE id = ?", b.event)

        service.reconcileHolds()

        assertEquals(listOf("HIDDEN"), statuses(b).distinct())
        assertEquals(2L, holds.countLiveHolds(b.id, "AUTO"))
    }

    @Test
    fun `one automatic decision row and one big-hide alert per case, however many passes`() {
        val b = batch(photos = 5)
        val c = case(b, "JUNK_DUMP")
        val service = service(chunkSize = 2, chunksPerTick = 1, bigHidePhotos = 4)

        repeat(4) { service.reconcileHolds() }
        photo(b)
        repeat(2) { service.reconcileHolds() }

        assertEquals(6L, holds.countLiveHolds(b.id, "AUTO"))
        val logged = jdbc.queryForList(
            "SELECT decision, automatic, admin_id, meta->>'caseId' AS case_id, meta->>'eventId' AS event_id, meta->>'photos' AS photos " +
                "FROM admin_decision_log WHERE target_user_id = ?", b.photographer,
        )
        assertEquals(listOf(mapOf("decision" to "batch_auto_hidden", "automatic" to true, "admin_id" to null,
            "case_id" to c.toString(), "event_id" to b.event.toString(), "photos" to "5")), logged)
        assertEquals(listOf(listOf("BIG_HIDE", "big_hide:$c", false, 5)),
            jdbc.queryForList("SELECT kind, dedupe_key, would_have, photos FROM relevance_alerts WHERE batch_id = ?", b.id)
                .map { listOf(it["kind"], it["dedupe_key"], it["would_have"], it["photos"]) })
    }

    @Test
    fun `a buyer whose order was created before the hide still gets the download grant, and the paid download works`() {
        val b = batch(photos = 1)
        val bought = statusesWithIds(b).keys.single()
        val buyer = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Buyer', 'RUNNER')", "relbuy-${UUID.randomUUID()}@test.local")
        val order = orders.saveAndFlush(Order(userId = buyer, eventId = b.event, recipientEmail = "relbuy@test.local",
            paymentMethodWire = "gcash", totalPhp = BigDecimal("125.00"), status = OrderStatus.PENDING)).id
        orderItems.saveAndFlush(OrderItem(OrderItemId(order, bought), BigDecimal("125.00")))
        val ref = "pi_${UUID.randomUUID()}"
        payments.saveAndFlush(Payment(orderId = order, provider = "paymongo", providerRef = ref, amountPhp = BigDecimal("125.00")))
        case(b, "COPIED")
        service().reconcileHolds()
        assertEquals("HIDDEN", status(bought))

        webhook.settlePaymentIntent(ref, "pay_synthetic")

        assertEquals(1, grants.findByIdOrderId(order).size)
        val token = tokens.issue(orders.findById(order).orElseThrow(), OrderCapability.BUNDLE)
        val response = HttpClient.newHttpClient().send(
            HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/orders/$order/download-bundle?token=$token&photo=$bought"))
                .timeout(Duration.ofSeconds(10)).GET().build(),
            HttpResponse.BodyHandlers.ofByteArray(),
        )
        assertEquals(200, response.statusCode())
        assertContentEquals(bytes, response.body())
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun service(
        mode: RelevanceMode = RelevanceMode.AUTO_HIDE,
        chunkSize: Int = 1000,
        chunksPerTick: Int = 10,
        bigHidePhotos: Int = 500,
        photoRepo: PhotoRepository = photos,
        holdRepo: RelevanceHoldRepository = holds,
    ): RelevanceCheckService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0),
            alerts = RelevanceProperties.Alerts(bigHidePhotos = bigHidePhotos),
            hide = RelevanceProperties.Hide(chunkSize = chunkSize, chunksPerTick = chunksPerTick))
        val switch = RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false))
        return RelevanceCheckService(batches, cases, props, switch, Clock.systemUTC(), tx, SimpleMeterRegistry(),
            Mockito.mock(RelevanceMeasurer::class.java), Mockito.mock(AnthropicClient::class.java), events, alerts,
            Mockito.mock(ApplicationEventPublisher::class.java), LegalProperties(currentVersion = "1.7", photographerMinVersion = "1.7"),
            holdRepo, photoRepo, decisions)
    }

    // A public shared race's batch with `photos` live photos.
    private fun batch(photos: Int): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "relchunk-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2007-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relchunk-${UUID.randomUUID()}",
        )
        jdbc.update(
            "INSERT INTO event_photographer (event_id, photographer_id, photo_count, first_upload_at, last_upload_at) VALUES (?, ?, ?, ?, ?)",
            event, photographer, photos, UPLOAD, UPLOAD,
        )
        val batchId = id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer)
        return Batch(batchId, event, photographer).also { b -> repeat(photos) { photo(b) } }
    }

    // A current case of the batch, applied in `mode` with target HIDE_ALL.
    private fun case(b: Batch, verdict: String, mode: String = "AUTO_HIDE"): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, decided_at, applied_mode, action, applied_at) " +
            "VALUES (?, 'QUIET', ?, 1, 'APPLIED', ?, ?, ?, 'HIDE_ALL', ?)",
        b.id, UPLOAD, verdict, NOW, mode, NOW,
    )

    private fun photo(b: Batch): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key, price_php, status) " +
            "VALUES (?, ?, ?, 'relchunk/thumb', 'relchunk/wm', 125, 'LIVE')",
        b.event, b.photographer, "relchunk/${UUID.randomUUID()}",
    )

    private fun statusesWithIds(b: Batch): Map<UUID, String> =
        jdbc.queryForList("SELECT id, status FROM photos WHERE event_id = ?", b.event).associate { it["id"] as UUID to it["status"] as String }

    private fun statuses(b: Batch): List<String> = statusesWithIds(b).values.toList()

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun row(caseId: UUID, vararg columns: String): List<Any?> =
        jdbc.queryForMap("SELECT ${columns.joinToString()} FROM relevance_cases WHERE id = ?", caseId).let { r -> columns.map { r[it] } }

    private fun untouched(): List<Any?> = listOf(
        jdbc.queryForList("SELECT status, count(*) AS n FROM photos GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM flag_hidden_photos", Long::class.java),
        jdbc.queryForList("SELECT status, count(*) AS n FROM flags GROUP BY status ORDER BY status"),
        jdbc.queryForObject("SELECT count(*) FROM photographer_messages", Long::class.java),
        jdbc.queryForObject("SELECT count(*) FROM admin_decision_log", Long::class.java),
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2007-03-04T05:06:07.123456Z")
        val UPLOAD: OffsetDateTime = NOW.minusMinutes(20)
    }
}
