package com.quickpitik.support

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.ai.FaceErasureResult
import com.quickpitik.dto.ai.FaceMatch
import com.quickpitik.dto.ai.FacesSearchResult
import com.quickpitik.dto.photos.PhotoErasureOutcome
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.removal.*
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.StoredObject
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.*
import org.mockito.Mockito.*
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.data.domain.PageRequest
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.math.BigDecimal
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.*

// TASK 16 qualification: one support case walked across three owners through the real HTTP/service chain.
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = ["app.photo-removal.cleanup-interval=PT24H"])
class PhotoRemovalWorkflowIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var service: PhotoRemovalService
    @Autowired lateinit var evidence: PhotoRemovalEvidenceService
    @Autowired lateinit var queries: PhotoRemovalQueryService
    @Autowired lateinit var sweep: PhotoRemovalSweep
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var reviews: PhotoRemovalReviewRepository
    @Autowired lateinit var history: PhotoRemovalHistoryRepository
    @Autowired lateinit var jobs: PhotoErasureJobRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var selfies: UserSelfieRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var orderItems: OrderItemRepository
    @Autowired lateinit var payments: PaymentRepository
    @Autowired lateinit var transactions: TransactionRepository
    @Autowired lateinit var grants: DownloadGrantRepository
    @Autowired lateinit var tokens: JwtTokenProvider
    @Autowired lateinit var mapper: ObjectMapper
    @Autowired lateinit var jdbc: JdbcTemplate
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var ai: AiApiClient
    @MockitoSpyBean lateinit var aiProperties: AiApiProperties
    @MockitoSpyBean lateinit var clock: Clock
    @LocalServerPort var port = 0
    private val client = HttpClient.newHttpClient()
    private val t0 = Instant.parse("2026-10-08T00:00:00Z")
    private val selfie = ByteArrayOutputStream().apply {
        ImageIO.write(BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), "png", this)
    }.toByteArray()
    private lateinit var admin: User

    @BeforeEach fun boundaries() {
        doReturn(t0).`when`(clock).instant()
        doReturn(true).`when`(aiProperties).enabled
        admin = user(Role.ADMIN)
        `when`(storage.put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())).thenAnswer {
            outsideTransaction(); StoredObject(it.getArgument(0), it.getArgument<ByteArray>(1).size.toLong(), it.getArgument(2))
        }
        `when`(storage.open(anyString())).thenAnswer { outsideTransaction(); selfie.inputStream() }
        `when`(storage.presignedGetUrl(anyString(), any() ?: java.time.Duration.ZERO)).thenReturn("https://synthetic.test/preview")
        doAnswer { outsideTransaction(); null }.`when`(storage).delete(anyString())
        `when`(ai.erasePhotoFaces(any(UUID::class.java) ?: UUID(0, 0), anyString())).thenAnswer { outsideTransaction(); FaceErasureResult.DELETED }
    }

    @Test fun oneSupportCaseAcrossThreeOwnersResolvesIndependently() {
        // Fixture: one public platform event with three owners, a false positive and a purchase; one unlisted control event.
        val platform = event(EventVisibility.PUBLIC); val control = event(EventVisibility.UNLISTED)
        val ownerA = user(); val ownerB = user(); val ownerC = user(); val buyer = user(Role.RUNNER)
        val a = photo(platform, ownerA, "person-a"); val b = photo(platform, ownerB, "person-b")
        val c = photo(platform, ownerC, "person-c"); val falsePositive = photo(platform, ownerA, "person-lookalike")
        val controlPhoto = photo(control, ownerA, "person-a") // Same provider person in another event.
        val order = purchase(buyer, ownerA, platform, a)
        val financeBefore = financeRows(order.id)
        provider(platform.id, listOf(FaceMatch("person-a", similarity = 0.97), FaceMatch("person-b", similarity = 0.95),
            FaceMatch("person-c", similarity = 0.93), FaceMatch("person-lookalike", similarity = 0.9)))

        // Support intake: case 1 is selfie-matched; candidates alone restrict nothing and stay inside the event.
        val created = service.create(admin.id, CreateRemovalRequestCommand(platform.id, "ticket-workflow", "runner@private.test",
            reason = "Runner asked for removal", commandId = UUID.randomUUID()))
        val withEvidence = evidence.replace(admin.id, created.id, UUID.randomUUID(), created.version, SearchImage(selfie, "image/png", "case.png"))
        val evidenceKey = requests.findById(created.id).orElseThrow().evidenceKey!!
        val candidates = page(admin("POST", created.id, "matches", mapOf("commandId" to UUID.randomUUID(), "expectedVersion" to withEvidence.version)))
        assertEquals(setOf(a.id, b.id, c.id, falsePositive.id), candidates["items"].map { UUID.fromString(it["photo"]["id"].asText()) }.toSet())
        assertTrue(listOf(a, b, c, falsePositive, controlPhoto).all { state(it) == RemovalState.NONE }, "Discovery must not restrict")
        val verified = service.verify(admin.id, created.id, VerifyRemovalCommand(UUID.randomUUID(), detail(created.id).version,
            "Support compared the selfie", "Admin confirmed three photos", listOf(a, b, c).map { VerifyRemovalSelection(it.id, RemovalMatchSource.SELFIE) }))
        val itemA = verified.items.single { it.photoIdSnapshot == a.id }
        val itemB = verified.items.single { it.photoIdSnapshot == b.id }
        val itemC = verified.items.single { it.photoIdSnapshot == c.id }
        // Case 2 shares photo B from a separate support request.
        val shared = service.create(admin.id, CreateRemovalRequestCommand(platform.id, "ticket-shared", "friend@private.test",
            reason = "Friend in the same photo", commandId = UUID.randomUUID()))
        val sharedCase = service.verify(admin.id, shared.id, VerifyRemovalCommand(UUID.randomUUID(), shared.version,
            "Support identity check", "Manual confirmation", listOf(VerifyRemovalSelection(b.id, RemovalMatchSource.MANUAL, "Group photo"))))
        assertEquals(mapOf(a.id to RemovalState.REVIEW, b.id to RemovalState.REVIEW, c.id to RemovalState.REVIEW,
            falsePositive.id to RemovalState.NONE, controlPhoto.id to RemovalState.NONE),
            listOf(a, b, c, falsePositive, controlPhoto).associate { it.id to state(it) }, "Only admin-confirmed photos are restricted")
        assertEquals(listOf(falsePositive.id), photos.findForEventNoBib(platform.id, OffsetDateTime.now(), 1, PageRequest.of(0, 20)).content.map { it.id })
        assertEquals(listOf(controlPhoto.id), photos.findForEventNoBib(control.id, OffsetDateTime.now(), 1, PageRequest.of(0, 20)).content.map { it.id })
        val reviewsByOwner = reviews.findByRequestIdOrderByIdAsc(created.id).associateBy { it.ownerIdSnapshot }
        assertEquals(setOf(ownerA.id, ownerB.id, ownerC.id), reviewsByOwner.keys)
        assertTrue(reviewsByOwner.values.all { it.notifiedAt == t0 && it.deadline == t0.plusSeconds(72 * 3600) })

        // Photographers see only their own items; no evidence, support contact or other events.
        val viewA = owner(ownerA, "GET", "/${created.id}")
        assertEquals(200, viewA.statusCode())
        assertEquals(listOf(a.id), data(viewA)["items"].map { UUID.fromString(it["photoIdSnapshot"].asText()) })
        listOf("runner@private.test", "ticket-workflow", evidenceKey, "Runner asked for removal", b.id.toString(), c.id.toString(), controlPhoto.id.toString())
            .forEach { assertFalse(viewA.body().contains(it), "Owner view disclosed $it") }
        assertTrue(data(owner(ownerA, "GET", "?eventId=${control.id}"))["items"].isEmpty)
        assertTrue(queries.list(admin.id, control.id, null, null, 0, 100).items.isEmpty())

        // A agrees, B objects, C stays silent past the 72-hour window.
        doReturn(t0.plusSeconds(3600)).`when`(clock).instant()
        assertEquals(200, owner(ownerA, "POST", "/${created.id}/responses", RespondToRemovalCommand(UUID.randomUUID(),
            detail(created.id).version, listOf(itemA.id), PhotographerRemovalResponse.AGREE)).statusCode())
        assertEquals(200, owner(ownerB, "POST", "/${created.id}/responses", RespondToRemovalCommand(UUID.randomUUID(),
            detail(created.id).version, listOf(itemB.id), PhotographerRemovalResponse.OBJECT, "I am the subject's coach")).statusCode())
        val overdueAt = t0.plusSeconds(73 * 3600)
        doReturn(overdueAt).`when`(clock).instant()
        assertTrue(sweep.sweepDue(overdueAt, 100) > 0)
        val afterSweep = reviews.findByRequestIdOrderByIdAsc(created.id).associateBy { it.ownerIdSnapshot }
        assertEquals(mapOf(ownerA.id to false, ownerB.id to false, ownerC.id to true), afterSweep.mapValues { it.value.overdue })
        assertTrue(history.findByRequestIdAndCommandId(created.id, UUID.nameUUIDFromBytes(
            "removal:admin:RESPONSE_OVERDUE:${afterSweep.getValue(ownerC.id).id}:".toByteArray())).isNotEmpty())
        assertEquals(RemovalItemState.VERIFIED, items.findById(itemC.id).orElseThrow().state, "Silence is never approval")
        assertEquals(RemovalState.REVIEW, state(c))
        assertNull(jobs.findByPhotoId(c.id))

        // Admin decides: A removed (cleanup fails), B removed over objection, C kept under a dated restriction.
        doThrow(IllegalStateException("synthetic storage outage")).`when`(storage).delete(a.s3Key)
        val decide = DecideRemovalCommand(UUID.randomUUID(), detail(created.id).version, listOf(
            RemovalItemDecision(itemA.id, RemovalDecision.REMOVE, "Owner agreed"),
            RemovalItemDecision(itemB.id, RemovalDecision.REMOVE, "Runner identity verified; objection does not outweigh privacy"),
            RemovalItemDecision(itemC.id, RemovalDecision.RETAIN_RESTRICTED, "Awaiting owner contact", overdueAt.plusSeconds(7 * 86400))))
        assertEquals(200, admin("POST", created.id, "decisions", decide).statusCode())
        val override = history.findByRequestIdAndCommandId(created.id, decide.commandId).single { it.itemId == itemB.id }
        assertEquals("ADMIN_DECIDED", override.action)
        assertEquals(admin.id, override.actorIdSnapshot)
        assertTrue(override.reason!!.contains("objection does not outweigh privacy"))
        assertEquals("photo_removal_remove", jdbc.queryForObject("SELECT decision FROM admin_decision_log WHERE id = ?::uuid",
            String::class.java, override.metadata.getValue("decisionLogId")))
        assertEquals(PhotographerRemovalResponse.OBJECT, items.findById(itemB.id).orElseThrow().ownerResponse, "Override must not invent consent")
        assertEquals(RemovalItemState.REMOVAL_AUTHORIZED, items.findById(itemA.id).orElseThrow().state, "Failed cleanup stays pending")
        assertNotEquals(PhotoErasureOutcome.COMPLETE, jobs.findByPhotoId(a.id)!!.outcome)
        assertEquals(RemovalItemState.REMOVED, items.findById(itemB.id).orElseThrow().state)
        assertEquals(RemovalState.REMOVED, state(a)); assertEquals(RemovalState.REMOVED, state(b)); assertEquals(RemovalState.REVIEW, state(c))
        assertEquals(RemovalRequestStatus.RESOLVING, detail(created.id).status)
        assertNull(detail(created.id).resolvedAt)
        // The shared case closes on B's confirmed erasure without its own authorization.
        val sharedAfter = detail(sharedCase.id)
        assertEquals(RemovalItemState.REMOVED, sharedAfter.items.single().state)
        assertNull(sharedAfter.items.single().adminDecision)
        assertEquals(RemovalRequestStatus.RESOLVED, sharedAfter.status)

        // Admin revisits the restricted photo while A's cleanup is still retrying (parked TASK 14 corner).
        val revisit = admin("POST", created.id, "decisions", DecideRemovalCommand(UUID.randomUUID(), detail(created.id).version,
            listOf(RemovalItemDecision(itemC.id, RemovalDecision.RETAIN, "Owner reached; photo does not show the runner"))))
        assertEquals(200, revisit.statusCode(), revisit.body())
        assertEquals(RemovalItemState.RETAINED, items.findById(itemC.id).orElseThrow().state)
        assertEquals(RemovalState.NONE, state(c))
        assertEquals(RemovalRequestStatus.RESOLVING, detail(created.id).status)

        // External-copy support notes are audit only; platform deletion state is untouched.
        fun deletionState() = Triple(items.findByRequestIdOrderByIdAsc(created.id).associate { it.id to it.state },
            listOf(a, b, c).associate { it.id to state(it) }, listOf(a, b).associate { it.id to jobs.findByPhotoId(it.id)!!.outcome })
        val beforeNote = deletionState()
        assertEquals(200, admin("POST", created.id, "support-updates", RemovalSupportUpdateCommand(UUID.randomUUID(),
            detail(created.id).version, "Runner reports a copy on https://external.test/repost; outside QuickPitik")).statusCode())
        assertEquals(beforeNote, deletionState())
        assertEquals(RemovalRequestStatus.RESOLVING, detail(created.id).status)

        // Retry completes A's cleanup and the case resolves with its selfie gone.
        doAnswer { outsideTransaction(); null }.`when`(storage).delete(a.s3Key)
        assertEquals(200, admin("POST", created.id, "retry-cleanup", RetryRemovalCleanupCommand(UUID.randomUUID(),
            detail(created.id).version, listOf(itemA.id))).statusCode())
        assertEquals(PhotoErasureOutcome.COMPLETE, jobs.findByPhotoId(a.id)!!.outcome)
        val resolved = detail(created.id)
        assertEquals(RemovalRequestStatus.RESOLVED, resolved.status)
        assertNotNull(resolved.resolvedAt)
        assertEquals(mapOf(itemA.id to RemovalItemState.REMOVED, itemB.id to RemovalItemState.REMOVED, itemC.id to RemovalItemState.RETAINED),
            resolved.items.associate { it.id to it.state })
        assertFalse(resolved.evidencePresent)
        val stored = requests.findById(created.id).orElseThrow()
        assertNull(stored.evidenceKey); assertNull(stored.candidateCache)
        assertEquals(RemovalEvidenceCleanupState.DELETED, stored.evidenceCleanupState)
        verify(storage, atLeastOnce()).delete(evidenceKey)
        assertEquals(0L, selfies.countByUserId(admin.id), "Case selfie never enters the admin's selfie library")

        // Seller and buyer finance rows survive removal unchanged; the control event was never touched.
        assertEquals(financeBefore, financeRows(order.id))
        assertEquals(RemovalState.NONE, state(controlPhoto)); assertNull(jobs.findByPhotoId(controlPhoto.id))
        assertEquals(RemovalState.NONE, state(falsePositive))
        verify(ai, never()).facesSearch(any(ByteArray::class.java) ?: byteArrayOf(), anyString(), anyString(), eq(control.id) ?: control.id, anyDouble(), anyInt())
    }

    private fun purchase(buyer: User, seller: User, event: Event, photo: Photo): Order {
        val paidAt = OffsetDateTime.parse("2026-10-07T10:00:00Z")
        val order = orders.saveAndFlush(Order(userId = buyer.id, eventId = event.id, recipientEmail = buyer.email,
            paymentMethodWire = "gcash", totalPhp = BigDecimal("125.00"), status = OrderStatus.PAID, paidAt = paidAt))
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, photo.id), BigDecimal("125.00")))
        payments.saveAndFlush(Payment(orderId = order.id, provider = "paymongo", providerRef = "pi_${UUID.randomUUID()}",
            amountPhp = BigDecimal("125.00"), status = PaymentStatus.SUCCEEDED, paidAt = paidAt))
        transactions.saveAndFlush(Transaction(paidAt = paidAt, photographerId = seller.id, eventId = event.id, photoId = photo.id,
            orderId = order.id, buyerId = buyer.id, amountKeptPhp = BigDecimal("93.75")))
        grants.saveAndFlush(DownloadGrant(DownloadGrantId(order.id, photo.id), paidAt.plusYears(1)))
        return order
    }
    private fun financeRows(orderId: UUID) = listOf("orders", "order_items", "payments", "download_grants", "transactions").map { table ->
        jdbc.queryForList("SELECT * FROM $table WHERE ${if (table == "orders") "id" else "order_id"} = ?", orderId)
    }
    private fun provider(eventId: UUID, matches: List<FaceMatch>) = doAnswer {
        outsideTransaction()
        assertEquals(eventId, it.getArgument<UUID>(3), "Face search must stay event-scoped")
        FacesSearchResult(matches)
    }.`when`(ai).facesSearch(any(ByteArray::class.java) ?: byteArrayOf(), anyString(), anyString(), any(UUID::class.java) ?: eventId, anyDouble(), anyInt())
    private fun outsideTransaction() = assertFalse(TransactionSynchronizationManager.isActualTransactionActive(), "I/O holds a transaction")
    private fun state(photo: Photo) = photos.findById(photo.id).orElseThrow().removalState
    private fun detail(id: UUID) = queries.adminDetail(admin.id, id)
    private fun event(visibility: EventVisibility) = events.saveAndFlush(Event(slug = "workflow-${UUID.randomUUID()}", name = "Synthetic $visibility run",
        date = LocalDate.of(2026, 10, 4), location = "Cebu", status = EventStatus.ACTIVE, visibility = visibility))
    private fun user(role: Role = Role.PHOTOGRAPHER) = users.saveAndFlush(User(email = "workflow-${UUID.randomUUID()}@test.local",
        passwordHash = "unused", name = "Synthetic $role", role = role))
    private fun photo(event: Event, owner: User, person: String) = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
        s3Key = "synthetic/${UUID.randomUUID()}.jpg", thumbnailS3Key = "synthetic/${UUID.randomUUID()}-thumb.jpg", pricePhp = BigDecimal("125.00"),
        facePersons = mutableSetOf(PhotoFacePersonEmbed(0, person, FaceProvider.AI_API))))
    private fun data(response: HttpResponse<String>): JsonNode = mapper.readTree(response.body())["data"]
    private fun page(response: HttpResponse<String>): JsonNode { assertEquals(200, response.statusCode(), response.body()); return data(response) }
    private fun admin(method: String, id: UUID, route: String, body: Any) =
        send(admin, method, "/api/v1/admin/photo-removal-requests/$id/$route", body)
    private fun owner(actor: User, method: String, path: String, body: Any? = null) =
        send(actor, method, "/api/v1/me/photographer/photo-removal-requests$path", body)
    private fun send(actor: User, method: String, path: String, body: Any?): HttpResponse<String> = client.send(HttpRequest.newBuilder(URI("http://localhost:$port$path"))
        .header("Authorization", "Bearer ${tokens.createAccessToken(actor)}").header("Content-Type", "application/json")
        .method(method, body?.let { HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(it)) } ?: HttpRequest.BodyPublishers.noBody()).build(),
        HttpResponse.BodyHandlers.ofString())
}
