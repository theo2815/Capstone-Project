package com.quickpitik.support

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.zaxxer.hikari.HikariConfig
import com.zaxxer.hikari.HikariDataSource
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.photographer.PhotographerMessageService
import com.quickpitik.service.removal.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.context.TestConfiguration
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Import
import org.springframework.context.annotation.Primary
import org.springframework.data.domain.PageRequest
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.*
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.*

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = ["app.photo-removal.cleanup-interval=PT24H", "app.rate-limit.enabled=false"])
@Import(PhotoRemovalPhotographerIntegrationTest.ClockConfig::class)
class PhotoRemovalPhotographerIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var service: PhotoRemovalService
    @Autowired lateinit var queries: PhotoRemovalQueryService
    @Autowired lateinit var inbox: PhotographerMessageService
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var reviews: PhotoRemovalReviewRepository
    @Autowired lateinit var history: PhotoRemovalHistoryRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var jwt: JwtTokenProvider
    @Autowired lateinit var mapper: ObjectMapper
    @Autowired lateinit var sweepService: PhotoRemovalSweep
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var clock: MutableClock
    @Autowired lateinit var datasource: javax.sql.DataSource
    @LocalServerPort var port: Int = 0
    private val client = HttpClient.newHttpClient()
    private val now = Instant.parse("2026-10-05T00:00:00Z")
    private lateinit var admin: User
    private lateinit var event: Event
    private val fixtureCases = mutableListOf<UUID>()

    @AfterEach fun closeOwnFixtures() {
        tx.execute { fixtureCases.forEach { requests.findForUpdate(it)?.status = RemovalRequestStatus.CANCELLED } }
        fixtureCases.clear()
    }

    @BeforeEach fun fixture() {
        datasource.connection.use { connection -> connection.createStatement().use { statement ->
            statement.executeQuery("SELECT current_database()").use { result ->
                assertTrue(result.next()); assertEquals("removal_responses_task7", result.getString(1))
            }
        } }
        clock.set(now)
        admin = user(Role.ADMIN)
        event = events.saveAndFlush(Event(slug = "response-${UUID.randomUUID()}", name = "Synthetic run",
            date = LocalDate.of(2026, 10, 5), location = "Cebu", status = EventStatus.ACTIVE))
    }

    // Missing owner scoping would disclose another owner's photos or private case evidence.
    @Test fun ownerCannotInspectAnotherGroup() {
        val a = user(); val b = user(); val case = verified(listOf(photo(b)))
        val own = get(b, case.id)
        assertEquals(200, own.statusCode())
        assertEquals(404, get(a, case.id).statusCode())
        val dto = data(own)
        assertEquals("Privacy removal request", dto["sanitizedReason"].asText())
        for (field in listOf("requesterContact", "supportReference", "requesterReason", "verificationNote", "evidenceKey", "evidenceGeneration"))
            assertFalse(dto.has(field), field)
        assertEquals(0, data(list(a))["total"].asInt())
        assertEquals(case.id.toString(), data(list(b))["items"].single()["id"].asText())
    }

    // Dropping conditional objection validation would accept an empty objection.
    @Test fun objectionRequiresReason() {
        val owner = user(); val case = verified(listOf(photo(owner))); val id = case.items.single().id
        assertEquals(400, respond(owner, case, listOf(id), PhotographerRemovalResponse.OBJECT, " ").statusCode())
        assertNull(items.findById(id).orElseThrow().ownerResponse)
        assertEquals(200, respond(owner, case, listOf(id), PhotographerRemovalResponse.OBJECT, "Please investigate").statusCode())
    }

    // Silence must create exactly one admin action, without consent, decision or erasure.
    @Test fun silenceDoesNotBecomeApproval() {
        val owner = user(); val case = verified(listOf(photo(owner)))
        val deadline = Instant.parse("2026-10-08T00:00:00Z")
        assertEquals(deadline, case.reviews.single().deadline)
        assertEquals(0, sweep(deadline.minusNanos(1)))
        assertEquals(1, sweep(deadline))
        assertEquals(0, sweep(deadline.plusSeconds(1)))
        val overdue = items.findById(case.items.single().id).orElseThrow()
        assertNull(overdue.ownerResponse)
        assertNull(overdue.adminDecision)
        assertEquals(RemovalItemState.VERIFIED, overdue.state)
        assertEquals(1, audit(case.id).count { it.action == "REVIEW_OVERDUE" })
        assertEquals(deadline, reviews.findById(case.reviews.single().id).orElseThrow().overdueAt)
        assertEquals(200, respond(owner, queries.adminDetail(admin.id, case.id), listOf(overdue.id), PhotographerRemovalResponse.AGREE).statusCode())
    }

    // Reading/removing the inbox notice or amending a response must not restart the clock.
    @Test fun deadlineIsNotExtendedByMessageRead() {
        val owner = user(); val case = verified(listOf(photo(owner)))
        val originalDeadline = case.reviews.single().deadline
        assertEquals(200, get(owner, case.id).statusCode())
        val review = reviews.findById(case.reviews.single().id).orElseThrow()
        clock.set(now.plusSeconds(3600))
        inbox.markRead(owner.id, review.notifiedMessageId!!)
        inbox.remove(owner.id, review.notifiedMessageId!!)
        assertEquals(200, respond(owner, case, listOf(case.items.single().id), PhotographerRemovalResponse.AGREE).statusCode())
        assertEquals(originalDeadline, data(get(owner, case.id))["reviews"].single()["deadline"].asText().let(Instant::parse))
    }

    // A group response must change only the authenticated owner's selected items.
    @Test fun mixedOwnerResponsesAreIndependent() {
        val a = user(); val b = user(); val case = verified(listOf(photo(a), photo(b)))
        val aItem = case.items.single { it.ownerIdSnapshot == a.id }; val bItem = case.items.single { it.ownerIdSnapshot == b.id }
        val first = respond(a, case, listOf(aItem.id), PhotographerRemovalResponse.OBJECT, "Ownership concern")
        assertEquals(200, first.statusCode())
        assertEquals(listOf(aItem.id.toString()), data(first)["items"].map { it["id"].asText() })
        assertNull(items.findById(bItem.id).orElseThrow().ownerResponse)
        val current = queries.adminDetail(admin.id, case.id)
        assertEquals(200, respond(b, current, listOf(bItem.id), PhotographerRemovalResponse.AGREE).statusCode())
        assertEquals(PhotographerRemovalResponse.OBJECT, items.findById(aItem.id).orElseThrow().ownerResponse)
        assertEquals(PhotographerRemovalResponse.AGREE, items.findById(bItem.id).orElseThrow().ownerResponse)
        assertEquals(0, sweep(now.plusSeconds(72 * 3600)))
    }

    @Test fun `whole response validates owner case duplicates and state before any write`() {
        val a = user(); val b = user(); val case = verified(listOf(photo(a), photo(a), photo(b)))
        val own = case.items.filter { it.ownerIdSnapshot == a.id }; val other = case.items.single { it.ownerIdSnapshot == b.id }
        assertEquals(404, respond(a, case, listOf(own.first().id, other.id), PhotographerRemovalResponse.AGREE).statusCode())
        assertEquals(404, respond(a, case, listOf(own.first().id, UUID.randomUUID()), PhotographerRemovalResponse.AGREE).statusCode())
        assertEquals(400, respond(a, case, listOf(own.first().id, own.first().id), PhotographerRemovalResponse.AGREE).statusCode())
        tx.execute { items.findById(own.last().id).orElseThrow().state = RemovalItemState.REMOVAL_AUTHORIZED }
        assertEquals(409, respond(a, case, own.map { it.id }, PhotographerRemovalResponse.AGREE).statusCode())
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).all { it.ownerResponse == null })
        assertEquals(0, audit(case.id).count { it.action == "OWNER_RESPONDED" })
    }

    @Test fun `receipt replay remains original and collisions stale amendments and terminal decisions are guarded`() {
        val owner = user(); val case = verified(listOf(photo(owner))); val id = case.items.single().id
        val command = UUID.randomUUID()
        val first = respond(owner, case, listOf(id), PhotographerRemovalResponse.AGREE, commandId = command)
        assertEquals(200, first.statusCode())
        assertEquals(409, respond(owner, case, listOf(id), PhotographerRemovalResponse.OBJECT, "Changed", command).statusCode())
        assertEquals(409, respond(owner, case, listOf(id), PhotographerRemovalResponse.AGREE).statusCode())
        val current = queries.adminDetail(admin.id, case.id)
        assertEquals(200, respond(owner, current, listOf(id), PhotographerRemovalResponse.OBJECT, "Amendment").statusCode())
        assertEquals(data(first), data(respond(owner, case, listOf(id), PhotographerRemovalResponse.AGREE, commandId = command)))
        assertEquals(2, audit(case.id).count { it.action == "OWNER_RESPONDED" })
        tx.execute { items.findById(id).orElseThrow().apply { state = RemovalItemState.RETAINED; adminDecision = RemovalDecision.RETAIN; adminDecisionReason = "Keep"; adminDecidedAt = now; adminDecisionActorId = admin.id } }
        assertEquals(409, respond(owner, queries.adminDetail(admin.id, case.id), listOf(id), PhotographerRemovalResponse.AGREE).statusCode())
    }

    @Test fun `already deleted response reports actual trash without marking erased`() {
        val owner = user(); val live = photo(owner); val case = verified(listOf(live)); val id = case.items.single().id
        val claim = respond(owner, case, listOf(id), PhotographerRemovalResponse.ALREADY_DELETED)
        assertEquals(200, claim.statusCode())
        assertEquals("PRESENT", data(claim)["items"].single()["photoState"].asText())
        assertEquals(RemovalItemState.VERIFIED, items.findById(id).orElseThrow().state)
        tx.execute { photos.trashOwned(owner.id, event.id, listOf(live.id), now.atOffset(ZoneOffset.UTC)) }
        val amended = respond(owner, queries.adminDetail(admin.id, case.id), listOf(id), PhotographerRemovalResponse.ALREADY_DELETED)
        assertEquals(200, amended.statusCode())
        assertEquals("TRASHED", data(amended)["items"].single()["photoState"].asText())
        assertNull(photos.findById(live.id).orElseThrow().purgedAt)
    }

    @Test fun `photo row observation never claims confirmed cleanup without an erasure job`() {
        val owner = user(); val case = verified(listOf(photo(owner))); val id = case.items.single().id
        val response = respond(owner, case, listOf(id), PhotographerRemovalResponse.ALREADY_DELETED)
        assertEquals(200, response.statusCode())
        val selected = data(response)["items"].single()
        assertTrue(selected["cleanupState"].isNull, "Photo presence is an observation, not completed cleanup")
        assertTrue(selected.has("photoState"), "The actual photo observation must be separate from cleanup")
        assertEquals("PRESENT", selected["photoState"].asText())
    }

    @Test fun `concurrent sweep records one overdue event and partial responses still escalate`() {
        val owner = user(); val case = verified(listOf(photo(owner), photo(owner)))
        assertEquals(200, respond(owner, case, listOf(case.items.first().id), PhotographerRemovalResponse.AGREE).statusCode())
        val pool = Executors.newFixedThreadPool(2)
        try {
            val results = (1..2).map { pool.submit<Int> { sweep(now.plusSeconds(72 * 3600)) } }.map { it.get(20, TimeUnit.SECONDS) }
            assertEquals(1, results.sum())
            assertEquals(1, audit(case.id).count { it.action == "REVIEW_OVERDUE" })
        } finally { pool.shutdownNow() }
    }

    @Test fun `sweep limit bounds actions within one case and cannot starve remaining owners`() {
        val owners = (1..3).map { user() }; val case = verified(owners.map { photo(it) })
        assertEquals(1, sweep(now.plusSeconds(72 * 3600), 1))
        assertEquals(1, reviews.findByRequestIdOrderByIdAsc(case.id).count { it.overdue })
        assertEquals(1, sweep(now.plusSeconds(72 * 3600), 1))
        assertEquals(1, sweep(now.plusSeconds(72 * 3600), 1))
        assertEquals(0, sweep(now.plusSeconds(72 * 3600), 1))
        assertEquals(3, audit(case.id).count { it.action == "REVIEW_OVERDUE" })
    }

    @Test fun `unavailable owner orphan and due restricted retention require admin action without erasure`() {
        val unavailable = user().apply { suspendedAt = now.atOffset(ZoneOffset.UTC) }.let(users::saveAndFlush)
        val owner = user(); val restrictedPhoto = photo(owner)
        val case = missingUnavailableHistory(listOf(photo(unavailable), photo(null), restrictedPhoto))
        val retention = case.items.single { it.photoIdSnapshot == restrictedPhoto.id }
        tx.execute {
            items.findById(retention.id).orElseThrow().apply {
                state = RemovalItemState.RETAINED_RESTRICTED; adminDecision = RemovalDecision.RETAIN_RESTRICTED
                adminDecisionReason = "Restricted retention"; retentionReviewAt = now.plusSeconds(60)
                adminDecidedAt = now; adminDecisionActorId = admin.id
            }
        }
        assertEquals(2, sweep(now))
        assertEquals(0, sweep(now.plusSeconds(59)))
        assertEquals(1, sweep(now.plusSeconds(60)))
        assertEquals(0, sweep(now.plusSeconds(61)))
        assertEquals(2, audit(case.id).count { it.action == "OWNER_UNAVAILABLE" })
        assertEquals(1, audit(case.id).count { it.action == "REVIEW_OVERDUE" })
        assertEquals(RemovalItemState.RETAINED_RESTRICTED, items.findById(retention.id).orElseThrow().state)
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).all { it.ownerResponse == null && it.cleanupJobId == null })
        assertNull(photos.findById(restrictedPhoto.id).orElseThrow().purgedAt)
    }

    @Test fun `owner list filters and pagination count only this owners unanswered items`() {
        val a = user(); val b = user(); val answered = verified(listOf(photo(a), photo(b)))
        val ownItem = answered.items.single { it.ownerIdSnapshot == a.id }
        assertEquals(200, respond(a, answered, listOf(ownItem.id), PhotographerRemovalResponse.AGREE).statusCode())
        val pending = verified(listOf(photo(a)))
        verified(listOf(photo(b)))
        val filtered = data(send(a, "?eventId=${event.id}&status=OPEN&needsAction=true&offset=0&limit=1"))
        assertEquals(1, filtered["total"].asInt())
        assertEquals(1, filtered["limit"].asInt())
        assertEquals(pending.id.toString(), filtered["items"].single()["id"].asText())
        assertEquals(answered.id.toString(), data(send(a, "?needsAction=false"))["items"].single()["id"].asText())
        assertEquals(0, data(send(a, "?eventId=${UUID.randomUUID()}"))["total"].asInt())
        assertEquals(0, data(send(a, "?status=CANCELLED"))["total"].asInt())
        val next = data(send(a, "?offset=1&limit=1"))
        assertEquals(2, next["total"].asInt()); assertEquals(1, next["offset"].asInt()); assertEquals(1, next["items"].size())
    }

    @Test fun `owner repository filters support null true false and scoped page totals`() {
        val a = user(); val b = user(); val answered = verified(listOf(photo(a), photo(b)))
        val owned = answered.items.single { it.ownerIdSnapshot == a.id }
        assertEquals(200, respond(a, answered, listOf(owned.id), PhotographerRemovalResponse.AGREE).statusCode())
        val pending = verified(listOf(photo(a))); verified(listOf(photo(b)))
        val first = PageRequest.of(0, 1)
        assertEquals(2L, requests.findForPhotographer(a.id, null, null, null, first).totalElements)
        val due = requests.findForPhotographer(a.id, event.id, RemovalRequestStatus.OPEN, true, first)
        assertEquals(1L, due.totalElements); assertEquals(listOf(pending.id), due.content.map { it.id })
        val done = requests.findForPhotographer(a.id, null, null, false, first)
        assertEquals(1L, done.totalElements); assertEquals(listOf(answered.id), done.content.map { it.id })
        assertEquals(0L, requests.findForPhotographer(a.id, UUID.randomUUID(), null, null, first).totalElements)
        assertEquals(0L, requests.findForPhotographer(a.id, null, RemovalRequestStatus.CANCELLED, null, first).totalElements)
        assertEquals(0L, requests.findForPhotographer(UUID.randomUUID(), null, null, null, first).totalElements)
        val next = requests.findForPhotographer(a.id, null, null, null, PageRequest.of(1, 1))
        assertEquals(2L, next.totalElements); assertEquals(1, next.content.size)
    }

    @Test fun `resolved restricted retention returns to admin follow up on its review date`() {
        val owner = user(); val case = verified(listOf(photo(owner))); val selected = case.items.single()
        tx.execute {
            items.findById(selected.id).orElseThrow().apply {
                state = RemovalItemState.RETAINED_RESTRICTED; adminDecision = RemovalDecision.RETAIN_RESTRICTED
                adminDecisionReason = "Documented retention exception"; retentionReviewAt = now.plusSeconds(60)
                adminDecidedAt = now; adminDecisionActorId = admin.id
            }
            requests.findForUpdate(case.id)!!.apply { status = RemovalRequestStatus.RESOLVED; resolvedAt = now }
        }
        assertEquals(0, sweep(now.plusSeconds(59)))
        assertEquals(1, sweep(now.plusSeconds(60)))
        assertEquals(0, sweep(now.plusSeconds(61)))
        val followup = audit(case.id).single { it.metadata["kind"] == "RETENTION_DUE" }
        assertEquals("REVIEW_OVERDUE", followup.action)
        assertEquals(RemovalRequestStatus.RESOLVED, requests.findById(case.id).orElseThrow().status)
        assertEquals(RemovalItemState.RETAINED_RESTRICTED, items.findById(selected.id).orElseThrow().state)
    }

    @Test fun `resolved restricted retention is visible in the admin action list when due`() {
        val owner = user(); val case = verified(listOf(photo(owner))); val selected = case.items.single()
        tx.execute {
            items.findById(selected.id).orElseThrow().apply {
                state = RemovalItemState.RETAINED_RESTRICTED; adminDecision = RemovalDecision.RETAIN_RESTRICTED
                adminDecisionReason = "Documented retention exception"; retentionReviewAt = now.plusSeconds(60)
            }
            requests.findForUpdate(case.id)!!.apply { status = RemovalRequestStatus.RESOLVED; resolvedAt = now }
        }
        clock.set(now.plusSeconds(59))
        assertTrue(queries.list(admin.id, event.id, null, true, 0, 100).items.isEmpty())
        clock.set(now.plusSeconds(60))
        assertEquals(listOf(case.id), queries.list(admin.id, event.id, null, true, 0, 100).items.map { it.id })
        assertTrue(queries.list(admin.id, event.id, null, false, 0, 100).items.isEmpty())
    }

    @Test fun `unavailable owner escalation never records an overdue response`() {
        val unavailable = user().apply { suspendedAt = now.atOffset(ZoneOffset.UTC) }.let(users::saveAndFlush)
        val case = missingUnavailableHistory(listOf(photo(unavailable), photo(null)))
        assertEquals(2, sweep(now))
        val followups = audit(case.id).filter { it.metadata["kind"] == "OWNER_UNAVAILABLE" }
        assertEquals(2, followups.size)
        assertTrue(followups.all { it.action == "OWNER_UNAVAILABLE" })
        assertEquals(0, audit(case.id).count { it.action == "REVIEW_OVERDUE" })
        assertTrue(reviews.findByRequestIdOrderByIdAsc(case.id).all { !it.overdue && it.overdueAt == null })
        assertEquals(0, sweep(now.plusSeconds(1)))
    }

    @Test fun `sweep never duplicates unavailable actions already recorded by verification and notification`() {
        val unavailable = user().apply { suspendedAt = now.atOffset(ZoneOffset.UTC) }.let(users::saveAndFlush)
        val case = verified(listOf(photo(unavailable), photo(null)))
        val original = audit(case.id)
        assertEquals(2, original.count { it.action == "OWNER_UNAVAILABLE" })
        assertEquals(0, sweep(now))
        assertEquals(0, sweep(now.plusSeconds(1)))
        assertEquals(original.map { it.id }, audit(case.id).map { it.id })
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).all { it.ownerResponse == null && it.cleanupJobId == null })
    }

    @Test fun `due candidate query finds real bounded actionable cases at the deadline`() {
        val owner = user()
        val first = verified(listOf(photo(owner))); val second = verified(listOf(photo(owner)))
        val deadline = Instant.parse("2026-10-08T00:00:00Z")
        assertTrue(requests.findDueActionRequestIds(deadline.minusMillis(1), PageRequest.of(0, 1)).isEmpty())
        val ordered = listOf(first.id, second.id).sortedBy { it.toString() }
        assertEquals(ordered.take(1), requests.findDueActionRequestIds(deadline, PageRequest.of(0, 1)))
        assertEquals(ordered.drop(1), requests.findDueActionRequestIds(deadline, PageRequest.of(1, 1)))
        assertTrue(requests.findDueActionRequestIds(deadline, PageRequest.of(2, 1)).isEmpty())
    }

    @Test fun `owner detail never exposes a private admin decision reason`() {
        val owner = user(); val case = verified(listOf(photo(owner))); val id = case.items.single().id
        val privateReason = "Private support contact runner-private@test.local ticket-secret"
        tx.execute {
            items.findById(id).orElseThrow().apply {
                state = RemovalItemState.RETAINED; adminDecision = RemovalDecision.RETAIN
                adminDecisionReason = privateReason; adminDecidedAt = now; adminDecisionActorId = admin.id
            }
        }
        val response = get(owner, case.id)
        assertEquals(200, response.statusCode())
        assertFalse(response.body().contains(privateReason), "Private admin reasons must remain outside owner projection")
        val selected = data(response)["items"].single()
        assertTrue(selected["decisionReason"].isNull)
        assertEquals("RETAIN", selected["decision"].asText())
        assertEquals(now, Instant.parse(selected["decidedAt"].asText()))
        assertEquals("Privacy removal request", data(response)["sanitizedReason"].asText())
    }

    @Test fun `group response summary derives only this owners item responses`() {
        val a = user(); val b = user(); val case = verified(listOf(photo(a), photo(a), photo(b)))
        val owned = case.items.filter { it.ownerIdSnapshot == a.id }
        assertEquals(200, respond(a, case, listOf(owned.first().id), PhotographerRemovalResponse.OBJECT, "Investigate").statusCode())
        val dto = data(get(a, case.id)); val review = dto["reviews"].single()
        assertTrue(review.has("responses"), "Owner review must derive its response summary from its own items")
        assertEquals(1, review["responses"]["unanswered"].asInt())
        assertEquals(1, review["responses"]["objected"].asInt())
        assertEquals(0, review["responses"]["agree"].asInt())
        assertEquals(0, review["responses"]["alreadyDeleted"].asInt())
    }

    private fun sweep(at: Instant, limit: Int = 50) = sweepService.sweepDue(at, limit)
    private fun audit(id: UUID) = history.findByRequestIdOrderByOccurredAtAscIdAsc(id, PageRequest.of(0, 100))
    // Persisted recovery fixture with unavailable review/items but no unavailable audit yet.
    // Normal Task6 verification/notifier history is exercised separately and never deleted.
    private fun missingUnavailableHistory(selected: List<Photo>): PhotoRemovalRequestDto {
        val case = service.create(admin.id, CreateRemovalRequestCommand(event.id, "private-support", "private-contact",
            reason = "private-reason", commandId = UUID.randomUUID()))
        fixtureCases += case.id
        tx.execute {
            requests.findForUpdate(case.id)!!.status = RemovalRequestStatus.OPEN
            val groups = selected.mapNotNull { it.photographerId }.distinct().associateWith { ownerId ->
                reviews.saveAndFlush(PhotoRemovalReview(requestId = case.id, ownerIdSnapshot = ownerId, ownerId = ownerId,
                    unavailableReason = if (users.findById(ownerId).orElseThrow().suspendedAt != null) "Owner account inactive" else null))
            }
            items.saveAllAndFlush(selected.map { PhotoRemovalItem(requestId = case.id, photoIdSnapshot = it.id,
                photoId = it.id, ownerIdSnapshot = it.photographerId, reviewId = groups[it.photographerId]?.id,
                state = RemovalItemState.VERIFIED, matchSource = RemovalMatchSource.MANUAL) })
        }
        return queries.adminDetail(admin.id, case.id)
    }
    private fun verified(selected: List<Photo>): PhotoRemovalRequestDto {
        val case = service.create(admin.id, CreateRemovalRequestCommand(event.id, "private-support", "private-contact", reason = "private-reason", commandId = UUID.randomUUID(), adminNote = "private-admin-note"))
        fixtureCases += case.id
        service.verify(admin.id, case.id, VerifyRemovalCommand(UUID.randomUUID(), case.version, "Support", "private-verification",
            selected.map { VerifyRemovalSelection(it.id, RemovalMatchSource.MANUAL, "private-note") }))
        return queries.adminDetail(admin.id, case.id)
    }
    private fun user(role: Role = Role.PHOTOGRAPHER) = users.saveAndFlush(User(email = "response-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Synthetic $role", role = role))
    private fun photo(owner: User?) = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner?.id, s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal("50")))
    private fun get(owner: User, id: UUID) = send(owner, "/$id")
    private fun list(owner: User) = send(owner, "")
    private fun respond(owner: User, case: PhotoRemovalRequestDto, ids: List<UUID>, response: PhotographerRemovalResponse, reason: String? = null, commandId: UUID = UUID.randomUUID()) =
        send(owner, "/${case.id}/responses", mapper.writeValueAsString(RespondToRemovalCommand(commandId, case.version, ids, response, reason)))
    private fun send(owner: User, suffix: String, body: String? = null): HttpResponse<String> {
        val builder = HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/me/photographer/photo-removal-requests$suffix"))
            .header("Authorization", "Bearer ${jwt.createAccessToken(owner)}").header("Content-Type", "application/json")
        return client.send(if (body == null) builder.GET().build() else builder.POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString())
    }
    private fun data(response: HttpResponse<String>): JsonNode = mapper.readTree(response.body())["data"]
    class MutableClock(initial: Instant) : Clock() {
        private val current = AtomicReference(initial)
        fun set(value: Instant) { current.set(value) }
        override fun getZone(): ZoneId = ZoneOffset.UTC
        override fun withZone(zone: ZoneId): Clock = Clock.fixed(current.get(), zone)
        override fun instant(): Instant = current.get()
    }
    @TestConfiguration class ClockConfig {
        @Bean @Primary fun responseClock() = MutableClock(Instant.parse("2026-10-05T00:00:00Z"))
        // The global sweep must not consume another integration class's retained fixtures.
        // Same real PG16 container, Flyway and JPA validation; only this class's database is separate.
        @Bean fun responseDatasource() = HikariDataSource(HikariConfig().apply {
            jdbcUrl = responseJdbcUrl
            username = postgres.username
            password = postgres.password
            maximumPoolSize = 5
        })
        companion object {
            private val responseJdbcUrl: String by lazy {
                java.sql.DriverManager.getConnection(postgres.jdbcUrl, postgres.username, postgres.password).use { connection ->
                    connection.createStatement().use { it.execute("CREATE DATABASE removal_responses_task7") }
                }
                postgres.jdbcUrl.replace("/quickpitik_test", "/removal_responses_task7")
            }
        }
    }
}
