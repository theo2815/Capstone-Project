package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.removal.*
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionSynchronizationManager
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.LocalDate
import java.util.UUID
import kotlin.test.*

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = ["app.photo-removal.cleanup-interval=PT24H"])
class PhotoRemovalDecisionIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var service: PhotoRemovalService
    @Autowired lateinit var queries: PhotoRemovalQueryService
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var jobs: PhotoErasureJobRepository
    @Autowired lateinit var commands: PhotoRemovalCommandRepository
    @Autowired lateinit var reviews: PhotoRemovalReviewRepository
    @Autowired lateinit var messages: PhotographerMessageRepository
    @Autowired lateinit var history: PhotoRemovalHistoryRepository
    @Autowired lateinit var sweep: PhotoRemovalSweep
    @Autowired lateinit var notifier: PhotoRemovalNotifier
    @Autowired lateinit var recorder: PhotoRemovalHistoryRecorder
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var orderItems: OrderItemRepository
    @Autowired lateinit var tokens: JwtTokenProvider
    @Autowired lateinit var mapper: ObjectMapper
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var jdbc: JdbcTemplate
    @MockitoBean lateinit var storage: StorageService
    @org.springframework.test.context.bean.override.mockito.MockitoSpyBean lateinit var clock: java.time.Clock
    @LocalServerPort var port = 0
    private val client = HttpClient.newHttpClient()
    private lateinit var admin: User
    private lateinit var event: Event

    @BeforeEach fun seed() {
        admin = user(Role.ADMIN)
        event = events.saveAndFlush(Event(slug = "decide-${UUID.randomUUID()}", name = "Synthetic run",
            date = LocalDate.of(2026, 10, 8), location = "Cebu", status = EventStatus.ACTIVE))
        doAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive(), "Storage I/O holds a transaction")
            null
        }.`when`(storage).delete(anyString())
    }

    // Deciding the agreed subset must neither wait for nor decide another owner's item.
    @Test fun oneOwnerApprovalDoesNotDecideAnotherOwnerItems() {
        val ownerA = user(); val ownerB = user()
        val case = verified(listOf(photo(ownerA), photo(ownerB)))
        val first = case.items.single { it.ownerIdSnapshot == ownerA.id }
        service.respond(ownerA.id, case.id, RespondToRemovalCommand(UUID.randomUUID(), case.version, listOf(first.id), PhotographerRemovalResponse.AGREE))
        assertEquals(200, decide(case.id, first.id).statusCode())
        val partial = queries.adminDetail(admin.id, case.id)
        assertEquals(RemovalItemState.REMOVED, partial.items.single { it.id == first.id }.state)
        assertEquals(RemovalItemState.VERIFIED, partial.items.single { it.ownerIdSnapshot == ownerB.id }.state)
        assertNull(partial.resolvedAt)
    }

    // An objection cannot silently become admin approval without a recorded reason.
    @Test fun overrideRequiresReason() {
        val owner = user(); val case = verified(listOf(photo(owner))); val item = case.items.single()
        service.respond(owner.id, case.id, RespondToRemovalCommand(UUID.randomUUID(), case.version, listOf(item.id), PhotographerRemovalResponse.OBJECT, "This is not the requester"))
        assertEquals(400, decide(case.id, item.id, reason = "").statusCode())
        assertEquals(RemovalItemState.VERIFIED, items.findById(item.id).orElseThrow().state)
    }

    // Urgent authority belongs to the admin; it must not manufacture owner consent.
    @Test fun urgentRemovalDoesNotPretendOwnerConsented() {
        val case = verified(listOf(photo(user()))); val item = case.items.single()
        assertEquals(200, decide(case.id, item.id, urgentReason = "Immediate privacy exposure").statusCode())
        val after = items.findById(item.id).orElseThrow()
        assertNull(after.ownerResponse)
        assertEquals(RemovalItemState.REMOVED, after.state)
    }

    // Final photo cleanup alone cannot close a case while its biometric evidence remains.
    @Test fun resolutionWaitsForEvidenceDeletion() {
        val case = verified(listOf(photo(user()))); val item = case.items.single()
        val key = "photo-removal/${case.id}/evidence/synthetic"
        tx.executeWithoutResult {
            val request = requests.findForUpdate(case.id)!!
            request.evidenceKey = key
            request.evidenceContentType = "image/png"
            request.evidenceCleanupState = RemovalEvidenceCleanupState.PRESENT
        }
        doThrow(IllegalStateException("synthetic evidence failure")).`when`(storage).delete(key)
        assertEquals(200, decide(case.id, item.id, urgentReason = "Immediate privacy exposure").statusCode())
        val after = queries.adminDetail(admin.id, case.id)
        assertEquals(RemovalItemState.REMOVED, after.items.single().state)
        assertEquals(RemovalRequestStatus.RESOLVING, after.status)
        assertNull(after.resolvedAt)
        assertTrue(after.evidencePresent)
    }

    // Releasing this case's restriction must leave another verified case's restriction intact.
    @Test fun rejectDoesNotReleaseOtherCase() {
        val photo = photo(user()); val first = verified(listOf(photo)); val second = verified(listOf(photo))
        assertEquals(200, decide(first.id, first.items.single().id, RemovalDecision.RETAIN).statusCode())
        assertEquals(RemovalState.REVIEW, photos.findById(photo.id).orElseThrow().removalState)
        assertEquals(RemovalItemState.VERIFIED, items.findById(second.items.single().id).orElseThrow().state)
    }

    // A missing row with no retained job manifest must never be reported as confirmed erasure.
    @Test fun alreadyRemovedNeedsJobProof() {
        val photo = photo(user()); val case = verified(listOf(photo)); val item = case.items.single()
        jdbc.update("DELETE FROM photos WHERE id = ?", photo.id)
        assertEquals(409, decide(case.id, item.id, urgentReason = "Immediate privacy exposure").statusCode())
        assertFalse(items.findById(item.id).orElseThrow().state == RemovalItemState.REMOVED)
        assertNull(jobs.findByPhotoId(photo.id))
        assertNull(requests.findById(case.id).orElseThrow().resolvedAt)
    }

    @Test fun `one confirmed job resolves overlapping cases without another authorization`() {
        val photo = photo(user()); val first = verified(listOf(photo)); val second = verified(listOf(photo))
        assertEquals(200, decide(first.id, first.items.single().id, urgentReason = "Immediate exposure").statusCode())
        val after = queries.adminDetail(admin.id, second.id)
        assertEquals(RemovalItemState.REMOVED, after.items.single().state)
        assertEquals(RemovalRequestStatus.RESOLVED, after.status)
        assertNull(after.items.single().adminDecision, "The other case must not invent an admin decision")
        assertEquals(1, jobs.findByPhotoId(photo.id)!!.authorizations.size)
    }

    @Test fun `late resource reopens completion and truthful cleanup until the job confirms again`() {
        val photo = photo(user()); val case = verified(listOf(photo))
        assertEquals(200, decide(case.id, case.items.single().id, urgentReason = "Immediate exposure").statusCode())
        val job = jobs.findByPhotoId(photo.id)!!
        tx.executeWithoutResult {
            jobs.lock(job.id)!!.apply { outcome = com.quickpitik.dto.photos.PhotoErasureOutcome.PENDING; completedAt = null; revision++ }
            jdbc.update("UPDATE photos SET purged_at = NULL WHERE id = ?", photo.id)
        }
        val staleRead = queries.adminDetail(admin.id, case.id)
        assertEquals("PENDING", staleRead.items.single().cleanupState)
        assertFalse(staleRead.items.single().state == RemovalItemState.REMOVED)
        assertTrue(sweep.sweepDue(java.time.Instant.now(), 100) > 0)
        assertEquals(RemovalRequestStatus.RESOLVING, requests.findById(case.id).orElseThrow().status)
        assertNull(requests.findById(case.id).orElseThrow().resolvedAt)
    }

    @Test fun `failed cleanup command retries the same authorization and replays its receipt`() {
        val photo = photo(user()); val case = verified(listOf(photo))
        doThrow(IllegalStateException("synthetic storage failure")).`when`(storage).delete(photo.s3Key)
        assertEquals(200, decide(case.id, case.items.single().id, urgentReason = "Immediate exposure").statusCode())
        doNothing().`when`(storage).delete(photo.s3Key)
        val command = mapOf("commandId" to UUID.randomUUID(), "expectedVersion" to queries.adminDetail(admin.id, case.id).version,
            "itemIds" to listOf(case.items.single().id))
        val first = post(case.id, "retry-cleanup", command)
        assertEquals(200, first.statusCode())
        assertEquals(RemovalItemState.REMOVED, items.findById(case.items.single().id).orElseThrow().state)
        assertEquals(mapper.readTree(first.body()), mapper.readTree(post(case.id, "retry-cleanup", command).body()))
        assertEquals(1, jobs.findByPhotoId(photo.id)!!.authorizations.size)
    }

    @Test fun `support update is private audited and replay safe`() {
        val case = verified(listOf(photo(user())))
        val commandId = UUID.randomUUID()
        val body = mapOf("commandId" to commandId, "expectedVersion" to case.version,
            "note" to "Support contacted requester; known external copy https://example.test/copy remains")
        val first = post(case.id, "support-updates", body)
        assertEquals(200, first.statusCode())
        assertEquals(mapper.readTree(first.body()), mapper.readTree(post(case.id, "support-updates", body).body()))
        val saved = history.findByRequestIdAndCommandId(case.id, commandId).single()
        assertEquals("SUPPORT_UPDATED", saved.action)
        assertTrue(saved.reason!!.contains("external copy"))
        assertEquals(409, post(case.id, "support-updates", body + ("note" to "changed content")).statusCode())
    }

    @Test fun `outcome notice survives deletion and contains only that owners results`() {
        val ownerA = user(); val ownerB = user(); val case = verified(listOf(photo(ownerA), photo(ownerB)))
        val item = case.items.single { it.ownerIdSnapshot == ownerA.id }
        assertEquals(200, decide(case.id, item.id, RemovalDecision.RETAIN).statusCode())
        val notice = messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(ownerA.id).singleOrNull { it.kindWire == "photo_removal_outcome" }
        assertNotNull(notice, "Decided owner must receive a durable scoped outcome")
        assertTrue(notice.body.contains(item.id.toString()))
        assertFalse(notice.body.contains(case.items.single { it.ownerIdSnapshot == ownerB.id }.id.toString()))
        assertFalse(notice.body.contains("private"))
        jdbc.update("DELETE FROM photographer_messages WHERE id = ?", notice.id)
        service.reconcile(case.id)
        assertTrue(messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(ownerA.id).none { it.kindWire == "photo_removal_outcome" })
    }

    @Test fun `ownership re-review moves access and restarts existing group clock without changing snapshots`() {
        val ownerA = user(); val ownerB = user(); val moved = photo(ownerA); val existing = photo(ownerB)
        val case = verified(listOf(moved, existing)); val original = case.items.single { it.photoIdSnapshot == moved.id }
        service.respond(ownerA.id, case.id, RespondToRemovalCommand(UUID.randomUUID(), case.version, listOf(original.id), PhotographerRemovalResponse.AGREE))
        val oldReview = reviews.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(case.id, ownerB.id).single()
        jdbc.update("UPDATE photo_removal_reviews SET notified_at = notified_at - interval '1 hour', deadline = deadline - interval '1 hour' WHERE id = ?", oldReview.id)
        jdbc.update("UPDATE photos SET photographer_id = ? WHERE id = ?", ownerB.id, moved.id)
        val cmd = VerifyRemovalCommand(UUID.randomUUID(), queries.adminDetail(admin.id, case.id).version,
            "Support checked current ownership", "Explicitly confirmed new owner", listOf(VerifyRemovalSelection(moved.id, RemovalMatchSource.MANUAL, "Re-reviewed ownership")))
        assertEquals(200, post(case.id, "verify", cmd).statusCode())
        val updated = items.findById(original.id).orElseThrow()
        assertEquals(ownerA.id, updated.ownerIdSnapshot)
        assertEquals(oldReview.id, updated.reviewId)
        assertNull(updated.ownerResponse)
        assertEquals(setOf(moved.id, existing.id), queries.photographerDetail(ownerB.id, case.id).items.map { it.photoIdSnapshot }.toSet())
        assertEquals(404, assertFailsWith<com.quickpitik.exception.ApiException> { queries.photographerDetail(ownerA.id, case.id) }.status.value())
        assertTrue(queries.photographerList(ownerA.id, 0, 100, event.id).items.isEmpty())
        assertEquals(listOf(case.id), queries.photographerList(ownerB.id, 0, 100, event.id).items.map { it.id })
        val group = reviews.findById(oldReview.id).orElseThrow()
        assertTrue(group.deadline!! >= oldReview.deadline!!)
        assertEquals(group.notifiedAt!!.plusSeconds(72 * 3600), group.deadline)
        assertEquals(2, messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(ownerB.id).count { it.kindWire == "photo_removal_request" })
        assertEquals(200, post(case.id, "verify", cmd).statusCode())
        assertEquals(2, messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(ownerB.id).count { it.kindWire == "photo_removal_request" })
    }

    @Test fun `terminal cancellation purges matching receipt candidates and marks replay explicitly`() {
        val case = service.create(admin.id, CreateRemovalRequestCommand(event.id, "support-scrub", "private contact", reason = "private reason", commandId = UUID.randomUUID()))
        val commandId = UUID.randomUUID()
        val match = MatchRemovalCommand(commandId, case.version)
        val hash = commandHash(mapper, match)
        commands.saveAndFlush(PhotoRemovalCommand(commandId, case.id, RemovalCommandOperation.MATCH_CANDIDATES, admin.id,
            hash, state = RemovalCommandState.SUCCEEDED, claimedAt = java.time.Instant.now(), result = mapper.readTree(
                """{"items":[{"matchScore":0.91,"photo":{"imageUrl":"https://example.test/signed-secret"}}],"candidateLimit":50,"possiblyTruncated":false,"requestVersion":0,"evidenceGeneration":1}""")))
        assertFailsWith<org.springframework.dao.DataIntegrityViolationException> {
            jdbc.update("UPDATE photo_removal_commands SET result = sanitize_photo_removal_result(operation, result) WHERE command_id = ?", commandId)
        }
        service.cancel(admin.id, case.id, CancelRemovalCommand(UUID.randomUUID(), case.version, "No confirmed matches"))
        val result = commands.findByCommandId(commandId)!!.result!!
        assertTrue(result["items"].isEmpty, "Transient candidates survived terminal cleanup")
        assertTrue(result["candidatesPurged"].asBoolean())
        assertEquals(hash, commands.findByCommandId(commandId)!!.inputHash)
        val replay = queries.matches(admin.id, case.id, match)
        assertTrue(replay.candidatesPurged); assertTrue(replay.items.isEmpty())
        assertEquals(replay, queries.matches(admin.id, case.id, match))
        assertFailsWith<org.springframework.dao.DataIntegrityViolationException> {
            jdbc.update("UPDATE photo_removal_commands SET result = result || '{\"candidateLimit\":99}'::jsonb WHERE command_id = ?", commandId)
        }
        assertFailsWith<org.springframework.dao.DataIntegrityViolationException> {
            jdbc.update("UPDATE photo_removal_commands SET actor_id_snapshot = ? WHERE command_id = ?", UUID.randomUUID(), commandId)
        }
        assertFalse(result.toString().contains("signed-secret"))
    }

    @Test fun `old response receipt cannot expose transferred item when former owner retains another`() {
        val owner = user(); val next = user(); val moved = photo(owner); val kept = photo(owner)
        val case = verified(listOf(moved, kept))
        val command = RespondToRemovalCommand(UUID.randomUUID(), case.version,
            listOf(case.items.single { it.photoIdSnapshot == kept.id }.id), PhotographerRemovalResponse.AGREE)
        assertEquals(2, service.respond(owner.id, case.id, command).items.size)
        jdbc.update("UPDATE photos SET photographer_id = ? WHERE id = ?", next.id, moved.id)
        service.verify(admin.id, case.id, VerifyRemovalCommand(UUID.randomUUID(), queries.adminDetail(admin.id, case.id).version,
            "Support verified transfer", "Current owner confirmed", listOf(VerifyRemovalSelection(moved.id, RemovalMatchSource.MANUAL, "Reviewed"))))
        assertEquals(listOf(kept.id), queries.photographerDetail(owner.id, case.id).items.map { it.photoIdSnapshot })
        assertEquals(404, assertFailsWith<com.quickpitik.exception.ApiException> {
            service.respond(owner.id, case.id, command)
        }.status.value())
    }

    @Test fun `admin list reflects reopened cleanup proof before reconciliation`() {
        val source = photo(user()); val case = verified(listOf(source))
        assertEquals(200, decide(case.id, case.items.single().id, urgentReason = "Immediate exposure").statusCode())
        tx.executeWithoutResult {
            jobs.lock(jobs.findByPhotoId(source.id)!!.id)!!.apply {
                outcome = com.quickpitik.dto.photos.PhotoErasureOutcome.PENDING; completedAt = null; revision++
            }
        }
        val row = queries.list(admin.id, event.id, null, null, 0, 100).items.single()
        assertEquals(RemovalRequestStatus.RESOLVING, row.status)
        assertNull(row.resolvedAt)
        assertEquals(0L, row.phaseCounts.removed)
        assertEquals(1L, row.phaseCounts.removalAuthorized)
    }

    @Test fun `outcome retries never acknowledge a partial decision version at batch boundary`() {
        val owner = user(); val case = verified((1..100).map { photo(owner) })
        val reviewId = case.items.first().reviewId!!
        val now = java.time.Instant.now()
        tx.executeWithoutResult {
            fun event(item: PhotoRemovalItemDto, version: String, at: java.time.Instant) = RemovalHistoryEvent(case.id,
                item.id, reviewId, item.photoIdSnapshot, RemovalActorType.ADMIN, admin.id,
                RemovalHistoryActions.ADMIN_DECIDED, toState = "RETAINED", occurredAt = at,
                commandId = UUID.randomUUID(), metadata = mapOf("decisionVersion" to version))
            recorder.record(event(case.items.first(), "1000", now))
            case.items.forEach { recorder.record(event(it, "1001", now.plusSeconds(1))) }
        }
        notifier.notifyReview(reviewId)
        notifier.notifyReview(reviewId)
        val notice = messages.findByNoticeKey("removal:review:$reviewId:outcome:1001")!!
        case.items.forEach { assertTrue(notice.body.contains(it.id.toString()), "Acknowledged notice omitted ${it.id}") }
    }

    @Test fun `restricted exception removal reopens the case while cleanup is pending`() {
        val source = photo(user()); val case = verified(listOf(source)); val item = case.items.single()
        val due = java.time.Instant.now().plusSeconds(3600)
        assertEquals(200, post(case.id, "decisions", DecideRemovalCommand(UUID.randomUUID(), case.version,
            listOf(RemovalItemDecision(item.id, RemovalDecision.RETAIN_RESTRICTED, "Temporary justified exception", due)))).statusCode())
        assertEquals(RemovalRequestStatus.RESOLVED, requests.findById(case.id).orElseThrow().status)
        assertEquals(RemovalState.REVIEW, photos.findById(source.id).orElseThrow().removalState)
        assertTrue(sweep.sweepDue(due.plusSeconds(1), 100) > 0)
        assertTrue(queries.list(admin.id, event.id, null, true, 0, 100).items.isEmpty()) // Clock has not reached due yet.
        doThrow(IllegalStateException("synthetic cleanup failure")).`when`(storage).delete(source.s3Key)
        assertEquals(200, decide(case.id, item.id, urgentReason = "Exception no longer justified").statusCode())
        assertEquals(RemovalRequestStatus.RESOLVING, requests.findById(case.id).orElseThrow().status)
        assertNull(requests.findById(case.id).orElseThrow().resolvedAt)
    }

    // Verification-only: the existing atomic decision path must support the promised batch cap.
    @Test fun `hundred item mixed owner decision is atomic audited and replay safe`() {
        val a = user(); val b = user(); val case = verified((1..100).map { photo(if (it % 2 == 0) a else b) })
        val command = DecideRemovalCommand(UUID.randomUUID(), case.version,
            case.items.map { RemovalItemDecision(it.id, RemovalDecision.REMOVE, "Support verified removal") }, "Immediate exposure")
        val invalid = command.copy(decisions = command.decisions.dropLast(1) + command.decisions.last().copy(reason = ""))
        assertEquals(400, post(case.id, "decisions", invalid).statusCode())
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).all { it.state == RemovalItemState.VERIFIED })
        assertNull(commands.findByCommandId(command.commandId))
        assertTrue(case.items.all { jobs.findByPhotoId(it.photoIdSnapshot) == null })
        val first = post(case.id, "decisions", command)
        assertEquals(200, first.statusCode())
        assertEquals(mapper.readTree(first.body()), mapper.readTree(post(case.id, "decisions", command).body()))
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).all { it.state == RemovalItemState.REMOVED })
        val audit = history.findByRequestIdAndCommandId(case.id, command.commandId)
        assertEquals(100, audit.size)
        assertEquals(100, audit.map { it.metadata.getValue("decisionLogId") }.toSet().size)
        assertTrue(case.items.all { jobs.findByPhotoId(it.photoIdSnapshot)!!.authorizations.size == 1 })
    }

    @Test fun `guest support completion binds server derived notice identity and preserves purchase`() {
        val source = photo(user()); val case = verified(listOf(source))
        val order = orders.saveAndFlush(Order(eventId = event.id, recipientEmail = "guest@test.local",
            paymentMethodWire = "gcash", totalPhp = java.math.BigDecimal("50.00"), paidAt = java.time.OffsetDateTime.now()))
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, source.id), java.math.BigDecimal("50.00")))
        assertEquals(200, decide(case.id, case.items.single().id, urgentReason = "Immediate exposure").statusCode())
        val notice = queries.adminDetail(admin.id, case.id).guestBuyerRemovalNotices.single()
        assertEquals("PENDING", notice.status)
        val other = verified(listOf(photo(user())))
        assertEquals(404, post(other.id, "support-updates", RemovalSupportUpdateCommand(UUID.randomUUID(), other.version,
            "Wrong case notice", notice.noticeId)).statusCode())
        val command = RemovalSupportUpdateCommand(UUID.randomUUID(), queries.adminDetail(admin.id, case.id).version,
            "Support reached guest buyer", notice.noticeId)
        assertEquals(200, post(case.id, "support-updates", command).statusCode())
        val saved = history.findByRequestIdAndCommandId(case.id, command.commandId).single()
        assertEquals(mapOf("noticeId" to notice.noticeId.toString(), "orderId" to order.id.toString(), "photoId" to source.id.toString()), saved.metadata)
        assertEquals("COMPLETED", queries.adminDetail(admin.id, case.id).guestBuyerRemovalNotices.single().status)
        assertEquals(java.math.BigDecimal("50.00"), orders.findById(order.id).orElseThrow().totalPhp)
        assertEquals(java.math.BigDecimal("50.00"), orderItems.findById(OrderItemId(order.id, source.id)).orElseThrow().pricePhpAtPurchase)
    }

    @Test fun `admin status and action filters include reopened cleanup and exclude resolved`() {
        val source = photo(user()); val case = verified(listOf(source))
        assertEquals(200, decide(case.id, case.items.single().id, urgentReason = "Immediate exposure").statusCode())
        tx.executeWithoutResult { jobs.lock(jobs.findByPhotoId(source.id)!!.id)!!.apply {
            outcome = com.quickpitik.dto.photos.PhotoErasureOutcome.PENDING; completedAt = null; revision++
        } }
        assertEquals(listOf(case.id), queries.list(admin.id, event.id, RemovalRequestStatus.RESOLVING, true, 0, 100).items.map { it.id })
        assertTrue(queries.list(admin.id, event.id, RemovalRequestStatus.RESOLVED, null, 0, 100).items.isEmpty())
    }

    @Test fun `owner filtered list and other owner case status agree with reopened cleanup`() {
        val owner = user(); val other = user(); val source = photo(owner); val kept = photo(other)
        val case = verified(listOf(source, kept))
        assertEquals(200, post(case.id, "decisions", DecideRemovalCommand(UUID.randomUUID(), case.version,
            case.items.map { RemovalItemDecision(it.id, RemovalDecision.REMOVE, "Reviewed") }, "Immediate exposure")).statusCode())
        tx.executeWithoutResult { jobs.lock(jobs.findByPhotoId(source.id)!!.id)!!.apply {
            outcome = com.quickpitik.dto.photos.PhotoErasureOutcome.PENDING; completedAt = null; revision++
        } }
        assertEquals(listOf(case.id), queries.photographerList(owner.id, 0, 100, event.id, RemovalRequestStatus.RESOLVING).items.map { it.id })
        assertTrue(queries.photographerList(owner.id, 0, 100, event.id, RemovalRequestStatus.RESOLVED).items.isEmpty())
        val otherDetail = queries.photographerDetail(other.id, case.id)
        assertEquals(RemovalRequestStatus.RESOLVING, otherDetail.status)
        assertEquals(listOf(kept.id), otherDetail.items.map { it.photoIdSnapshot })
        assertEquals(listOf(case.id), queries.photographerList(other.id, 0, 100, event.id, RemovalRequestStatus.RESOLVING).items.map { it.id })
    }

    @Test fun `terminal sanitizer preserves selected audit while clearing both signed photo shapes`() {
        val raw = """{"items":[{"matchScore":0.91,"photo":{"photo":{"imageUrl":"https://test/signed","cleanUrl":"https://test/clean","downloadUrl":"https://test/get"}}},{"photo":{"imageUrl":"https://test/owner","id":"selected"}}],"version":7}"""
        val cleaned = mapper.readTree(jdbc.queryForObject("SELECT sanitize_photo_removal_result('VERIFY', ?::jsonb)::text", String::class.java, raw))
        assertEquals(0.91, cleaned["items"][0]["matchScore"].asDouble())
        assertEquals(7, cleaned["version"].asInt())
        assertTrue(cleaned["items"][0]["photo"]["photo"]["imageUrl"].isNull)
        assertTrue(cleaned["items"][0]["photo"]["photo"]["cleanUrl"].isNull)
        assertTrue(cleaned["items"][0]["photo"]["photo"]["downloadUrl"].isNull)
        assertTrue(cleaned["items"][1]["photo"]["imageUrl"].isNull)
        assertEquals("selected", cleaned["items"][1]["photo"]["id"].asText())
    }

    @Test fun `concurrent same decision command creates one authorization log and guest notice`() {
        val source = photo(user()); val case = verified(listOf(source)); val item = case.items.single()
        val order = orders.saveAndFlush(Order(eventId = event.id, recipientEmail = "raced-guest@test.local",
            paymentMethodWire = "gcash", totalPhp = BigDecimal("50.00"), paidAt = java.time.OffsetDateTime.now()))
        orderItems.saveAndFlush(OrderItem(OrderItemId(order.id, source.id), BigDecimal("50.00")))
        val command = DecideRemovalCommand(UUID.randomUUID(), case.version,
            listOf(RemovalItemDecision(item.id, RemovalDecision.REMOVE, "Reviewed")), "Immediate exposure")
        val results = race(listOf(command, command)) { post(case.id, "decisions", it) }
        assertEquals(listOf(200, 200), results.map { it.statusCode() })
        assertEquals(mapper.readTree(results[0].body()), mapper.readTree(results[1].body()))
        assertEquals(1, history.findByRequestIdAndCommandId(case.id, command.commandId).size)
        assertEquals(1, jobs.findByPhotoId(source.id)!!.authorizations.size)
        assertEquals(1, queries.adminDetail(admin.id, case.id).guestBuyerRemovalNotices.size)
        assertEquals(BigDecimal("50.00"), orders.findById(order.id).orElseThrow().totalPhp)
        assertEquals(RemovalState.REMOVED, photos.findById(source.id).orElseThrow().removalState)
    }

    @Test fun `competing stale admin decisions cannot release another case restriction`() {
        val source = photo(user()); val case = verified(listOf(source)); verified(listOf(source))
        val commands = listOf(RemovalDecision.REMOVE, RemovalDecision.RETAIN).map {
            DecideRemovalCommand(UUID.randomUUID(), case.version,
                listOf(RemovalItemDecision(case.items.single().id, it, "Reviewed")), "Immediate exposure")
        }
        val results = race(commands) { post(case.id, "decisions", it) }
        assertEquals(listOf(200, 409), results.map { it.statusCode() }.sorted())
        assertEquals(1, commands.sumOf { history.findByRequestIdAndCommandId(case.id, it.commandId).size })
        assertTrue(photos.findById(source.id).orElseThrow().removalState in setOf(RemovalState.REVIEW, RemovalState.REMOVED))
        assertTrue((jobs.findByPhotoId(source.id)?.authorizations?.size ?: 0) <= 1)
    }

    @Test fun `failed owner outcome rolls back its inbox only and retry respects durable backoff`() {
        val a = user(); val b = user(); val case = verified(listOf(photo(a), photo(b)))
        jdbc.execute("ALTER TABLE photographer_messages ADD CONSTRAINT task11_outcome_failure CHECK (photographer_id <> '${a.id}'::uuid OR kind <> 'photo_removal_outcome') NOT VALID")
        try {
            assertEquals(200, post(case.id, "decisions", DecideRemovalCommand(UUID.randomUUID(), case.version,
                case.items.map { RemovalItemDecision(it.id, RemovalDecision.RETAIN, "Reviewed") })).statusCode())
            assertEquals(RemovalRequestStatus.RESOLVED, requests.findById(case.id).orElseThrow().status)
            assertTrue(items.findByRequestIdOrderByIdAsc(case.id).all { it.state == RemovalItemState.RETAINED })
            assertTrue(messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(a.id).none { it.kindWire == "photo_removal_outcome" })
            assertEquals(1, messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(b.id).count { it.kindWire == "photo_removal_outcome" })
            assertTrue(reviews.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(case.id, a.id).single().notificationNextAttemptAt!! > java.time.Instant.now())
        } finally { jdbc.execute("ALTER TABLE photographer_messages DROP CONSTRAINT task11_outcome_failure") }
        notifier.retryPending()
        assertTrue(messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(a.id).none { it.kindWire == "photo_removal_outcome" })
        val review = reviews.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(case.id, a.id).single()
        jdbc.update("UPDATE photo_removal_reviews SET notification_next_attempt_at = now() - interval '1 second' WHERE id = ?", review.id)
        notifier.retryPending()
        assertEquals(1, messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(a.id).count { it.kindWire == "photo_removal_outcome" })
        assertEquals(1, messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(b.id).count { it.kindWire == "photo_removal_outcome" })
    }

    @Test fun `restricted transfer gets fresh durable notice even at equal decision timestamps`() {
        val fixture = restrictedCase(); reverifyRestricted(fixture)
        val review = reviews.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(fixture.case.id, fixture.newOwner.id).single()
        assertNotNull(review.notifiedAt, "Fresh retained-item review must be delivered")
        assertEquals(fixture.now.plusSeconds(72 * 3600), review.deadline)
        assertEquals(2, messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(fixture.newOwner.id).count { it.kindWire == "photo_removal_request" })
        val item = items.findById(fixture.moved.id).orElseThrow()
        assertEquals(item.adminDecidedAt, item.verifiedAt, "Equal Clock timestamps must not erase ordered re-review provenance")
        assertEquals(RemovalItemState.RETAINED_RESTRICTED, item.state)
        assertEquals(RemovalDecision.RETAIN_RESTRICTED, item.adminDecision)
        assertEquals(RemovalState.REVIEW, photos.findById(item.photoIdSnapshot).orElseThrow().removalState)
    }

    @Test fun `fresh restricted response is scoped actionable and preserves existing retention decision`() {
        val fixture = restrictedCase(); reverifyRestricted(fixture)
        val before = queries.photographerDetail(fixture.newOwner.id, fixture.case.id)
        assertEquals(1, before.reviews.single().responses.unanswered)
        assertEquals(listOf(fixture.case.id), queries.photographerList(fixture.newOwner.id, 0, 100, event.id, needsAction = true).items.map { it.id })
        val itemJson = mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(before.items.single { it.id == fixture.moved.id })
        assertEquals(true, itemJson["ownerReviewPending"]?.asBoolean())
        val current = queries.adminDetail(admin.id, fixture.case.id)
        assertEquals(1, mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(current.phaseCounts)["pendingOwnerReviews"]?.asInt())
        val command = RespondToRemovalCommand(UUID.randomUUID(), current.version, listOf(fixture.moved.id), PhotographerRemovalResponse.AGREE)
        assertEquals(409, respond(fixture.newOwner, fixture.case.id, command.copy(itemIds = listOf(fixture.moved.id, fixture.target.id))).statusCode())
        assertNull(items.findById(fixture.moved.id).orElseThrow().ownerResponse)
        val first = respond(fixture.newOwner, fixture.case.id, command)
        assertEquals(200, first.statusCode())
        assertEquals(mapper.readTree(first.body()), mapper.readTree(respond(fixture.newOwner, fixture.case.id, command).body()))
        assertEquals(RemovalDecision.RETAIN_RESTRICTED, items.findById(fixture.moved.id).orElseThrow().adminDecision)
        assertEquals(RemovalRequestStatus.OPEN, requests.findById(fixture.case.id).orElseThrow().status)
        assertTrue(queries.photographerList(fixture.newOwner.id, 0, 100, event.id, needsAction = true).items.isEmpty())
        assertEquals(404, respond(fixture.oldOwner, fixture.case.id, fixture.oldResponse).statusCode())
    }

    @Test fun `pending restricted review keeps case open while unrelated cleanup and overdue progress`() {
        val fixture = restrictedCase(); reverifyRestricted(fixture)
        assertEquals(200, decide(fixture.case.id, fixture.other.id, urgentReason = "Immediate unrelated exposure").statusCode())
        assertEquals(RemovalItemState.REMOVED, items.findById(fixture.other.id).orElseThrow().state)
        assertEquals(RemovalRequestStatus.OPEN, requests.findById(fixture.case.id).orElseThrow().status)
        assertNull(requests.findById(fixture.case.id).orElseThrow().resolvedAt)
        val later = fixture.now.plusSeconds(73 * 3600)
        doReturn(later).`when`(clock).instant()
        assertTrue(sweep.sweepDue(later, 100) > 0)
        service.reconcile(fixture.case.id)
        assertTrue(reviews.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(fixture.case.id, fixture.newOwner.id).single().overdue)
        assertEquals(RemovalRequestStatus.OPEN, requests.findById(fixture.case.id).orElseThrow().status)
        val selected = items.findById(fixture.moved.id).orElseThrow()
        assertNull(selected.ownerResponse)
        assertEquals(RemovalDecision.RETAIN_RESTRICTED, selected.adminDecision)
        assertEquals(200, post(fixture.case.id, "decisions", DecideRemovalCommand(UUID.randomUUID(), queries.adminDetail(admin.id, fixture.case.id).version,
            listOf(RemovalItemDecision(fixture.moved.id, RemovalDecision.RETAIN_RESTRICTED, "Explicitly reconfirmed exception", later.plusSeconds(3600))))).statusCode())
        assertEquals(RemovalRequestStatus.RESOLVED, requests.findById(fixture.case.id).orElseThrow().status)
    }

    @Test fun `new explicit decision completes fresh review by version even with identical Clock time`() {
        val fixture = restrictedCase(); reverifyRestricted(fixture)
        val command = RespondToRemovalCommand(UUID.randomUUID(), queries.adminDetail(admin.id, fixture.case.id).version,
            listOf(fixture.moved.id), PhotographerRemovalResponse.OBJECT, "Please reconsider")
        assertEquals(200, respond(fixture.newOwner, fixture.case.id, command).statusCode())
        assertEquals(200, post(fixture.case.id, "decisions", DecideRemovalCommand(UUID.randomUUID(), queries.adminDetail(admin.id, fixture.case.id).version,
            listOf(RemovalItemDecision(fixture.moved.id, RemovalDecision.RETAIN_RESTRICTED, "Explicit confirmation", fixture.now.plusSeconds(3600))))).statusCode())
        val detail = queries.photographerDetail(fixture.newOwner.id, fixture.case.id)
        assertEquals(RemovalRequestStatus.OPEN, detail.status, "Other undecided owner keeps the case open")
        assertEquals(false, mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(detail.items.single { it.id == fixture.moved.id })["ownerReviewPending"]?.asBoolean())
        assertEquals(409, respond(fixture.newOwner, fixture.case.id, command.copy(commandId = UUID.randomUUID(), expectedVersion = detail.version)).statusCode())
        assertEquals(200, respond(fixture.newOwner, fixture.case.id, command).statusCode(), "Existing receipt stays replayable without a new reply")
        val item = items.findById(fixture.moved.id).orElseThrow()
        assertEquals(item.adminDecidedAt, item.verifiedAt)
        assertEquals(RemovalItemState.RETAINED_RESTRICTED, item.state)
    }

    @Test fun `restricted fresh notice remains due after insertion failure and uses stored notice time`() {
        val fixture = restrictedCase()
        jdbc.execute("ALTER TABLE photographer_messages ADD CONSTRAINT task11_review_failure CHECK (photographer_id <> '${fixture.newOwner.id}'::uuid OR kind <> 'photo_removal_request') NOT VALID")
        try { reverifyRestricted(fixture) }
        finally { jdbc.execute("ALTER TABLE photographer_messages DROP CONSTRAINT task11_review_failure") }
        val review = reviews.findByRequestIdAndOwnerIdSnapshotOrderByIdAsc(fixture.case.id, fixture.newOwner.id).single()
        assertNull(review.notifiedAt); assertNull(review.deadline)
        assertEquals(fixture.now.plusSeconds(300), review.notificationNextAttemptAt)
        notifier.retryPending()
        assertNull(reviews.findById(review.id).orElseThrow().notifiedAt)
        val later = fixture.now.plusSeconds(301)
        doReturn(later).`when`(clock).instant()
        notifier.retryPending()
        val delivered = reviews.findById(review.id).orElseThrow()
        assertEquals(later, delivered.notifiedAt)
        assertEquals(later.plusSeconds(72 * 3600), delivered.deadline)
        assertEquals(2, messages.findByPhotographerIdOrderByCreatedAtDescIdAsc(fixture.newOwner.id).count { it.kindWire == "photo_removal_request" })
    }

    @Test fun `reopened unrelated cleanup cannot close fresh owner response access`() {
        val fixture = restrictedCase(); reverifyRestricted(fixture)
        assertEquals(200, decide(fixture.case.id, fixture.other.id, urgentReason = "Immediate unrelated exposure").statusCode())
        tx.executeWithoutResult { jobs.lock(jobs.findByPhotoId(fixture.other.photoIdSnapshot)!!.id)!!.apply {
            outcome = com.quickpitik.dto.photos.PhotoErasureOutcome.PENDING; completedAt = null; revision++
        } }
        service.reconcile(fixture.case.id)
        assertEquals(RemovalRequestStatus.OPEN, requests.findById(fixture.case.id).orElseThrow().status)
        assertEquals(RemovalItemState.REMOVAL_AUTHORIZED, items.findById(fixture.other.id).orElseThrow().state)
        assertEquals(200, respond(fixture.newOwner, fixture.case.id, RespondToRemovalCommand(UUID.randomUUID(),
            queries.adminDetail(admin.id, fixture.case.id).version, listOf(fixture.moved.id), PhotographerRemovalResponse.AGREE)).statusCode())
    }

    private data class RestrictedFixture(val case: PhotoRemovalRequestDto, val oldOwner: User, val newOwner: User,
        val moved: PhotoRemovalItemDto, val target: PhotoRemovalItemDto, val other: PhotoRemovalItemDto,
        val oldResponse: RespondToRemovalCommand, val now: java.time.Instant)

    private fun restrictedCase(): RestrictedFixture {
        val now = java.time.Instant.parse("2026-10-08T00:00:00Z")
        doReturn(now).`when`(clock).instant()
        val oldOwner = user(); val newOwner = user(); val otherOwner = user()
        val case = verified(listOf(photo(oldOwner), photo(newOwner), photo(otherOwner)))
        val moved = case.items.single { it.ownerIdSnapshot == oldOwner.id }
        val target = case.items.single { it.ownerIdSnapshot == newOwner.id }
        val other = case.items.single { it.ownerIdSnapshot == otherOwner.id }
        val response = RespondToRemovalCommand(UUID.randomUUID(), case.version, listOf(moved.id), PhotographerRemovalResponse.AGREE)
        service.respond(oldOwner.id, case.id, response)
        service.decide(admin.id, case.id, DecideRemovalCommand(UUID.randomUUID(), queries.adminDetail(admin.id, case.id).version,
            listOf(moved, target).map { RemovalItemDecision(it.id, RemovalDecision.RETAIN_RESTRICTED, "Initial exception", now.plusSeconds(3600)) }))
        return RestrictedFixture(case, oldOwner, newOwner, moved, target, other, response, now)
    }

    private fun reverifyRestricted(fixture: RestrictedFixture) {
        jdbc.update("UPDATE photos SET photographer_id = ? WHERE id = ?", fixture.newOwner.id, fixture.moved.photoIdSnapshot)
        service.verify(admin.id, fixture.case.id, VerifyRemovalCommand(UUID.randomUUID(), queries.adminDetail(admin.id, fixture.case.id).version,
            "Support verified ownership transfer", "Explicit fresh review preserving retention", listOf(VerifyRemovalSelection(fixture.moved.photoIdSnapshot, RemovalMatchSource.MANUAL, "Current owner confirmed"))))
    }

    private fun respond(owner: User, requestId: UUID, command: RespondToRemovalCommand): HttpResponse<String> = client.send(HttpRequest.newBuilder(
        URI.create("http://localhost:$port/api/v1/me/photographer/photo-removal-requests/$requestId/responses"))
        .header("Authorization", "Bearer ${tokens.createAccessToken(owner)}").header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(command))).build(), HttpResponse.BodyHandlers.ofString())

    private fun <T, R> race(values: List<T>, action: (T) -> R): List<R> {
        val pool = java.util.concurrent.Executors.newFixedThreadPool(2)
        val ready = java.util.concurrent.CountDownLatch(2)
        val start = java.util.concurrent.CountDownLatch(1)
        try {
            val futures = values.map { value -> pool.submit<R> { ready.countDown(); start.await(); action(value) } }
            assertTrue(ready.await(10, java.util.concurrent.TimeUnit.SECONDS)); start.countDown()
            return futures.map { it.get(30, java.util.concurrent.TimeUnit.SECONDS) }
        } finally { start.countDown(); pool.shutdownNow() }
    }

    private fun decide(requestId: UUID, itemId: UUID, decision: RemovalDecision = RemovalDecision.REMOVE,
        reason: String = "Support reviewed the verified request", urgentReason: String? = null): HttpResponse<String> =
        post(requestId, "decisions", DecideRemovalCommand(UUID.randomUUID(), queries.adminDetail(admin.id, requestId).version,
            listOf(RemovalItemDecision(itemId, decision, reason)), urgentReason))

    private fun post(requestId: UUID, route: String, body: Any): HttpResponse<String> = client.send(HttpRequest.newBuilder(
        URI.create("http://localhost:$port/api/v1/admin/photo-removal-requests/$requestId/$route"))
        .header("Authorization", "Bearer ${tokens.createAccessToken(admin)}").header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body))).build(), HttpResponse.BodyHandlers.ofString())

    private fun verified(selected: List<Photo>): PhotoRemovalRequestDto {
        val case = service.create(admin.id, CreateRemovalRequestCommand(event.id, "support-${UUID.randomUUID()}", "private contact",
            reason = "private reason", commandId = UUID.randomUUID()))
        service.verify(admin.id, case.id, VerifyRemovalCommand(UUID.randomUUID(), case.version, "Support identity check", "Private verification",
            selected.map { VerifyRemovalSelection(it.id, RemovalMatchSource.MANUAL, "Visually confirmed") }))
        return queries.adminDetail(admin.id, case.id)
    }
    private fun user(role: Role = Role.PHOTOGRAPHER) = users.saveAndFlush(User(email = "decide-${UUID.randomUUID()}@test.local",
        passwordHash = "unused", name = "Synthetic $role", role = role))
    private fun photo(owner: User) = photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id,
        s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal("50")))
}
