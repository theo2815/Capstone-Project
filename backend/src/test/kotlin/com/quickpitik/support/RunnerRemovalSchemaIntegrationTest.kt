package com.quickpitik.support

import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.removal.PhotoRemovalHistoryRecorder
import com.quickpitik.service.removal.RemovalHistoryActions
import com.quickpitik.service.removal.RemovalHistoryEvent
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.Instant
import java.time.LocalDate
import java.util.UUID
import kotlin.test.*

// RD-1: the runner deletion columns, CHECK values, indexes and repository queries.
class RunnerRemovalSchemaIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var commands: PhotoRemovalCommandRepository
    @Autowired lateinit var history: PhotoRemovalHistoryRepository
    @Autowired lateinit var recorder: PhotoRemovalHistoryRecorder
    @Autowired lateinit var runnerMessages: RunnerMessageRepository
    @Autowired lateinit var photographerMessages: PhotographerMessageRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var tx: TransactionTemplate
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var provider: AiApiClient
    private val now = Instant.parse("2026-11-10T00:00:00Z")
    private lateinit var runner: User
    private lateinit var photographer: User
    private lateinit var event: Event

    @BeforeEach fun fixture() {
        runner = users.saveAndFlush(User(email = "rd1-runner-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Runner", role = Role.RUNNER))
        photographer = users.saveAndFlush(User(email = "rd1-photog-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Photographer", role = Role.PHOTOGRAPHER))
        event = newEvent()
    }

    private fun newEvent() = events.saveAndFlush(Event(slug = "rd1-${UUID.randomUUID()}", name = "Runner removal run",
        date = LocalDate.of(2026, 11, 1), location = "Cebu", status = EventStatus.ACTIVE))

    private fun newPhoto(on: Event = event) = photos.saveAndFlush(Photo(eventId = on.id, photographerId = photographer.id,
        s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal("125")))

    private fun request(on: Event = event, status: RemovalRequestStatus = RemovalRequestStatus.OPEN,
                        source: RemovalRequestSource = RemovalRequestSource.RUNNER,
                        closeReason: RunnerRemovalCloseReason? = null, requester: User = runner) =
        requests.saveAndFlush(PhotoRemovalRequest(eventId = on.id, eventIdSnapshot = on.id, eventNameSnapshot = on.name,
            requesterUserId = requester.id, requesterContact = requester.email, supportReference = "Runner request",
            requesterReason = "PRIVACY", publicToOwnerReason = "Privacy removal request", creatorId = requester.id,
            creationCommandId = UUID.randomUUID(), status = status, source = source, closeReason = closeReason))

    private fun item(of: PhotoRemovalRequest, state: RemovalItemState, decision: RemovalDecision? = null,
                     verifiedBy: UUID? = null, verifiedAt: Instant? = null) =
        items.saveAndFlush(PhotoRemovalItem(requestId = of.id, photoIdSnapshot = newPhoto().id, ownerIdSnapshot = photographer.id,
            state = state, matchSource = RemovalMatchSource.SELFIE, matchScore = 0.995, verificationActorId = verifiedBy,
            verifiedAt = verifiedAt, adminDecision = decision, adminDecisionReason = decision?.let { "Admin decision" }))

    private fun historyRow(of: PhotoRemovalRequest, action: String, actor: RemovalActorType = RemovalActorType.ADMIN) =
        tx.executeWithoutResult {
            recorder.record(RemovalHistoryEvent(requestId = of.id, actorType = actor, actorId = UUID.randomUUID(),
                action = action, occurredAt = now, commandId = UUID.randomUUID()))
        }

    @Test fun `a runner request saves with source, liveness band, consent and risk flags`() {
        val saved = requests.saveAndFlush(PhotoRemovalRequest(eventId = event.id, eventIdSnapshot = event.id,
            eventNameSnapshot = event.name, requesterUserId = runner.id, requesterContact = runner.email,
            supportReference = "Runner request", requesterReason = "Details", publicToOwnerReason = "Privacy removal request",
            creatorId = runner.id, creationCommandId = UUID.randomUUID(), source = RemovalRequestSource.RUNNER,
            reasonCode = "PRIVACY", consentVersion = "1.3", consentAt = now, livenessSessionId = "session-${UUID.randomUUID()}",
            livenessBand = RunnerLivenessBand.UNCERTAIN, livenessConfidence = BigDecimal("72.50"),
            riskFlags = mutableListOf(RunnerRiskFlag.LIVENESS_UNCERTAIN.name, RunnerRiskFlag.GROUP_PHOTOS.name)))
        val loaded = requests.findById(saved.id).orElseThrow()
        assertEquals(RemovalRequestSource.RUNNER, loaded.source)
        assertEquals("PRIVACY", loaded.reasonCode)
        assertEquals("1.3", loaded.consentVersion)
        assertEquals(now, loaded.consentAt)
        assertEquals(saved.livenessSessionId, loaded.livenessSessionId)
        assertEquals(RunnerLivenessBand.UNCERTAIN, loaded.livenessBand)
        assertEquals(BigDecimal("72.50"), loaded.livenessConfidence)
        assertEquals(listOf("LIVENESS_UNCERTAIN", "GROUP_PHOTOS"), loaded.riskFlags)
        assertNull(loaded.closeReason)
        // A support case written without the new fields keeps today's meaning.
        assertEquals(RemovalRequestSource.SUPPORT, requests.findById(request(source = RemovalRequestSource.SUPPORT).id).orElseThrow().source)
    }

    @Test fun `a second open runner request for the same runner and event violates uq_photo_removal_requests_runner_open`() {
        request(status = RemovalRequestStatus.DRAFT)
        val error = assertFailsWith<DataIntegrityViolationException> { request() }
        assertTrue("uq_photo_removal_requests_runner_open" in (error.message ?: ""), error.message)
        // Support cases and other events are not limited.
        request(source = RemovalRequestSource.SUPPORT)
        request(on = newEvent())
    }

    @Test fun `one liveness session backs at most one request`() {
        val session = "session-${UUID.randomUUID()}"
        val first = request()
        jdbc.update("UPDATE photo_removal_requests SET liveness_session_id = ? WHERE id = ?", session, first.id)
        val second = request(on = newEvent())
        assertFailsWith<DataIntegrityViolationException> {
            jdbc.update("UPDATE photo_removal_requests SET liveness_session_id = ? WHERE id = ?", session, second.id)
        }
    }

    @Test fun `a withdrawn request still in RESOLVING frees the slot`() {
        request(status = RemovalRequestStatus.RESOLVING, closeReason = RunnerRemovalCloseReason.WITHDRAWN)
        request()
    }

    @Test fun `a decided request in RESOLVING still holds the slot`() {
        request(status = RemovalRequestStatus.RESOLVING)
        assertFailsWith<DataIntegrityViolationException> { request() }
    }

    @Test fun `email_sent_at and email_delivered_at save on both message tables`() {
        val runnerMessage = runnerMessages.saveAndFlush(RunnerMessage(runnerId = runner.id,
            kindWire = RunnerMessageKind.ADMIN_MESSAGE.wire, body = "Hello"))
        val photographerMessage = photographerMessages.saveAndFlush(PhotographerMessage(photographerId = photographer.id,
            kindWire = PhotographerMessageKind.ADMIN_MESSAGE.wire, body = "Hello"))
        for ((table, id) in listOf("runner_messages" to runnerMessage.id, "photographer_messages" to photographerMessage.id)) {
            assertNull(jdbc.queryForObject("SELECT email_sent_at FROM $table WHERE id = ?", java.sql.Timestamp::class.java, id))
            assertEquals(1, jdbc.update("UPDATE $table SET email_sent_at = now(), email_delivered_at = now() WHERE id = ?", id))
            assertEquals(1, jdbc.queryForObject(
                "SELECT count(*) FROM $table WHERE id = ? AND email_sent_at IS NOT NULL AND email_delivered_at IS NOT NULL",
                Int::class.java, id))
        }
    }

    @Test fun `photographer claimEmailSend claims once`() {
        val m = photographerMessages.saveAndFlush(PhotographerMessage(photographerId = photographer.id,
            kindWire = PhotographerMessageKind.PHOTO_REMOVAL_REQUEST.wire, body = "Hello"))
        val at = java.time.OffsetDateTime.now()
        assertEquals(1, tx.execute { photographerMessages.claimEmailSend(m.id, at) })
        assertEquals(0, tx.execute { photographerMessages.claimEmailSend(m.id, at) })
        assertEquals(1, tx.execute { photographerMessages.stampEmailDelivered(m.id, at) })
    }

    @Test fun `the photographer stale-claim query skips delivered rows and rows older than 24 hours`() {
        fun notice(kind: String = PhotographerMessageKind.PHOTO_REMOVAL_REQUEST.wire) = photographerMessages.saveAndFlush(
            PhotographerMessage(photographerId = photographer.id, kindWire = kind, body = "Hello")).id
        val claimedAt = java.time.OffsetDateTime.now().minusMinutes(30)
        val stale = notice(); val delivered = notice(); val old = notice(); val unclaimed = notice(); val other = notice(PhotographerMessageKind.ADMIN_MESSAGE.wire)
        for (id in listOf(stale, delivered, old, other)) jdbc.update("UPDATE photographer_messages SET email_sent_at = ? WHERE id = ?", claimedAt, id)
        jdbc.update("UPDATE photographer_messages SET email_delivered_at = now() WHERE id = ?", delivered)
        jdbc.update("UPDATE photographer_messages SET created_at = now() - interval '2 days' WHERE id = ?", old)
        val found = photographerMessages.findStaleEmailClaimIds(java.time.OffsetDateTime.now().minusMinutes(10), java.time.OffsetDateTime.now().minusHours(24), 50)
        assertEquals(listOf(stale), found.filter { it in setOf(stale, delivered, old, unclaimed, other) })
        assertEquals(1, tx.execute { photographerMessages.releaseStaleEmailSend(stale, java.time.OffsetDateTime.now().minusMinutes(10)) })
        assertEquals(0, tx.execute { photographerMessages.releaseStaleEmailSend(delivered, java.time.OffsetDateTime.now().minusMinutes(10)) })
    }

    @Test fun `a RUNNER history row and a SUGGESTED action save`() {
        val case = request()
        historyRow(case, RemovalHistoryActions.SUGGESTED, RemovalActorType.RUNNER)
        historyRow(case, RemovalHistoryActions.SUGGESTION_DISMISSED)
        assertEquals(setOf(RemovalHistoryActions.SUGGESTED, RemovalHistoryActions.SUGGESTION_DISMISSED),
            jdbc.queryForList("SELECT action FROM photo_removal_history WHERE request_id = ?", String::class.java, case.id).toSet())
        assertEquals(listOf("RUNNER"), jdbc.queryForList(
            "SELECT actor_type FROM photo_removal_history WHERE request_id = ? AND action = 'SUGGESTED'", String::class.java, case.id))
    }

    @Test fun `a REVIEW_SUGGESTIONS command receipt saves`() {
        val case = request()
        val saved = commands.saveAndFlush(PhotoRemovalCommand(commandId = UUID.randomUUID(), requestId = case.id,
            operation = RemovalCommandOperation.REVIEW_SUGGESTIONS, actorIdSnapshot = UUID.randomUUID(),
            inputHash = "a".repeat(64), claimedAt = now))
        assertEquals(RemovalCommandOperation.REVIEW_SUGGESTIONS, commands.findById(saved.commandId).orElseThrow().operation)
    }

    @Test fun `a photo_removal_update runner message saves`() {
        val case = request()
        val saved = runnerMessages.saveAndFlush(RunnerMessage(runnerId = runner.id,
            kindWire = RunnerMessageKind.PHOTO_REMOVAL_UPDATE.wire, body = "Your request is under review",
            photoRemovalRequestId = case.id))
        assertEquals(RunnerMessageKind.PHOTO_REMOVAL_UPDATE, runnerMessages.findById(saved.id).orElseThrow().kind)
    }

    @Test fun `countOpenRunnerRequests counts only RUNNER rows in DRAFT, OPEN, or RESOLVING with no close reason`() {
        request(on = newEvent(), status = RemovalRequestStatus.DRAFT)
        request(on = newEvent(), status = RemovalRequestStatus.OPEN)
        request(on = newEvent(), status = RemovalRequestStatus.RESOLVING)
        request(on = newEvent(), status = RemovalRequestStatus.RESOLVING, closeReason = RunnerRemovalCloseReason.NO_MATCHES)
        request(on = newEvent(), status = RemovalRequestStatus.RESOLVED)
        request(on = newEvent(), status = RemovalRequestStatus.CANCELLED)
        request(on = newEvent(), source = RemovalRequestSource.SUPPORT)
        assertEquals(3L, requests.countOpenRunnerRequests(runner.id))
    }

    @Test fun `existsOpenRunnerRequest and findOpenRunnerRequestId match the open-index predicate`() {
        assertFalse(requests.existsOpenRunnerRequest(runner.id, event.id))
        assertNull(requests.findOpenRunnerRequestId(runner.id, event.id))
        request(status = RemovalRequestStatus.RESOLVING, closeReason = RunnerRemovalCloseReason.WITHDRAWN)
        request(status = RemovalRequestStatus.RESOLVED)
        request(source = RemovalRequestSource.SUPPORT)
        assertFalse(requests.existsOpenRunnerRequest(runner.id, event.id))
        val open = request(status = RemovalRequestStatus.DRAFT)
        assertTrue(requests.existsOpenRunnerRequest(runner.id, event.id))
        assertEquals(open.id, requests.findOpenRunnerRequestId(runner.id, event.id))
        assertNull(requests.findOpenRunnerRequestId(runner.id, newEvent().id))
    }

    @Test fun `existsRejectedRunnerRequest is true after a RESOLVED case with only RETAINED items and false after a WITHDRAWN one or an undecided admin cancel`() {
        val retained = newEvent()
        item(request(on = retained, status = RemovalRequestStatus.RESOLVED), RemovalItemState.RETAINED, RemovalDecision.RETAIN)
        assertTrue(requests.existsRejectedRunnerRequest(runner.id, retained.id))

        val partlyRemoved = newEvent()
        val mixed = request(on = partlyRemoved, status = RemovalRequestStatus.RESOLVED)
        item(mixed, RemovalItemState.RETAINED, RemovalDecision.RETAIN)
        item(mixed, RemovalItemState.REMOVED, RemovalDecision.REMOVE)
        assertFalse(requests.existsRejectedRunnerRequest(runner.id, partlyRemoved.id))

        val withdrawn = newEvent()
        // Withdrawn after an admin already kept one photo: still not a rejection.
        item(request(on = withdrawn, status = RemovalRequestStatus.CANCELLED, closeReason = RunnerRemovalCloseReason.WITHDRAWN),
            RemovalItemState.RETAINED, RemovalDecision.RETAIN)
        assertFalse(requests.existsRejectedRunnerRequest(runner.id, withdrawn.id))

        val cancelled = newEvent()
        item(request(on = cancelled, status = RemovalRequestStatus.CANCELLED), RemovalItemState.EXCLUDED)
        assertFalse(requests.existsRejectedRunnerRequest(runner.id, cancelled.id))

        // Another runner's rejection and a support case's are not this runner's.
        val other = users.saveAndFlush(User(email = "rd1-other-${UUID.randomUUID()}@test.local", passwordHash = "unused",
            name = "Other runner", role = Role.RUNNER))
        val elsewhere = newEvent()
        item(request(on = elsewhere, status = RemovalRequestStatus.RESOLVED, requester = other), RemovalItemState.RETAINED, RemovalDecision.RETAIN)
        item(request(on = elsewhere, status = RemovalRequestStatus.RESOLVED, source = RemovalRequestSource.SUPPORT),
            RemovalItemState.RETAINED, RemovalDecision.RETAIN)
        assertFalse(requests.existsRejectedRunnerRequest(runner.id, elsewhere.id))
    }

    @Test fun `existsRejectedRunnerRequest is true after a closed case whose suggestions were all dismissed, and false after a demote-only one`() {
        val dismissed = newEvent()
        val case = request(on = dismissed, status = RemovalRequestStatus.CANCELLED)
        item(case, RemovalItemState.EXCLUDED)
        historyRow(case, RemovalHistoryActions.SUGGESTION_DISMISSED)
        assertTrue(requests.existsRejectedRunnerRequest(runner.id, dismissed.id))

        // A dismissal on a case the runner withdrew is not a rejection.
        val withdrawn = newEvent()
        val gone = request(on = withdrawn, status = RemovalRequestStatus.CANCELLED, closeReason = RunnerRemovalCloseReason.WITHDRAWN)
        historyRow(gone, RemovalHistoryActions.SUGGESTION_DISMISSED)
        assertFalse(requests.existsRejectedRunnerRequest(runner.id, withdrawn.id))

        // Demote (a rollback) turns hidden photos back into suggestions; no decision, no dismissal.
        val demoted = newEvent()
        val rollback = request(on = demoted, status = RemovalRequestStatus.CANCELLED)
        item(rollback, RemovalItemState.EXCLUDED)
        historyRow(rollback, RemovalHistoryActions.SUGGESTED)
        assertFalse(requests.existsRejectedRunnerRequest(runner.id, demoted.id))
    }

    @Test fun `countRunnerAutoHidesSince counts only system-verified runner items since the given time, per owner`() {
        val case = request()
        item(case, RemovalItemState.VERIFIED, verifiedAt = now)
        item(case, RemovalItemState.VERIFIED, verifiedAt = now.plusSeconds(60))
        item(case, RemovalItemState.VERIFIED, verifiedAt = now.minusSeconds(60)) // before the window
        item(case, RemovalItemState.VERIFIED, verifiedBy = UUID.randomUUID(), verifiedAt = now) // an admin verified it
        item(case, RemovalItemState.CANDIDATE) // a suggestion
        val support = request(source = RemovalRequestSource.SUPPORT)
        item(support, RemovalItemState.VERIFIED, verifiedAt = now)
        item(request(on = newEvent()), RemovalItemState.VERIFIED, verifiedAt = now) // another event

        val counts = items.countRunnerAutoHidesSince(event.id, listOf(photographer.id, UUID.randomUUID()), now)
        assertEquals(listOf(OwnerCount(photographer.id, 2)), counts)
    }
}
