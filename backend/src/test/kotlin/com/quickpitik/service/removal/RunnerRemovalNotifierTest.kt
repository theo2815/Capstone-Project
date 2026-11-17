package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.email.EmailService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.websocket.AdminInboxEvent
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.mockito.Mockito.*
import org.springframework.context.ApplicationEventPublisher
import org.springframework.scheduling.annotation.Async
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.event.TransactionPhase
import org.springframework.transaction.event.TransactionalEventListener
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.*

class RunnerRemovalNotifierTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val items = mock(PhotoRemovalItemRepository::class.java)
    private val history = mock(PhotoRemovalHistoryRepository::class.java)
    private val messages = mock(RunnerMessageRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val email = mock(EmailService::class.java)
    private val publisher = mock(ApplicationEventPublisher::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val now = Instant.parse("2026-11-12T04:00:00Z")
    private val at = OffsetDateTime.ofInstant(now, ZoneOffset.UTC)
    private val runner = User(email = "runner@test.local", passwordHash = "x", name = "Runner", role = Role.RUNNER)
    private val notifier = RunnerRemovalNotifier(requests, RunnerPhotoRemovalQueryService(requests, items, history),
        RunnerMessagesService(messages, publisher), messages, users, email, publisher, TransactionTemplate(manager),
        Clock.fixed(now, ZoneOffset.UTC), PhotoRemovalProperties())
    private val stored = linkedMapOf<String, RunnerMessage>()

    private fun <T> anyArg(): T = Mockito.any()

    init {
        `when`(manager.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
        `when`(users.findById(runner.id)).thenReturn(Optional.of(runner))
        `when`(history.findByRequestIdOrderByOccurredAtAscIdAsc(anyArg(), anyArg())).thenReturn(emptyList())
        // The keyed insert as Postgres runs it: one row per notice key.
        `when`(messages.insertKeyedNotice(anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg())).thenAnswer { inv ->
            val key = inv.getArgument<String>(3)
            if (key in stored) 0 else {
                stored[key] = RunnerMessage(id = inv.getArgument(0), runnerId = inv.getArgument(1),
                    kindWire = RunnerMessageKind.PHOTO_REMOVAL_UPDATE.wire, photoRemovalRequestId = inv.getArgument(2),
                    noticeKey = key, title = inv.getArgument(4), body = inv.getArgument(5))
                1
            }
        }
        `when`(messages.findById(anyArg())).thenAnswer { inv -> Optional.ofNullable(stored.values.firstOrNull { it.id == inv.getArgument(0) }) }
    }

    private fun case(source: RemovalRequestSource = RemovalRequestSource.RUNNER, requester: UUID? = runner.id) =
        PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Cebu Run", requesterUserId = requester,
            requesterContact = "runner@test.local", supportReference = "Runner request", requesterReason = "Privacy: secret reason",
            publicToOwnerReason = "Privacy removal request", creatorId = runner.id, creationCommandId = UUID.randomUUID(),
            status = RemovalRequestStatus.OPEN, source = source).also {
            `when`(requests.findById(it.id)).thenReturn(Optional.of(it))
        }

    private fun counts(request: PhotoRemovalRequest, vararg rows: Pair<RemovalItemState, Long>) =
        `when`(items.countStatesByRequestIds(listOf(request.id)))
            .thenReturn(rows.map { RequestStateCount(request.id, it.first, null, it.second) })

    private fun changed(request: PhotoRemovalRequest) = notifier.onRemovalChanged(removalChangedEvent(request.id, null, now))

    private fun noticesStored() = mockingDetails(publisher).invocations
        .mapNotNull { it.arguments.firstOrNull() as? RunnerRemovalNoticeStored }

    @Test fun `one message per status, and a repeated event sends nothing`() {
        val request = case()
        counts(request, RemovalItemState.VERIFIED to 4L)
        changed(request); changed(request)
        assertEquals(listOf("removal:runner:${request.id}:PENDING_ADMIN_REVIEW"), stored.keys.toList())
        assertEquals("4 photos are hidden while we review your request.", stored.values.single().body)
        counts(request, RemovalItemState.REMOVED to 4L)
        changed(request)
        assertEquals("removal:runner:${request.id}:COMPLETED", stored.keys.last())
        assertEquals(2, noticesStored().size)
    }

    @Test fun `no runner notice when the requester is gone`() {
        val request = case(requester = null)
        counts(request, RemovalItemState.VERIFIED to 1L)
        changed(request)
        assertTrue(stored.isEmpty())
        assertTrue(noticesStored().isEmpty())
    }

    @Test fun `support cases send no runner notice`() {
        val request = case(source = RemovalRequestSource.SUPPORT)
        counts(request, RemovalItemState.VERIFIED to 1L)
        changed(request)
        assertTrue(stored.isEmpty())
    }

    @Test fun `other admin inbox events are ignored`() {
        notifier.onRemovalChanged(AdminInboxEvent(mapOf("type" to "dispute_changed", "entityId" to UUID.randomUUID().toString())))
        verifyNoInteractions(requests)
    }

    @Test fun `an email goes out only for a newly stored message`() {
        val request = case()
        counts(request, RemovalItemState.CANDIDATE to 2L)
        changed(request); changed(request)
        assertEquals(1, noticesStored().size)
        verifyNoInteractions(email)
        verify(messages, never()).claimEmailSend(anyArg(), anyArg())
    }

    @Test fun `the email is claimed only after the message row commits`() {
        val listener = RunnerRemovalNotifier::class.java.getMethod("onNoticeStored", RunnerRemovalNoticeStored::class.java)
        assertEquals(TransactionPhase.AFTER_COMMIT, listener.getAnnotation(TransactionalEventListener::class.java)?.phase)
        assertEquals("emailSending", listener.getAnnotation(Async::class.java)?.value)
    }

    @Test fun `a claimed message is never emailed twice`() {
        val request = case()
        counts(request, RemovalItemState.REMOVED to 1L)
        changed(request)
        val m = stored.values.single()
        `when`(messages.claimEmailSend(m.id, at)).thenReturn(1, 0)
        notifier.sendIfPending(m.id); notifier.sendIfPending(m.id)
        verify(email, times(1)).sendRunnerRemovalUpdateEmail("runner@test.local", "Cebu Run", m.title!!, m.body, request.id)
        verify(messages, times(1)).stampEmailDelivered(m.id, at)
    }

    @Test fun `a failed send keeps the claim and is not stamped delivered`() {
        val request = case()
        counts(request, RemovalItemState.REMOVED to 1L)
        changed(request)
        val m = stored.values.single()
        `when`(messages.claimEmailSend(m.id, at)).thenReturn(1)
        doThrow(RuntimeException("resend down")).`when`(email).sendRunnerRemovalUpdateEmail("runner@test.local", "Cebu Run", m.title!!, m.body, request.id)
        assertFailsWith<RuntimeException> { notifier.sendIfPending(m.id) }
        verify(messages, never()).stampEmailDelivered(m.id, at)
    }

    @Test fun `the sweep re-sends a stale claim, skips a delivered one and ignores rows older than 24 hours`() {
        val request = case()
        counts(request, RemovalItemState.REMOVED to 1L)
        changed(request)
        val stale = stored.values.single().id
        val taken = UUID.randomUUID()
        val cutoff = at.minus(Duration.ofMinutes(10))
        // The query owns the delivered and 24-hour filters (RunnerRemovalSchemaIntegrationTest); the sweep passes them.
        `when`(messages.findStaleEmailClaimIds(cutoff, at.minus(Duration.ofHours(24)), 50)).thenReturn(listOf(stale, taken))
        `when`(messages.releaseStaleEmailSend(stale, cutoff)).thenReturn(1)
        `when`(messages.releaseStaleEmailSend(taken, cutoff)).thenReturn(0)
        `when`(messages.claimEmailSend(stale, at)).thenReturn(1)
        notifier.retryUnsentEmails()
        verify(email, times(1)).sendRunnerRemovalUpdateEmail(anyArg(), anyArg(), anyArg(), anyArg(), anyArg())
        verify(messages, never()).claimEmailSend(taken, at)
    }

    @Test fun `the body names no photographer, photo id or score`() {
        val request = case()
        counts(request, RemovalItemState.VERIFIED to 1L)
        changed(request)
        val texts = RunnerRemovalStatus.entries.flatMap { runnerRemovalCopy(it, 3).toList() } +
            stored.values.flatMap { listOf(it.title!!, it.body) }
        texts.forEach { text ->
            assertTrue(text.isNotBlank())
            assertFalse(Regex("[0-9a-f]{8}-[0-9a-f]{4}").containsMatchIn(text), text)
            assertFalse(Regex("0\\.\\d").containsMatchIn(text), text)
            assertFalse(text.contains("photographer", ignoreCase = true), text)
            assertFalse(text.contains("secret reason"), text)
        }
        assertTrue(runnerRemovalCopy(RunnerRemovalStatus.REJECTED, 0).second.contains("within 14 days"))
        assertEquals("1 photo is hidden while we review your request.", runnerRemovalCopy(RunnerRemovalStatus.PENDING_ADMIN_REVIEW, 1).second)
    }
}
