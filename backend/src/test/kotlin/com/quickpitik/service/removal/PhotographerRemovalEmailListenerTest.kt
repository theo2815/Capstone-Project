package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.*
import com.quickpitik.repository.PhotoRemovalRequestRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertFailsWith

class PhotographerRemovalEmailListenerTest {
    private val messages = mock(PhotographerMessageRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val email = mock(EmailService::class.java)
    private val manager = mock(PlatformTransactionManager::class.java)
    private val now = Instant.parse("2026-11-10T04:00:00Z")
    private val clock = Clock.fixed(now, ZoneOffset.UTC)
    private val owner = User(email = "photog@test.local", passwordHash = "x", name = "P", role = Role.PHOTOGRAPHER)
    private val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Cebu Run",
        requesterContact = "private contact", supportReference = "private ticket", requesterReason = "private reason",
        publicToOwnerReason = "Privacy removal request", creatorId = UUID.randomUUID(), creationCommandId = UUID.randomUUID(), status = RemovalRequestStatus.OPEN)
    private fun message(kind: String = "photo_removal_request") = PhotographerMessage(photographerId = owner.id, kindWire = kind,
        body = "1 photo: Photo ab12cd34.", photoRemovalRequestId = request.id, noticeKey = "k-$kind", createdAt = now.atOffset(ZoneOffset.UTC))
    private val listener = PhotographerRemovalEmailListener(messages, users, requests, email, TransactionTemplate(manager), clock, PhotoRemovalProperties())

    init {
        `when`(manager.getTransaction(org.mockito.ArgumentMatchers.any())).thenReturn(SimpleTransactionStatus())
        `when`(users.findById(owner.id)).thenReturn(Optional.of(owner))
        `when`(requests.findById(request.id)).thenReturn(Optional.of(request))
    }

    @Test fun `a new review notice is claimed, emailed to its photographer and stamped delivered`() {
        val m = message()
        `when`(messages.findById(m.id)).thenReturn(Optional.of(m))
        `when`(messages.claimEmailSend(m.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))).thenReturn(1)
        listener.sendIfPending(m.id)
        verify(email).sendPhotographerRemovalReviewEmail("photog@test.local", "Cebu Run", m.body, request.id)
        verify(messages).stampEmailDelivered(m.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
    }

    @Test fun `an outcome notice uses the outcome email`() {
        val m = message("photo_removal_outcome")
        `when`(messages.findById(m.id)).thenReturn(Optional.of(m))
        `when`(messages.claimEmailSend(m.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))).thenReturn(1)
        listener.sendIfPending(m.id)
        verify(email).sendPhotographerRemovalOutcomeEmail("photog@test.local", "Cebu Run", m.body, request.id)
        verify(email, never()).sendPhotographerRemovalReviewEmail(anyString(), anyString(), anyString(), org.mockito.ArgumentMatchers.any() ?: request.id)
    }

    @Test fun `a claimed notice is never emailed twice`() {
        val m = message()
        `when`(messages.findById(m.id)).thenReturn(Optional.of(m))
        `when`(messages.claimEmailSend(m.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))).thenReturn(1, 0)
        listener.sendIfPending(m.id); listener.sendIfPending(m.id)
        verify(email, times(1)).sendPhotographerRemovalReviewEmail("photog@test.local", "Cebu Run", m.body, request.id)
    }

    @Test fun `a failed send keeps the claim and is not stamped delivered`() {
        val m = message()
        `when`(messages.findById(m.id)).thenReturn(Optional.of(m))
        `when`(messages.claimEmailSend(m.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))).thenReturn(1)
        doThrow(RuntimeException("resend down")).`when`(email).sendPhotographerRemovalReviewEmail("photog@test.local", "Cebu Run", m.body, request.id)
        assertFailsWith<RuntimeException> { listener.sendIfPending(m.id) }
        verify(messages, never()).stampEmailDelivered(m.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
        verify(messages, never()).releaseStaleEmailSend(m.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
    }

    @Test fun `a notice for a missing or non-removal message sends nothing`() {
        val other = message("admin_message")
        `when`(messages.findById(other.id)).thenReturn(Optional.of(other))
        listener.sendIfPending(other.id); listener.sendIfPending(UUID.randomUUID())
        verifyNoInteractions(email)
    }

    @Test fun `the sweep re-sends a stale claim, skips one another instance took, and uses the 24 hour window`() {
        val stale = message(); val taken = message()
        val cutoff = OffsetDateTime.ofInstant(now.minus(java.time.Duration.ofMinutes(10)), ZoneOffset.UTC)
        val since = OffsetDateTime.ofInstant(now.minus(java.time.Duration.ofHours(24)), ZoneOffset.UTC)
        `when`(messages.findStaleEmailClaimIds(cutoff, since, 50)).thenReturn(listOf(stale.id, taken.id))
        `when`(messages.releaseStaleEmailSend(stale.id, cutoff)).thenReturn(1)
        `when`(messages.releaseStaleEmailSend(taken.id, cutoff)).thenReturn(0)
        `when`(messages.findById(stale.id)).thenReturn(Optional.of(stale))
        `when`(messages.claimEmailSend(stale.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))).thenReturn(1)
        listener.retryUnsentEmails()
        verify(email, times(1)).sendPhotographerRemovalReviewEmail("photog@test.local", "Cebu Run", stale.body, request.id)
        verify(messages, never()).claimEmailSend(taken.id, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
    }
}
