package com.quickpitik.service.relevance

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.Test
import org.mockito.Mockito.*
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertFailsWith

// RV-3.1 (A.5, P20): the e-mail of a relevance note goes out on the claimed-send path of removal notices.
class RelevanceNoticeEmailListenerTest {
    private val messages = mock(PhotographerMessageRepository::class.java)
    private val users = mock(UserRepository::class.java)
    private val email = mock(EmailService::class.java)
    private val now = Instant.parse("2026-11-17T04:00:00Z")
    private val at = OffsetDateTime.ofInstant(now, ZoneOffset.UTC)
    private val owner = User(email = "photog@test.local", passwordHash = "x", name = "Pia", role = Role.PHOTOGRAPHER)
    private val caseId = UUID.randomUUID()
    private val listener = RelevanceNoticeEmailListener(messages, users, email, testTransactionTemplate(),
        Clock.fixed(now, ZoneOffset.UTC), PhotoRemovalProperties())

    private fun message(note: String = "verdict", kind: String = "relevance_notice") = PhotographerMessage(
        photographerId = owner.id, kindWire = kind, title = "Your photos in Cebu Run are off sale", body = "The body.",
        noticeKey = "relevance:$caseId:$note", relevanceCaseId = caseId, createdAt = at,
    ).also { `when`(messages.findById(it.id)).thenReturn(Optional.of(it)) }

    init {
        `when`(users.findById(owner.id)).thenReturn(Optional.of(owner))
    }

    @Test
    fun `a new note is claimed, e-mailed in its frame and stamped delivered`() {
        val verdict = message()
        val blurry = message("blurry")
        `when`(messages.claimEmailSend(verdict.id, at)).thenReturn(1)
        `when`(messages.claimEmailSend(blurry.id, at)).thenReturn(1)

        listener.sendIfPending(verdict.id)
        listener.sendIfPending(blurry.id)

        // The appeal footer goes only with the notes that start the appeal clock (V2-T21).
        verify(email).sendRelevanceNotice("photog@test.local", "Pia", verdict.title!!, "The body.", caseId, true)
        verify(email).sendRelevanceNotice("photog@test.local", "Pia", blurry.title!!, "The body.", caseId, false)
        verify(messages).stampEmailDelivered(verdict.id, at)
    }

    @Test
    fun `a claimed note is never e-mailed twice and other kinds are not sent`() {
        val m = message()
        `when`(messages.claimEmailSend(m.id, at)).thenReturn(1, 0)
        listener.sendIfPending(m.id); listener.sendIfPending(m.id)
        listener.sendIfPending(message(kind = "photo_removal_request").id)
        verify(email, times(1)).sendRelevanceNotice(anyString(), anyString(), anyString(), anyString(), any() ?: caseId, anyBoolean())
    }

    @Test
    fun `a failed e-mail is sent again by the retry sweep`() {
        val m = message()
        `when`(messages.claimEmailSend(m.id, at)).thenReturn(1)
        doThrow(RuntimeException("resend down")).doNothing().`when`(email)
            .sendRelevanceNotice("photog@test.local", "Pia", m.title!!, "The body.", caseId, true)
        assertFailsWith<RuntimeException> { listener.sendIfPending(m.id) }
        verify(messages, never()).stampEmailDelivered(m.id, at)

        // The claim stays dated; the sweep releases a stale one and sends again.
        val cutoff = at.minus(PhotoRemovalProperties().runner.emailClaimStaleAfter)
        `when`(messages.findStaleRelevanceEmailClaimIds(cutoff, at.minus(Duration.ofHours(24)), 50)).thenReturn(listOf(m.id))
        `when`(messages.releaseStaleEmailSend(m.id, cutoff)).thenReturn(1)
        listener.retryUnsentEmails()

        verify(email, times(2)).sendRelevanceNotice("photog@test.local", "Pia", m.title!!, "The body.", caseId, true)
        verify(messages).stampEmailDelivered(m.id, at)
    }
}
