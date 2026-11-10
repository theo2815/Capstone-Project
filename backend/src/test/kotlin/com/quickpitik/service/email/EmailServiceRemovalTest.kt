package com.quickpitik.service.email

import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.dto.email.ResendSendEmailResponse
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class EmailServiceRemovalTest {
    private val resend = Mockito.mock(ResendClient::class.java)
    private val requestId = UUID.randomUUID()

    private fun service() = EmailService(resend, ResendProperties(apiKey = "re_live_test"), "http://localhost:3000")

    private fun <T> anyArg(): T = Mockito.any()

    private fun capture(): () -> ResendSendEmailRequest {
        var sent: ResendSendEmailRequest? = null
        Mockito.`when`(resend.send(anyArg())).thenAnswer { sent = it.getArgument(0); ResendSendEmailResponse(id = "re_1") }
        return { assertNotNull(sent) }
    }

    @Test
    fun `the review email links the case page, escapes the event name and carries no image`() {
        val sent = capture()
        service().sendPhotographerRemovalReviewEmail(
            "photog@example.com", "Run <b>5K</b> & more", "2 photos need your review: Photo ab12cd34.", requestId,
        )
        val html = sent().html
        assertEquals(listOf("photog@example.com"), sent().to)
        assertTrue(html.contains("href=\"http://localhost:3000/dashboard/photo-removal-requests/$requestId\""), html)
        assertTrue(html.contains("Run &lt;b&gt;5K&lt;/b&gt; &amp; more"), html)
        assertFalse(html.contains("<b>5K"), html)
        assertFalse(html.contains("<img"), html)
        assertTrue(html.contains("Photo ab12cd34"), html)
    }

    @Test
    fun `the outcome email has its own subject and the same safe link`() {
        val sent = capture()
        service().sendPhotographerRemovalOutcomeEmail("photog@example.com", "Cebu Run", "Photo ab12cd34: Platform cleanup confirmed", requestId)
        assertTrue(sent().subject.contains("outcome", ignoreCase = true), sent().subject)
        assertTrue(sent().html.contains("/dashboard/photo-removal-requests/$requestId"))
    }

    @Test
    fun `a send failure reaches the caller so the claim stays for the sweep`() {
        Mockito.`when`(resend.send(anyArg())).thenThrow(RuntimeException("resend down"))
        var thrown = false
        try { service().sendPhotographerRemovalReviewEmail("p@example.com", "Run", "body", requestId) } catch (e: RuntimeException) { thrown = true }
        assertTrue(thrown)
    }
}
