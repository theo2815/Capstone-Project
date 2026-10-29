package com.quickpitik.service.email

import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.dto.email.ResendSendEmailResponse
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class EmailServiceRefundLinkTest {
    private val resend = Mockito.mock(ResendClient::class.java)

    private fun service(apiKey: String) =
        EmailService(resend, ResendProperties(apiKey = apiKey), "http://localhost:3000")

    private fun <T> anyArg(): T = Mockito.any()

    @Test
    fun `the refund link mail carries the escaped link, the amount and the 3-day line`() {
        var sent: ResendSendEmailRequest? = null
        Mockito.`when`(resend.send(anyArg())).thenAnswer {
            sent = it.getArgument(0)
            ResendSendEmailResponse(id = "re_1")
        }

        service("re_live_test").sendRefundLinkEmail("runner@example.com", "https://transfer.test/a?x=1&y=2", BigDecimal("127.50"))

        val request = assertNotNull(sent)
        assertEquals(listOf("runner@example.com"), request.to)
        assertEquals("Your QuickPitik refund link", request.subject)
        assertTrue(request.html.contains("href=\"https://transfer.test/a?x=1&amp;y=2\""), request.html)
        assertTrue(request.html.contains("The link works for 3 days."), request.html)
        assertTrue(request.html.contains("127.50"), request.html)
    }

    @Test
    fun `quotes and angle brackets in the link cannot break out of the href`() {
        var sent: ResendSendEmailRequest? = null
        Mockito.`when`(resend.send(anyArg())).thenAnswer {
            sent = it.getArgument(0)
            ResendSendEmailResponse(id = "re_1")
        }

        service("re_live_test").sendRefundLinkEmail("runner@example.com", "https://transfer.test/x\"><b>&", BigDecimal("127.50"))

        val html = assertNotNull(sent).html
        assertTrue(html.contains("href=\"https://transfer.test/x&quot;&gt;&lt;b&gt;&amp;\""), html)
        assertTrue(!html.contains("<b>"), html)
    }

    @Test
    fun `a send failure is logged and dropped`() {
        Mockito.`when`(resend.send(anyArg())).thenThrow(RuntimeException("resend down"))

        service("re_live_test").sendRefundLinkEmail("runner@example.com", "https://transfer.test/abc", BigDecimal("127.50"))
    }

    @Test
    fun `dev mode only logs and never calls Resend`() {
        service("re_dev-only-x").sendRefundLinkEmail("runner@example.com", "https://transfer.test/abc", BigDecimal("127.50"))

        Mockito.verifyNoInteractions(resend)
    }
}
