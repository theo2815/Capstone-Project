package com.quickpitik.service.email

import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.dto.email.ResendSendEmailResponse
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class EmailServicePayoutHoldTest {
    private val resend = Mockito.mock(ResendClient::class.java)

    private fun <T> anyArg(): T = Mockito.any()

    @Test
    fun `the hold alert escapes the name and names no other photographer`() {
        var sent: ResendSendEmailRequest? = null
        Mockito.`when`(resend.send(anyArg())).thenAnswer {
            sent = it.getArgument(0)
            ResendSendEmailResponse(id = "re_1")
        }

        EmailService(resend, ResendProperties(apiKey = "re_live_test"), "http://localhost:3000")
            .sendPayoutHoldAlert("juan@example.com", "<b>Juan</b>")

        val request = assertNotNull(sent)
        assertEquals(listOf("juan@example.com"), request.to)
        assertEquals("Your QuickPitik payouts are on hold", request.subject)
        assertTrue(request.html.contains("&lt;b&gt;Juan&lt;/b&gt;"), request.html)
        assertFalse(request.html.contains("<b>Juan"), request.html)
        for (text in listOf(
            "so your payouts are on hold until an admin finishes the review.",
            "Your earnings stay in your balance. An admin may also take some of these photos off sale during the review. " +
                "You will also find this message in your QuickPitik inbox.",
            "Contact support within 14 days to appeal.",
        )) {
            assertTrue(request.html.contains(text), "missing: $text")
        }
        for (forbidden in listOf("copied", "copy", "another photographer")) {
            assertFalse(request.html.contains(forbidden), "the alert must not say '$forbidden'")
        }
    }
}
