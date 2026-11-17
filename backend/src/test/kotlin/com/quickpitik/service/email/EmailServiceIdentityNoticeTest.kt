package com.quickpitik.service.email

import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.dto.email.ResendSendEmailResponse
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.LocalDate
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class EmailServiceIdentityNoticeTest {
    private val resend = Mockito.mock(ResendClient::class.java)
    private val startsOn = LocalDate.of(2026, 12, 7)

    private fun service(apiKey: String = "re_live_test") =
        EmailService(resend, ResendProperties(apiKey = apiKey), "https://quickpitik.com,https://preview.test")

    private fun <T> anyArg(): T = Mockito.any()

    @Test
    fun `the notice names the date, links settings and the seller-checks privacy section`() {
        var sent: ResendSendEmailRequest? = null
        Mockito.`when`(resend.send(anyArg())).thenAnswer {
            sent = it.getArgument(0)
            ResendSendEmailResponse(id = "re_1")
        }

        assertTrue(service().sendIdentityCheckNotice("p@example.com", "Ana <b>", startsOn))

        val request = assertNotNull(sent)
        assertEquals(listOf("p@example.com"), request.to)
        assertEquals("Before Monday 7 December 2026: a quick live selfie to keep uploading", request.subject)
        val html = request.html
        assertTrue(html.contains("Hi Ana &lt;b&gt;,"), html)
        assertTrue(html.contains("From Monday 7 December 2026, QuickPitik will ask every photographer"), html)
        assertTrue(html.contains("href=\"https://quickpitik.com/dashboard/settings\""), html)
        assertTrue(html.contains("Amazon Web Services in Malaysia and Singapore"), html)
        assertTrue(html.contains("href=\"https://quickpitik.com/privacy#seller-checks\""), html)
        assertTrue(html.contains("Have an admin check instead"), html)
    }

    @Test
    fun `a failed send is reported as not sent`() {
        Mockito.`when`(resend.send(anyArg())).thenThrow(RuntimeException("resend down"))

        assertFalse(service().sendIdentityCheckNotice("p@example.com", "Ana", startsOn))
    }

    @Test
    fun `the dev key only logs and calls nothing`() {
        assertTrue(service("re_dev-only-x").sendIdentityCheckNotice("p@example.com", "Ana", startsOn))
        Mockito.verifyNoInteractions(resend)
    }
}
