package com.quickpitik.controller

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.UnauthorizedException
import com.quickpitik.security.PaymongoSignatureVerifier
import com.quickpitik.service.orders.PaymongoWebhookService
import jakarta.servlet.http.HttpServletRequest
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

class PaymongoWebhookControllerTest {
    @Test
    fun `signature failure wins before malformed JSON is parsed`() {
        val verifier = Mockito.mock(PaymongoSignatureVerifier::class.java)
        val service = Mockito.mock(PaymongoWebhookService::class.java)
        val request = Mockito.mock(HttpServletRequest::class.java)
        val failure = UnauthorizedException(ErrorCodes.UNAUTHORIZED, "bad signature")
        Mockito.doThrow(failure).`when`(verifier).verify(request)

        val thrown = assertFailsWith<UnauthorizedException> {
            PaymongoWebhookController(verifier, service, ObjectMapper()).handle(request, "not-json")
        }

        assertSame(failure, thrown)
        Mockito.verifyNoInteractions(service)
    }
}
