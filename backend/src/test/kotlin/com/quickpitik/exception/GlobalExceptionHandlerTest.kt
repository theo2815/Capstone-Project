package com.quickpitik.exception

import com.quickpitik.common.ErrorCodes
import com.quickpitik.service.ai.AiApiException
import org.junit.jupiter.api.Test
import org.springframework.http.HttpStatus
import org.springframework.http.converter.HttpMessageNotReadableException
import org.springframework.mock.http.MockHttpInputMessage
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class GlobalExceptionHandlerTest {

    private val handler = GlobalExceptionHandler()

    private fun codeFor(aiCode: String?): String? {
        val ex = AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, aiCode, "boom")
        return handler.handleAiApi(ex).body?.errors?.first()?.code
    }

    @Test
    fun `documented ai-api wire codes pass through`() {
        assertEquals("LOW_QUALITY", codeFor("LOW_QUALITY"))
        assertEquals("NO_FACES", codeFor("NO_FACES"))
        assertEquals("MODEL_UNAVAILABLE", codeFor("MODEL_UNAVAILABLE"))
    }

    @Test
    fun `QuickPitikError class names collapse to AI_API_UNAVAILABLE`() {
        // ai-api sets error.code to the Python class name for QuickPitikError
        // subclasses — those must never reach our public envelope.
        assertEquals(ErrorCodes.AI_API_UNAVAILABLE, codeFor("ImageValidationError"))
        assertEquals(ErrorCodes.AI_API_UNAVAILABLE, codeFor("JobNotFoundError"))
    }

    @Test
    fun `null and unknown codes fall back to AI_API_UNAVAILABLE`() {
        assertEquals(ErrorCodes.AI_API_UNAVAILABLE, codeFor(null))
        assertEquals(ErrorCodes.AI_API_UNAVAILABLE, codeFor("SOME_FUTURE_CODE"))
    }

    // Runner-flow audit (2026-05-27), "Events + photo discovery" item E5.
    //
    // Previously every AiApiException became a 503, so a selfie ai-api had
    // understood and rejected was reported as "service down, try again" — advice
    // that can never work. AiApiException.status already knows the difference.

    @Test
    fun `ai-api 4xx means the input was rejected, not that the service is down`() {
        val ex = AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "LOW_QUALITY", "boom")
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, handler.handleAiApi(ex).statusCode)
    }

    @Test
    fun `ai-api 4xx other than 422 also maps to 422`() {
        val ex = AiApiException(HttpStatus.BAD_REQUEST, null, "boom")
        assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, handler.handleAiApi(ex).statusCode)
    }

    @Test
    fun `ai-api 5xx and transport failures stay 503`() {
        assertEquals(
            HttpStatus.SERVICE_UNAVAILABLE,
            handler.handleAiApi(AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "offline")).statusCode,
        )
        assertEquals(
            HttpStatus.SERVICE_UNAVAILABLE,
            handler.handleAiApi(AiApiException(HttpStatus.BAD_GATEWAY, null, "malformed")).statusCode,
        )
    }

    // Spotted 2026-08-14 while probing the refund endpoint: with no handler for
    // it, a syntactically bad body fell through to the catch-all and told the
    // caller the server had broken. Affects every @RequestBody route.

    @Test
    fun `a body Jackson cannot read is the caller's fault, not a 500`() {
        val response = handler.handleUnreadableBody(unreadable("Unexpected end-of-input"))

        assertEquals(HttpStatus.BAD_REQUEST, response.statusCode)
        assertEquals(ErrorCodes.VALIDATION_ERROR, response.body?.errors?.first()?.code)
    }

    @Test
    fun `the envelope does not echo Jackson's parse context`() {
        // Jackson's message quotes the offending payload and names the Kotlin
        // target class; neither belongs in a public error envelope.
        val leaky = """Cannot construct instance of `com.quickpitik.dto.orders.CreateOrderItem`, """ +
            """problem: photoId at [Source: (String)"{"items":[{"eventId":"secret"}]}"; line: 1]"""

        val message = handler.handleUnreadableBody(unreadable(leaky)).body?.errors?.first()?.message

        assertEquals("Request body could not be parsed.", message)
        assertFalse(message!!.contains("CreateOrderItem"))
        assertFalse(message.contains("secret"))
    }

    private fun unreadable(message: String) = HttpMessageNotReadableException(
        message,
        MockHttpInputMessage(ByteArray(0)),
    )

    @Test
    fun `status mapping does not disturb the code allowlist`() {
        val clientFault = AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NO_FACES", "boom")
        assertEquals("NO_FACES", handler.handleAiApi(clientFault).body?.errors?.first()?.code)

        val outage = AiApiException(HttpStatus.SERVICE_UNAVAILABLE, "ImageValidationError", "boom")
        assertEquals(ErrorCodes.AI_API_UNAVAILABLE, handler.handleAiApi(outage).body?.errors?.first()?.code)
    }
}
