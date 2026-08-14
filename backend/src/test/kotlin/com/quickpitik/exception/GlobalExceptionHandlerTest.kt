package com.quickpitik.exception

import com.quickpitik.common.ErrorCodes
import com.quickpitik.service.ai.AiApiException
import org.junit.jupiter.api.Test
import org.springframework.http.HttpStatus
import kotlin.test.assertEquals

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

    @Test
    fun `ai-api failures always map to 503 regardless of code`() {
        val ex = AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "LOW_QUALITY", "boom")
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, handler.handleAiApi(ex).statusCode)
    }
}
