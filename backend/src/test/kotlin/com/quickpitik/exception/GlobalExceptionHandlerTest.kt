package com.quickpitik.exception

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import com.quickpitik.service.ai.AiApiException
import jakarta.validation.ConstraintViolationException
import org.junit.jupiter.api.Test
import org.springframework.core.MethodParameter
import org.springframework.http.HttpMethod
import org.springframework.http.HttpStatus
import org.springframework.http.converter.HttpMessageNotReadableException
import org.springframework.mock.http.MockHttpInputMessage
import org.springframework.validation.BeanPropertyBindingResult
import org.springframework.validation.FieldError
import org.springframework.web.HttpRequestMethodNotSupportedException
import org.springframework.web.bind.MethodArgumentNotValidException
import org.springframework.web.bind.MissingServletRequestParameterException
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException
import org.springframework.web.multipart.support.MissingServletRequestPartException
import org.springframework.web.servlet.resource.NoResourceFoundException
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

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

    // An unmatched path used to fall through to the catch-all and answer 500
    // INTERNAL_ERROR, so a client typo looked like a server fault. Found
    // 2026-08-15 by requesting `/photos/upload` (the real path is `/photos`).
    @Test
    fun `an unmatched route is a 404, not a 500`() {
        val response = handler.handleNoResource(
            NoResourceFoundException(HttpMethod.GET, "/api/v1/photos/upload"),
        )

        assertEquals(HttpStatus.NOT_FOUND, response.statusCode)
        assertEquals(ErrorCodes.NOT_FOUND, response.body?.errors?.first()?.code)
    }

    @Test
    fun `the unmatched-route envelope does not echo the requested path`() {
        // The path is attacker-controlled; reflecting it invites probing and
        // response-splitting noise in whatever renders the message.
        val response = handler.handleNoResource(
            NoResourceFoundException(HttpMethod.GET, "/api/v1/<script>alert(1)</script>"),
        )

        val message = response.body?.errors?.first()?.message
        assertEquals("Resource not found", message)
        assertFalse(message!!.contains("script"))
    }

    // ─── Malformed-request mappings (2026-08-27): each previously fell
    // through to the 500 catch-all.

    @Test
    fun `a malformed path variable is a 400, not a 500`() {
        val ex = MethodArgumentTypeMismatchException(
            "not-a-uuid",
            java.util.UUID::class.java,
            "eventId",
            MethodParameter(javaClass.getDeclaredMethod("dummy", String::class.java), 0),
            IllegalArgumentException("bad uuid"),
        )

        val response = handler.handleTypeMismatch(ex)

        assertEquals(HttpStatus.BAD_REQUEST, response.statusCode)
        assertEquals(ErrorCodes.VALIDATION_ERROR, response.body?.errors?.first()?.code)
        assertEquals("eventId", response.body?.errors?.first()?.field)
    }

    @Test
    fun `a missing multipart part is a 400 naming the part`() {
        val response = handler.handleMissingPart(MissingServletRequestPartException("file"))

        assertEquals(HttpStatus.BAD_REQUEST, response.statusCode)
        assertEquals("file", response.body?.errors?.first()?.field)
    }

    @Test
    fun `a missing query parameter is a 400 naming the parameter`() {
        val response = handler.handleMissingParameter(
            MissingServletRequestParameterException("token", "String"),
        )

        assertEquals(HttpStatus.BAD_REQUEST, response.statusCode)
        assertEquals("token", response.body?.errors?.first()?.field)
    }

    @Test
    fun `a wrong HTTP method is a 405`() {
        val response = handler.handleMethodNotSupported(
            HttpRequestMethodNotSupportedException("DELETE"),
        )

        assertEquals(HttpStatus.METHOD_NOT_ALLOWED, response.statusCode)
        assertEquals(ErrorCodes.METHOD_NOT_ALLOWED, response.body?.errors?.first()?.code)
    }

    @Test
    fun `a bean-validation constraint violation is a 400`() {
        val response = handler.handleConstraintViolation(ConstraintViolationException(emptySet()))

        assertEquals(HttpStatus.BAD_REQUEST, response.statusCode)
        assertEquals(ErrorCodes.VALIDATION_ERROR, response.body?.errors?.first()?.code)
    }

    // ─── 2026-09-11 test-suite audit: the three handlers every client leans
    // on that had no test — the 500 catch-all, the Retry-After carrier, and
    // @Valid body failures.

    @Test
    fun `the catch-all never echoes the exception message`() {
        val response = handler.handleGeneric(RuntimeException("connection refused: db password is hunter2"))

        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.statusCode)
        assertEquals("INTERNAL_ERROR", response.body?.errors?.first()?.code)
        assertEquals("Internal server error", response.body?.errors?.first()?.message)
    }

    // The only place ACCOUNT_LOCKED / RATE_LIMITED become a header the client
    // can act on; every other test asserts the field on the exception.
    @Test
    fun `an ApiException with retryAfterSeconds carries a Retry-After header`() {
        val limited = ApiException(HttpStatus.TOO_MANY_REQUESTS, ErrorCodes.RATE_LIMITED, "slow down", retryAfterSeconds = 7)
        val plain = ApiException(HttpStatus.CONFLICT, "HANDLE_TAKEN", "taken")

        assertEquals(HttpStatus.TOO_MANY_REQUESTS, handler.handleApi(limited).statusCode)
        assertEquals("7", handler.handleApi(limited).headers.getFirst("Retry-After"))
        assertNull(handler.handleApi(plain).headers.getFirst("Retry-After"))
    }

    @Test
    fun `a @Valid body failure reports one error per field naming the field`() {
        val binding = BeanPropertyBindingResult(Any(), "request")
        binding.addError(FieldError("request", "email", "must be a valid email"))
        binding.addError(FieldError("request", "password", "must not be blank"))
        val ex = MethodArgumentNotValidException(
            MethodParameter(javaClass.getDeclaredMethod("dummy", String::class.java), 0),
            binding,
        )

        val response = handler.handleMethodArgNotValid(ex)

        assertEquals(HttpStatus.BAD_REQUEST, response.statusCode)
        assertEquals(listOf("email", "password"), response.body?.errors?.map { it.field })
        assertEquals(listOf("must be a valid email", "must not be blank"), response.body?.errors?.map { it.message })
        assertTrue(response.body?.errors?.all { it.code == "VALIDATION_FAILED" } == true)
    }

    @Suppress("UNUSED_PARAMETER", "unused")
    private fun dummy(value: String) = Unit
}
