package com.quickpitik.mobile.data.remote

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import retrofit2.HttpException
import retrofit2.Response
import java.net.SocketTimeoutException

/**
 * The one place that decides what a `{success, data, errors}` reply means.
 * Every repository call and the auth ViewModel route through it, so a slip
 * here hands back success on a missing payload, or throws away the backend's
 * own wording, at some thirty call sites at once.
 */
class ApiResultTest {

    @Test fun `an envelope carrying data succeeds with that data`() {
        val envelope = ApiResponseEnvelope(success = true, data = "p1")
        assertEquals("p1", envelope.toResult("Couldn't load the photo").getOrNull())
    }

    @Test fun `a refusal reads the backend's own words, the fallback only when it sent none`() {
        val refused = ApiResponseEnvelope<String>(
            success = false,
            errors = listOf(ApiError("COUPON_EXPIRED", "That coupon has expired.")),
        )
        assertEquals(
            "That coupon has expired.",
            refused.toResult("Couldn't apply that coupon").exceptionOrNull()?.message,
        )
        assertEquals(
            "Couldn't apply that coupon",
            ApiResponseEnvelope<String>(success = false).toResult("Couldn't apply that coupon")
                .exceptionOrNull()?.message,
        )
        // success with no payload is a refusal too - the data is what the caller asked for.
        assertEquals(
            "Couldn't apply that coupon",
            ApiResponseEnvelope<String>(success = true).toResult("Couldn't apply that coupon")
                .exceptionOrNull()?.message,
        )
    }

    @Test fun `a thrown backend error surfaces its envelope message, not the status code`() = runBlocking {
        val rateLimited = HttpException(
            Response.error<Any>(
                429,
                """{"success":false,"errors":[{"code":"RATE_LIMITED","message":"Slow down. Try again in a minute."}]}"""
                    .toResponseBody("application/json".toMediaType()),
            )
        )
        val result = apiCall<String>("Checkout failed") { throw rateLimited }
        assertEquals("Slow down. Try again in a minute.", result.exceptionOrNull()?.message)
    }

    @Test fun `a dropped connection reads as plain copy and never leaks the exception`() = runBlocking {
        val result = apiCall<String>("Checkout failed") { throw SocketTimeoutException("timeout") }
        assertEquals("The connection timed out. Try again.", result.exceptionOrNull()?.message)
        assertEquals("The connection timed out. Try again.", result.exceptionOrNull()!!.uiMessage)
    }

    @Test fun `a cancelled call is rethrown, never handed back as a failed Result`() = runBlocking {
        // Screens fold a failure into an Error state. If cancellation arrived as
        // one, leaving a tab (or any cancelled reload) would paint "Couldn't
        // load…" over the surface that was on its way out.
        val outcome = runCatching {
            apiCall<String>("Couldn't load the photo") { throw CancellationException("scope cancelled") }
        }
        assertTrue(
            "expected the CancellationException to propagate, got $outcome",
            outcome.exceptionOrNull() is CancellationException,
        )
    }
}
