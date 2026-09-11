package com.quickpitik.mobile.data.remote

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import retrofit2.HttpException
import retrofit2.Response
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException

/**
 * The origin helpers every image tile and socket dial goes through, and the
 * two error readers the ViewModels rely on for backend copy. Robolectric only
 * because setBaseUrl persists the override to SharedPreferences.
 */
@RunWith(RobolectricTestRunner::class)
class RetrofitClientTest {
    @After fun tearDown() = RetrofitClient.resetBaseUrl()

    @Test fun `a typed host gains the scheme and slash Retrofit needs, junk changes nothing`() {
        assertTrue(RetrofitClient.setBaseUrl("192.168.1.5:8080"))
        assertEquals("http://192.168.1.5:8080/", RetrofitClient.BASE_URL)
        assertTrue(RetrofitClient.setBaseUrl("https://api.example.com"))
        assertEquals("https://api.example.com/", RetrofitClient.BASE_URL)
        for (junk in listOf("", "   ", "not a url")) {
            assertFalse(junk, RetrofitClient.setBaseUrl(junk))
            assertEquals(junk, "https://api.example.com/", RetrofitClient.BASE_URL)
        }
        RetrofitClient.resetBaseUrl()
        assertEquals(RetrofitClient.DEFAULT_BASE_URL, RetrofitClient.BASE_URL)
    }

    @Test fun `image urls follow the live origin, relative paths prefixed and loopback hosts swapped`() {
        RetrofitClient.setBaseUrl("10.0.2.2:8080")
        assertEquals("http://10.0.2.2:8080/storage/a.jpg", RetrofitClient.resolveImageUrl("/storage/a.jpg"))
        assertEquals("http://10.0.2.2:9000/b.jpg?sig=1", RetrofitClient.resolveImageUrl("http://localhost:9000/b.jpg?sig=1"))
        assertEquals("https://cdn.example.com/c.jpg", RetrofitClient.resolveImageUrl("https://cdn.example.com/c.jpg"))
        assertEquals("not a url", RetrofitClient.resolveImageUrl("not a url"))
        assertNull(RetrofitClient.resolveImageUrl(null))
        assertEquals("", RetrofitClient.resolveImageUrl(""))
    }

    @Test fun `the websocket origin mirrors the scheme of the base url`() {
        assertEquals("wss://api.quickpitik.com:443", RetrofitClient.wsOrigin)
        RetrofitClient.setBaseUrl("10.0.2.2:8080")
        assertEquals("ws://10.0.2.2:8080", RetrofitClient.wsOrigin)
    }

    @Test fun `transport failures become plain copy and never leak the exception`() {
        val unreachable = "Couldn't reach QuickPitik — check your connection."
        assertEquals(unreachable, RetrofitClient.parseError(UnknownHostException("api.quickpitik.com")))
        assertEquals(unreachable, RetrofitClient.parseError(ConnectException("Failed to connect to /192.168.1.232:8080")))
        assertEquals("The connection timed out. Try again.", RetrofitClient.parseError(SocketTimeoutException("timeout")))
        assertEquals("Network error. Check your connection and try again.", RetrofitClient.parseError(IOException("unexpected end of stream")))
        assertEquals("boom", RetrofitClient.parseError(IllegalStateException("boom")))
    }

    @Test fun `a backend envelope yields its code and message, and peeking never consumes the body`() {
        val limited = httpException(429, """{"success":false,"errors":[{"code":"RATE_LIMITED","message":"Slow down. Try again in a minute."}]}""")
        assertEquals(ApiError("RATE_LIMITED", "Slow down. Try again in a minute."), RetrofitClient.parseHttpError(limited))
        assertEquals("Slow down. Try again in a minute.", RetrofitClient.parseError(limited))

        val garbage = httpException(500, "<html>Bad Gateway</html>")
        assertNull(RetrofitClient.parseHttpError(garbage))
        assertEquals("HTTP 500", RetrofitClient.parseError(garbage))
        assertNull(RetrofitClient.parseHttpError(IOException("not http")))
    }

    private fun httpException(code: Int, body: String) =
        HttpException(Response.error<Any>(code, body.toResponseBody("application/json".toMediaType())))
}
