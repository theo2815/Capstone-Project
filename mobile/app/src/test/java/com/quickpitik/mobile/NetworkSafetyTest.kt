package com.quickpitik.mobile

import com.quickpitik.mobile.data.remote.rewriteLoopbackUrl
import com.quickpitik.mobile.data.readAtMost
import com.quickpitik.mobile.ui.runner.isSafeCheckoutUrl
import java.io.ByteArrayInputStream
import okhttp3.HttpUrl.Companion.toHttpUrl
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class NetworkSafetyTest {
    @Test
    fun loopbackMediaUrlUsesBackendHost() {
        val rewritten = rewriteLoopbackUrl(
            "http://localhost:8080/storage/photo.jpg?signature=abc".toHttpUrl(),
            "http://10.0.2.2:8080/".toHttpUrl(),
        )

        assertEquals(
            "http://10.0.2.2:8080/storage/photo.jpg?signature=abc",
            rewritten.toString(),
        )
    }

    @Test
    fun loopbackMediaUrlKeepsItsOwnPort() {
        // A presigned URL can point at a different loopback service (MinIO on
        // :9000) — only the host may change, never the port.
        val rewritten = rewriteLoopbackUrl(
            "http://localhost:9000/quickpitik-dev/photo.jpg".toHttpUrl(),
            "http://10.0.2.2:8080/".toHttpUrl(),
        )

        assertEquals(
            "http://10.0.2.2:9000/quickpitik-dev/photo.jpg",
            rewritten.toString(),
        )
    }

    @Test
    fun externalMediaUrlIsUntouched() {
        val source = "https://cdn.example.com/photo.jpg".toHttpUrl()
        assertEquals(source, rewriteLoopbackUrl(source, "https://api.example.com/".toHttpUrl()))
    }

    @Test
    fun checkoutOnlyAcceptsHttps() {
        assertTrue(isSafeCheckoutUrl("https://checkout.paymongo.com/session"))
        assertFalse(isSafeCheckoutUrl("http://checkout.paymongo.com/session"))
        assertFalse(isSafeCheckoutUrl("javascript:alert(1)"))
    }

    @Test
    fun boundedReadStopsAtTheLimit() {
        val input = ByteArrayInputStream(byteArrayOf(1, 2, 3, 4, 5))
        assertArrayEquals(byteArrayOf(1, 2, 3), input.readAtMost(3))
    }
}
