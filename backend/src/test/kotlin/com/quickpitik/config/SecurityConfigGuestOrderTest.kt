package com.quickpitik.config

import org.junit.jupiter.api.Test
import org.springframework.mock.web.MockHttpServletRequest
import java.util.UUID
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Guards the guest checkout-return flow. PayMongo redirects the buyer to
// /orders/return, which polls the token-gated guest order routes that
// SecurityConfig permitAll's by regex. RegexRequestMatcher matches its pattern
// against `servletPath + '?' + queryString` (confirmed in spring-security-web
// source), so a bare `...$` anchor silently rejected every request carrying the
// ?token=... param: the guest (no JWT) fell through to authenticated() and got a
// 401 — web bounced to /login, mobile spun on "Sealing your photos". These pin
// that the token query string is permitted while sibling paths stay private.
//
// NOTE: MockMvc's `.param()` does NOT populate getQueryString(), which is why the
// old controller tests passed while real browser/app requests failed — so these
// assertions set queryString the way a real request carries it.
class SecurityConfigGuestOrderTest {

    private val orderId = UUID.randomUUID().toString()
    private val token = "v1.return.$orderId.1788168931.s9yVyngt3dmorqS-13hFOgkVOJy3Cl3gkbOIfq0i_48"

    private fun get(path: String, query: String?) = MockHttpServletRequest().apply {
        method = "GET"
        servletPath = path
        queryString = query
    }

    @Test
    fun `status route is permitted with a token query string`() {
        assertTrue(
            SecurityConfig.guestOrderGet("/status")
                .matches(get("/api/v1/orders/$orderId/status", "token=$token")),
        )
    }

    @Test
    fun `detail route is permitted with a token query string`() {
        assertTrue(
            SecurityConfig.guestOrderGet("")
                .matches(get("/api/v1/orders/$orderId", "token=$token")),
        )
    }

    @Test
    fun `bundle route is permitted with a token query string`() {
        assertTrue(
            SecurityConfig.guestOrderGet("/download-bundle")
                .matches(get("/api/v1/orders/$orderId/download-bundle", "token=$token")),
        )
    }

    // Per-photo downloads add `photo=`; Meta's in-app browsers add `fbclid=`
    // on top. Both must stay permitted — that tolerance is the whole reason
    // downloads route through here instead of a presigned URL (2026-09-05).
    @Test
    fun `bundle route stays permitted with a photo id and a stray tracking param`() {
        assertTrue(
            SecurityConfig.guestOrderGet("/download-bundle")
                .matches(
                    get(
                        "/api/v1/orders/$orderId/download-bundle",
                        "token=$token&photo=${UUID.randomUUID()}&fbclid=IwZXh0bgNhZW0",
                    ),
                ),
        )
    }

    @Test
    fun `status route is still permitted without a query string`() {
        assertTrue(
            SecurityConfig.guestOrderGet("/status")
                .matches(get("/api/v1/orders/$orderId/status", null)),
        )
    }

    // The end anchor must keep sibling routes private even though a query string
    // is now allowed — an extra path segment is not a query string.
    @Test
    fun `a sibling path segment is not permitted by the detail matcher`() {
        assertFalse(
            SecurityConfig.guestOrderGet("")
                .matches(get("/api/v1/orders/$orderId/refund", "token=$token")),
        )
    }

    @Test
    fun `a non-uuid id is not permitted`() {
        assertFalse(
            SecurityConfig.guestOrderGet("/status")
                .matches(get("/api/v1/orders/not-a-uuid/status", "token=$token")),
        )
    }
}
