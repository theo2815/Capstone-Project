package com.quickpitik.controller

import com.quickpitik.dto.photographer.FileAppealRequest
import com.quickpitik.dto.photographer.PhotoIdsRequest
import com.quickpitik.entity.Role
import com.quickpitik.exception.ApiException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import com.quickpitik.service.relevance.RelevanceAppealService
import com.quickpitik.service.relevance.RelevancePhotographerHideService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import org.springframework.mock.web.MockHttpServletResponse
import java.time.Duration
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// RVF-16 (RVR-17): the three writing routes share the photographer's media-upload budget; the reads are not limited.
class MePhotographerRelevanceControllerTest {
    private val principal = AuthPrincipal(UUID.randomUUID(), "seller@test.local", Role.PHOTOGRAPHER)
    private val appeals = Mockito.mock(RelevanceAppealService::class.java)
    private val hides = Mockito.mock(RelevancePhotographerHideService::class.java)
    private val limiter = Mockito.mock(RateLimiter::class.java)
    private val controller = MePhotographerRelevanceController(appeals, hides, limiter)
    private val caseId = UUID.randomUUID()

    @Test
    fun `appeal, hide and show again answer 429 over the limit`() {
        Mockito.`when`(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_MEDIA_UPLOAD, principal.userId.toString()))
            .thenReturn(RateLimiter.Decision(false, Duration.ofSeconds(10)))
        val ids = PhotoIdsRequest(listOf(UUID.randomUUID()))
        val calls = listOf<(MockHttpServletResponse) -> Any?>(
            { controller.appeal(principal, caseId, FileAppealRequest("mine"), it) },
            { controller.hide(principal, caseId, ids, it) },
            { controller.unhide(principal, caseId, ids, it) },
        )

        for (call in calls) {
            val response = MockHttpServletResponse()
            assertEquals(HttpStatus.TOO_MANY_REQUESTS, assertFailsWith<ApiException> { call(response) }.status)
            assertEquals("private, no-store", response.getHeader("Cache-Control"))
        }
        Mockito.verifyNoInteractions(appeals, hides)
    }

    @Test
    fun `the reads are not limited`() {
        controller.case(principal, caseId)
        controller.blurry(principal, caseId, null, null)
        controller.hidden(principal, caseId, null, null)

        Mockito.verifyNoInteractions(limiter)
    }
}
