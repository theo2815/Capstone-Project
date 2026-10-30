package com.quickpitik.controller

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.service.photographer.PublicPhotographerService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.mock.web.MockHttpServletRequest
import org.springframework.mock.web.MockHttpServletResponse
import kotlin.test.assertEquals

// A bib on the photographer gallery is a search, so it shares the search limit
// with the event bib search and face search; plain browsing must not.
class PublicPhotographerControllerTest {

    private val policies = mutableListOf<String>()
    private val limiter = object : RateLimiter {
        override fun tryAcquire(policy: String, key: String): RateLimiter.Decision {
            policies += policy
            return RateLimiter.Decision(allowed = true)
        }
    }
    private val service = Mockito.mock(PublicPhotographerService::class.java).also {
        Mockito.`when`(it.listEventPhotos(anyArg(), anyArg(), anyArg(), Mockito.any(), Mockito.any()))
            .thenReturn(PaginatedResponse.empty(PaginationParams.of(0, 20)))
    }
    private val controller = PublicPhotographerController(service, limiter)

    private fun list(bib: String?) = controller.listEventPhotos(
        null, "shooter", "race", bib, null, null, MockHttpServletRequest(), MockHttpServletResponse(),
    )

    @Test
    fun `a bib search takes a token from the shared search limit`() {
        list("10342")
        assertEquals(listOf(Bucket4jRateLimiter.POLICY_PUBLIC_GALLERY, Bucket4jRateLimiter.POLICY_PHOTO_SEARCH), policies)
    }

    @Test
    fun `plain browsing does not touch the search limit`() {
        list(null)
        list(" ")
        assertEquals(listOf(Bucket4jRateLimiter.POLICY_PUBLIC_GALLERY, Bucket4jRateLimiter.POLICY_PUBLIC_GALLERY), policies)
    }

    private fun <T> anyArg(): T = Mockito.any()
}
