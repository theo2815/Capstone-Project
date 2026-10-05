package com.quickpitik.controller

import com.quickpitik.entity.Role
import com.quickpitik.exception.ApiException
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.earnings.EarningsService
import com.quickpitik.service.earnings.PayoutReportService
import com.quickpitik.service.earnings.PayoutRequestService
import com.quickpitik.service.earnings.PhotographerPurchaseService
import com.quickpitik.service.earnings.TransactionsLedgerService
import com.quickpitik.service.ratelimit.Bucket4jRateLimiter
import com.quickpitik.service.ratelimit.RateLimiter
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.time.Duration
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class MePhotographerPurchaseControllerTest {
    @Test
    fun `rate limited purchase reads do not access private history`() {
        val principal = AuthPrincipal(UUID.randomUUID(), "seller@test.local", Role.PHOTOGRAPHER)
        val purchases = Mockito.mock(PhotographerPurchaseService::class.java)
        val limiter = Mockito.mock(RateLimiter::class.java)
        Mockito.`when`(limiter.tryAcquire(Bucket4jRateLimiter.POLICY_ORDER_READ, principal.userId.toString()))
            .thenReturn(RateLimiter.Decision(false, Duration.ofSeconds(10)))
        val controller = MePhotographerEarningsController(
            Mockito.mock(EarningsService::class.java),
            Mockito.mock(PayoutReportService::class.java),
            Mockito.mock(TransactionsLedgerService::class.java),
            Mockito.mock(PayoutRequestService::class.java),
            purchases,
            limiter,
        )

        val eventError = assertFailsWith<ApiException> {
            controller.listPurchasedPhotos(principal, UUID.randomUUID(), null, null)
        }
        val photoError = assertFailsWith<ApiException> {
            controller.listPhotoPurchases(principal, UUID.randomUUID(), null, null, null)
        }

        assertEquals(HttpStatus.TOO_MANY_REQUESTS, eventError.status)
        assertEquals(HttpStatus.TOO_MANY_REQUESTS, photoError.status)
        Mockito.verifyNoInteractions(purchases)
    }
}
