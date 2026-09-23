package com.quickpitik.config

import com.quickpitik.service.auth.EmailVerificationListener
import com.quickpitik.service.auth.UserRegisteredEvent
import com.quickpitik.service.orders.OrderPaidEmailListener
import com.quickpitik.service.orders.OrderPaidEvent
import com.quickpitik.service.photographer.PayoutAccountChangedEmailListener
import com.quickpitik.service.photographer.PayoutAccountChangedEvent
import org.junit.jupiter.api.Test
import org.springframework.context.annotation.Bean
import org.springframework.scheduling.annotation.Async
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Spring resolves an @Async qualifier at the first invocation, not at startup:
// a pool name matching no bean throws only when the first email is actually
// sent — in production, at a race finish. These two asserts move that failure
// to build time, and pin the 2026-09-11 audit's P-4 finding: a Resend outage
// must not be able to occupy the AI-indexing pool.
class AsyncConfigTest {

    private val declaredPools: Set<String> =
        AsyncConfig::class.java.declaredMethods
            .flatMap { it.getAnnotation(Bean::class.java)?.name?.toList().orEmpty() }
            .toSet()

    private val emailListeners = listOf(
        OrderPaidEmailListener::class.java.getDeclaredMethod("onOrderPaid", OrderPaidEvent::class.java),
        EmailVerificationListener::class.java.getDeclaredMethod("onUserRegistered", UserRegisteredEvent::class.java),
        PayoutAccountChangedEmailListener::class.java
            .getDeclaredMethod("onPayoutAccountChanged", PayoutAccountChangedEvent::class.java),
    )

    @Test
    fun `every email listener runs on emailSending, never the indexing pool`() {
        emailListeners.forEach {
            assertEquals("emailSending", it.getAnnotation(Async::class.java).value, it.declaringClass.simpleName)
        }
    }

    @Test
    fun `the pool the email listeners name is a real bean`() {
        assertTrue("emailSending" in declaredPools, "AsyncConfig declares $declaredPools")
    }
}
