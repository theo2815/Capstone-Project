package com.quickpitik.service.orders

import com.quickpitik.config.PlatformProperties
import com.quickpitik.entity.Order
import com.quickpitik.entity.PaymentMethod
import org.junit.jupiter.api.Test
import java.math.BigDecimal
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class OrderAccessTokenServiceTest {
    private val service = OrderAccessTokenService(
        PlatformProperties(orderCapabilitySecret = "x".repeat(32)),
    )

    @Test
    fun `return and bundle capabilities cannot be exchanged`() {
        val order = order()
        val returnToken = service.issue(order, OrderCapability.RETURN)
        val bundleToken = service.issue(order, OrderCapability.BUNDLE)

        assertTrue(service.isValid(order, returnToken, OrderCapability.RETURN))
        assertFalse(service.isValid(order, returnToken, OrderCapability.BUNDLE))
        assertTrue(service.isValid(order, bundleToken, OrderCapability.BUNDLE))
        assertFalse(service.isValid(order, bundleToken, OrderCapability.RETURN))
    }

    @Test
    fun `a capability is bound to its order and signature`() {
        val order = order()
        val token = service.issue(order, OrderCapability.BUNDLE)

        assertFalse(service.isValid(order(), token, OrderCapability.BUNDLE))
        assertFalse(service.isValid(order, token.dropLast(1) + "x", OrderCapability.BUNDLE))
    }

    @Test
    fun `a migrated legacy token is checked by hash`() {
        val raw = "legacy-token"
        val order = order(
            legacyHash = MessageDigest.getInstance("SHA-256")
                .digest(raw.toByteArray(StandardCharsets.UTF_8))
                .joinToString("") { "%02x".format(it) },
        )

        assertTrue(service.isValid(order, raw, OrderCapability.BUNDLE))
        assertFalse(service.isValid(order, "wrong", OrderCapability.BUNDLE))
    }

    private fun order(legacyHash: String? = null) = Order(
        id = UUID.randomUUID(),
        eventId = UUID.randomUUID(),
        recipientEmail = "runner@example.com",
        paymentMethodWire = PaymentMethod.GCASH.wire,
        totalPhp = BigDecimal("125.00"),
        legacyShareTokenHash = legacyHash,
        tokenExpiresAt = OffsetDateTime.now().plusDays(1),
    )
}
