package com.quickpitik.service.orders

import com.quickpitik.config.PlatformProperties
import com.quickpitik.entity.Order
import org.springframework.stereotype.Service
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.time.OffsetDateTime
import java.util.Base64
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

enum class OrderCapability(val wire: String) {
    RETURN("return"),
    BUNDLE("bundle"),
}

@Service
class OrderAccessTokenService(
    private val properties: PlatformProperties,
) {
    private val secret = properties.orderCapabilitySecret.toByteArray(StandardCharsets.UTF_8)

    init {
        require(secret.size >= 32) { "ORDER_CAPABILITY_SECRET must be at least 32 bytes" }
    }

    fun issue(order: Order, capability: OrderCapability): String {
        val expiresAt = when (capability) {
            OrderCapability.RETURN -> minOf(order.tokenExpiresAt, OffsetDateTime.now().plusMinutes(15))
            OrderCapability.BUNDLE -> order.tokenExpiresAt
        }
        val payload = "v1.${capability.wire}.${order.id}.${expiresAt.toEpochSecond()}"
        return "$payload.${sign(payload)}"
    }

    fun isValid(order: Order, token: String?, capability: OrderCapability): Boolean {
        if (token.isNullOrBlank()) return false
        if (isValidSigned(order.id, token, capability)) return true
        return isValidLegacy(order, token)
    }

    private fun isValidSigned(orderId: UUID, token: String, capability: OrderCapability): Boolean {
        val parts = token.split('.')
        if (parts.size != 5 || parts[0] != "v1" || parts[1] != capability.wire) return false
        if (parts[2] != orderId.toString()) return false
        val expiresAt = parts[3].toLongOrNull() ?: return false
        if (expiresAt < OffsetDateTime.now().toEpochSecond()) return false
        val payload = parts.take(4).joinToString(".")
        return MessageDigest.isEqual(
            sign(payload).toByteArray(StandardCharsets.US_ASCII),
            parts[4].toByteArray(StandardCharsets.US_ASCII),
        )
    }

    private fun isValidLegacy(order: Order, token: String): Boolean {
        val expected = order.legacyShareTokenHash ?: return false
        if (order.tokenExpiresAt.isBefore(OffsetDateTime.now())) return false
        return MessageDigest.isEqual(
            expected.toByteArray(StandardCharsets.US_ASCII),
            sha256Hex(token).toByteArray(StandardCharsets.US_ASCII),
        )
    }

    private fun sign(payload: String): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret, "HmacSHA256"))
        return Base64.getUrlEncoder().withoutPadding()
            .encodeToString(mac.doFinal(payload.toByteArray(StandardCharsets.UTF_8)))
    }

    private fun sha256Hex(value: String): String =
        MessageDigest.getInstance("SHA-256")
            .digest(value.toByteArray(StandardCharsets.UTF_8))
            .joinToString("") { "%02x".format(it) }
}
