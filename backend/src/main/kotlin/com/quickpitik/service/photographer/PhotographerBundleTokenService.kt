package com.quickpitik.service.photographer

import com.quickpitik.config.PlatformProperties
import org.springframework.stereotype.Service
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.time.Duration
import java.time.OffsetDateTime
import java.util.Base64
import java.util.UUID
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

// Signs a short-lived capability to stream one photographer's own "Download
// All" ZIP for one event. Mirrors OrderAccessTokenService: the ZIP endpoint is
// reached by a top-level <a> navigation (web) that carries no JWT, so the token
// in the query string is the credential. The grant is bound to
// (photographerId, eventId) and HMAC-signed — a forged or replayed token can
// never widen access, and the stream side only ever loads that photographer's
// own photos regardless.
//
// Reuses ORDER_CAPABILITY_SECRET (already required ≥32 bytes in prod) rather
// than adding another secret to rotate.
@Service
class PhotographerBundleTokenService(
    properties: PlatformProperties,
) {
    private val secret = properties.orderCapabilitySecret.toByteArray(StandardCharsets.UTF_8)

    init {
        require(secret.size >= 32) { "ORDER_CAPABILITY_SECRET must be at least 32 bytes" }
    }

    fun issue(photographerId: UUID, eventId: UUID): Grant {
        val expiresAt = OffsetDateTime.now().plus(TTL)
        val payload = "v1.pb.$photographerId.$eventId.${expiresAt.toEpochSecond()}"
        return Grant(token = "$payload.${sign(payload)}", expiresAt = expiresAt)
    }

    // Returns the photographerId the token grants, or null if the token is
    // missing/malformed/expired/for another event. Constant-time signature
    // compare; every failure is a null so the caller answers a uniform 404.
    fun verify(token: String?, eventId: UUID): UUID? {
        if (token.isNullOrBlank()) return null
        val parts = token.split('.')
        if (parts.size != 6 || parts[0] != "v1" || parts[1] != "pb") return null
        val photographerId = runCatching { UUID.fromString(parts[2]) }.getOrNull() ?: return null
        if (parts[3] != eventId.toString()) return null
        val expiresAt = parts[4].toLongOrNull() ?: return null
        if (expiresAt < OffsetDateTime.now().toEpochSecond()) return null
        val payload = parts.take(5).joinToString(".")
        val ok = MessageDigest.isEqual(
            sign(payload).toByteArray(StandardCharsets.US_ASCII),
            parts[5].toByteArray(StandardCharsets.US_ASCII),
        )
        return if (ok) photographerId else null
    }

    private fun sign(payload: String): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret, "HmacSHA256"))
        return Base64.getUrlEncoder().withoutPadding()
            .encodeToString(mac.doFinal(payload.toByteArray(StandardCharsets.UTF_8)))
    }

    data class Grant(val token: String, val expiresAt: OffsetDateTime)

    private companion object {
        val TTL: Duration = Duration.ofMinutes(15)
    }
}
