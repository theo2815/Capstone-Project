package com.quickpitik.service.admin

import org.springframework.stereotype.Service
import java.net.URLEncoder
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * RFC 6238 TOTP (HMAC-SHA1, 30-second step, 6 digits) — hand-rolled on javax.crypto
 * so we take no new dependency. `verify` accepts the current window plus one step
 * either side for clock skew. Correctness is pinned to the RFC 6238 published test
 * vectors in TotpServiceTest.
 */
@Service
class TotpService {
    private val rng = SecureRandom()

    /** 160-bit random secret (RFC 6238 recommends >= 160 bits for SHA-1). */
    fun randomSecret(): ByteArray = ByteArray(20).also { rng.nextBytes(it) }

    fun generate(secret: ByteArray, timeSeconds: Long, digits: Int = DIGITS): String {
        val counter = Math.floorDiv(timeSeconds, STEP_SECONDS)
        val msg = ByteBuffer.allocate(8).putLong(counter).array()
        val mac = Mac.getInstance("HmacSHA1")
        mac.init(SecretKeySpec(secret, "HmacSHA1"))
        val hash = mac.doFinal(msg)
        val offset = hash[hash.size - 1].toInt() and 0x0f
        val binary = ((hash[offset].toInt() and 0x7f) shl 24) or
            ((hash[offset + 1].toInt() and 0xff) shl 16) or
            ((hash[offset + 2].toInt() and 0xff) shl 8) or
            (hash[offset + 3].toInt() and 0xff)
        return (binary % pow10(digits)).toString().padStart(digits, '0')
    }

    fun verify(
        secret: ByteArray,
        code: String,
        nowSeconds: Long = System.currentTimeMillis() / 1000,
        digits: Int = DIGITS,
    ): Boolean = matchingStep(secret, code, nowSeconds, digits) != null

    fun matchingStep(
        secret: ByteArray,
        code: String,
        nowSeconds: Long = System.currentTimeMillis() / 1000,
        digits: Int = DIGITS,
    ): Long? {
        val candidate = code.trim()
        if (candidate.length != digits || candidate.any { !it.isDigit() }) return null
        for (delta in -1..1) {
            val expected = generate(secret, nowSeconds + delta * STEP_SECONDS, digits)
            if (MessageDigest.isEqual(expected.toByteArray(), candidate.toByteArray())) {
                return Math.floorDiv(nowSeconds, STEP_SECONDS) + delta
            }
        }
        return null
    }

    /** RFC 4648 base32, uppercase, unpadded — the form authenticator apps expect. */
    fun base32(data: ByteArray): String {
        val sb = StringBuilder()
        var buffer = 0
        var bits = 0
        for (b in data) {
            buffer = (buffer shl 8) or (b.toInt() and 0xff)
            bits += 8
            while (bits >= 5) {
                sb.append(BASE32[(buffer shr (bits - 5)) and 0x1f])
                bits -= 5
            }
        }
        if (bits > 0) sb.append(BASE32[(buffer shl (5 - bits)) and 0x1f])
        return sb.toString()
    }

    fun otpauthUri(secret: ByteArray, accountLabel: String, issuer: String = "QuickPitik"): String {
        val label = URLEncoder.encode("$issuer:$accountLabel", StandardCharsets.UTF_8)
        val iss = URLEncoder.encode(issuer, StandardCharsets.UTF_8)
        return "otpauth://totp/$label?secret=${base32(secret)}&issuer=$iss&digits=$DIGITS&period=$STEP_SECONDS&algorithm=SHA1"
    }

    private fun pow10(n: Int): Int {
        var r = 1
        repeat(n) { r *= 10 }
        return r
    }

    companion object {
        const val STEP_SECONDS = 30L
        const val DIGITS = 6
        private const val BASE32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    }
}
