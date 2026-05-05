package com.quickpitik.security

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64

object OpaqueTokens {
    private val random = SecureRandom()

    fun generate(byteLength: Int = 32): String {
        val bytes = ByteArray(byteLength)
        random.nextBytes(bytes)
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
    }

    fun hash(token: String): String {
        val md = MessageDigest.getInstance("SHA-256")
        val digest = md.digest(token.toByteArray(StandardCharsets.UTF_8))
        return Base64.getUrlEncoder().withoutPadding().encodeToString(digest)
    }
}
