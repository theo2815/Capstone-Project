package com.quickpitik.service.admin

import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Component
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * AES-256-GCM at-rest encryption for TOTP secrets. The key is DERIVED from the
 * already-required, prod-guarded JWT_SECRET via SHA-256 with a domain-separation
 * tag — so it is a distinct key from the JWT signing key, and no new managed
 * secret (or ProductionSecretsGuard entry) is introduced. Output is
 * base64(iv || ciphertext||tag); a fresh 96-bit IV per encryption.
 */
@Component
class MfaSecretCipher(
    @Value("\${jwt.secret}") jwtSecret: String,
) {
    private val key = SecretKeySpec(
        MessageDigest.getInstance("SHA-256").digest((jwtSecret + KEY_DOMAIN).toByteArray()),
        "AES",
    )
    private val rng = SecureRandom()

    fun encrypt(plaintext: ByteArray): String {
        val iv = ByteArray(IV_BYTES).also { rng.nextBytes(it) }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key, GCMParameterSpec(TAG_BITS, iv))
        val ct = cipher.doFinal(plaintext)
        return Base64.getEncoder().encodeToString(iv + ct)
    }

    fun decrypt(stored: String): ByteArray {
        val all = Base64.getDecoder().decode(stored)
        val iv = all.copyOfRange(0, IV_BYTES)
        val ct = all.copyOfRange(IV_BYTES, all.size)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, key, GCMParameterSpec(TAG_BITS, iv))
        return cipher.doFinal(ct)
    }

    private companion object {
        const val KEY_DOMAIN = "|mfa-totp-v1"
        const val IV_BYTES = 12
        const val TAG_BITS = 128
    }
}
