package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.admin.MfaConfirmResponse
import com.quickpitik.dto.admin.MfaEnrollResponse
import com.quickpitik.dto.admin.MfaStatusResponse
import com.quickpitik.entity.AdminMfa
import com.quickpitik.entity.AdminMfaRecoveryCode
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.AdminMfaRecoveryCodeRepository
import com.quickpitik.repository.AdminMfaRepository
import com.quickpitik.repository.UserRepository
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.security.MessageDigest
import java.security.SecureRandom
import java.time.OffsetDateTime
import java.util.UUID

/**
 * Opt-in admin TOTP MFA. Enroll → confirm (enables + returns one-time recovery
 * codes) → the login flow (Phase 5b) then challenges enrolled admins. Nothing
 * here touches login, so building it can't break authentication.
 */
@Service
@Transactional
class AdminMfaService(
    private val adminMfaRepository: AdminMfaRepository,
    private val adminMfaRecoveryCodeRepository: AdminMfaRecoveryCodeRepository,
    private val totpService: TotpService,
    private val mfaSecretCipher: MfaSecretCipher,
    private val userRepository: UserRepository,
) {
    @Transactional(readOnly = true)
    fun status(userId: UUID): MfaStatusResponse = MfaStatusResponse(enabled = isEnabled(userId))

    @Transactional(readOnly = true)
    fun isEnabled(userId: UUID): Boolean =
        adminMfaRepository.findById(userId).map { it.enabled }.orElse(false)

    fun enroll(userId: UUID): MfaEnrollResponse {
        val existing = adminMfaRepository.findById(userId).orElse(null)
        if (existing?.enabled == true) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.MFA_ALREADY_ENABLED, "MFA is already enabled.")
        }
        val secret = totpService.randomSecret()
        val row = existing ?: AdminMfa(userId = userId, secretEnc = "")
        row.secretEnc = mfaSecretCipher.encrypt(secret)
        row.enabled = false
        row.confirmedAt = null
        adminMfaRepository.save(row)
        val email = userRepository.findById(userId).map { it.email }.orElse("admin")
        return MfaEnrollResponse(
            otpauthUri = totpService.otpauthUri(secret, email),
            secret = totpService.base32(secret),
        )
    }

    fun confirm(userId: UUID, code: String): MfaConfirmResponse {
        val row = adminMfaRepository.findById(userId).orElseThrow {
            ApiException(HttpStatus.CONFLICT, ErrorCodes.MFA_NOT_ENROLLED, "Start MFA enrollment first.")
        }
        val secret = mfaSecretCipher.decrypt(row.secretEnc)
        if (!totpService.verify(secret, code)) {
            throw ApiException(HttpStatus.UNAUTHORIZED, ErrorCodes.INVALID_MFA_CODE, "That code is not valid.")
        }
        row.enabled = true
        row.confirmedAt = OffsetDateTime.now()
        adminMfaRepository.save(row)
        // Fresh single-use recovery codes; only the hashes are stored.
        adminMfaRecoveryCodeRepository.deleteByUserId(userId)
        val codes = List(RECOVERY_COUNT) { randomRecoveryCode() }
        adminMfaRecoveryCodeRepository.saveAll(
            codes.map { AdminMfaRecoveryCode(userId = userId, codeHash = sha256Hex(it)) },
        )
        return MfaConfirmResponse(recoveryCodes = codes)
    }

    fun disable(userId: UUID, code: String): Map<String, Boolean> {
        val row = adminMfaRepository.findById(userId).orElseThrow {
            ApiException(HttpStatus.CONFLICT, ErrorCodes.MFA_NOT_ENROLLED, "MFA is not set up.")
        }
        if (!verifyAgainst(userId, code, row)) {
            throw ApiException(HttpStatus.UNAUTHORIZED, ErrorCodes.INVALID_MFA_CODE, "That code is not valid.")
        }
        adminMfaRecoveryCodeRepository.deleteByUserId(userId)
        adminMfaRepository.delete(row)
        return mapOf("disabled" to true)
    }

    /** Login-verify (Phase 5b): true if the code is a valid TOTP or an unused recovery code. */
    fun verifyCode(userId: UUID, code: String): Boolean {
        val row = adminMfaRepository.findById(userId).orElse(null) ?: return false
        if (!row.enabled) return false
        return verifyAgainst(userId, code, row)
    }

    private fun verifyAgainst(userId: UUID, code: String, row: AdminMfa): Boolean {
        val secret = mfaSecretCipher.decrypt(row.secretEnc)
        if (totpService.verify(secret, code)) return true
        return consumeRecovery(userId, code)
    }

    private fun consumeRecovery(userId: UUID, code: String): Boolean {
        val match = adminMfaRecoveryCodeRepository
            .findByUserIdAndCodeHashAndUsedAtIsNull(userId, sha256Hex(code.trim()))
            ?: return false
        match.usedAt = OffsetDateTime.now()
        adminMfaRecoveryCodeRepository.save(match)
        return true
    }

    private fun randomRecoveryCode(): String {
        val bytes = ByteArray(8).also { rng.nextBytes(it) }
        return bytes.joinToString("") { "%02x".format(it) }
    }

    private fun sha256Hex(s: String): String =
        MessageDigest.getInstance("SHA-256").digest(s.toByteArray()).joinToString("") { "%02x".format(it) }

    private companion object {
        val rng = SecureRandom()
        const val RECOVERY_COUNT = 8
    }
}
