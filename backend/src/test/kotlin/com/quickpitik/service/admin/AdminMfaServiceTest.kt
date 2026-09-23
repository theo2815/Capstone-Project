package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.AdminMfa
import com.quickpitik.entity.AdminMfaRecoveryCode
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.AdminMfaRecoveryCodeRepository
import com.quickpitik.repository.AdminMfaRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// Real TotpService + MfaSecretCipher (so TOTP round-trips), mock repos.
class AdminMfaServiceTest {

    private lateinit var adminMfaRepository: AdminMfaRepository
    private lateinit var adminMfaRecoveryCodeRepository: AdminMfaRecoveryCodeRepository
    private lateinit var userRepository: UserRepository
    private val totpService = TotpService()
    private val cipher = MfaSecretCipher("dev-jwt-secret-for-mfa-service-tests-0123456789abcdef")
    private val userId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        adminMfaRepository = Mockito.mock(AdminMfaRepository::class.java)
        adminMfaRecoveryCodeRepository = Mockito.mock(AdminMfaRecoveryCodeRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        Mockito.`when`(adminMfaRepository.save(anyArg<AdminMfa>())).thenAnswer { it.getArgument(0) }
        Mockito.`when`(adminMfaRecoveryCodeRepository.saveAll(anyArg<List<AdminMfaRecoveryCode>>()))
            .thenAnswer { it.getArgument(0) }
        Mockito.`when`(userRepository.findById(anyArg())).thenReturn(Optional.empty())
    }

    private fun service(secretCipher: MfaSecretCipher = cipher) = AdminMfaService(
        adminMfaRepository, adminMfaRecoveryCodeRepository, totpService, secretCipher, userRepository,
    )

    private fun rowWith(secret: ByteArray, enabled: Boolean) =
        AdminMfa(userId = userId, secretEnc = cipher.encrypt(secret), enabled = enabled)

    private fun currentCode(secret: ByteArray) =
        totpService.generate(secret, System.currentTimeMillis() / 1000)

    @Test
    fun `enroll returns a provisioning uri and stores a disabled row`() {
        Mockito.`when`(adminMfaRepository.findById(userId)).thenReturn(Optional.empty())

        val res = service().enroll(userId)

        assertTrue(res.otpauthUri.startsWith("otpauth://totp/"))
        assertTrue(res.secret.isNotBlank())
        val captor = ArgumentCaptor.forClass(AdminMfa::class.java)
        Mockito.verify(adminMfaRepository).save(captor.capture())
        assertEquals(false, captor.value.enabled)
    }

    @Test
    fun `enroll is refused when MFA is already enabled`() {
        Mockito.`when`(adminMfaRepository.findById(userId))
            .thenReturn(Optional.of(rowWith(totpService.randomSecret(), enabled = true)))

        val ex = assertFailsWith<ApiException> { service().enroll(userId) }
        assertEquals(ErrorCodes.MFA_ALREADY_ENABLED, ex.code)
    }

    @Test
    fun `confirm with a valid code enables mfa and returns recovery codes`() {
        val secret = totpService.randomSecret()
        val row = rowWith(secret, enabled = false)
        Mockito.`when`(adminMfaRepository.findById(userId)).thenReturn(Optional.of(row))

        val res = service().confirm(userId, currentCode(secret))

        assertTrue(row.enabled)
        assertEquals(8, res.recoveryCodes.size)
    }

    @Test
    fun `confirm with a wrong code fails`() {
        val row = rowWith(totpService.randomSecret(), enabled = false)
        Mockito.`when`(adminMfaRepository.findById(userId)).thenReturn(Optional.of(row))

        val ex = assertFailsWith<ApiException> { service().confirm(userId, "000000") }
        assertEquals(ErrorCodes.INVALID_MFA_CODE, ex.code)
    }

    @Test
    fun `verifyCode accepts a valid totp for an enabled row`() {
        val secret = totpService.randomSecret()
        Mockito.`when`(adminMfaRepository.findById(userId)).thenReturn(Optional.of(rowWith(secret, enabled = true)))

        assertTrue(service().verifyCode(userId, currentCode(secret)))
    }

    @Test
    fun `verifyCode consumes a recovery code once`() {
        Mockito.`when`(adminMfaRepository.findById(userId))
            .thenReturn(Optional.of(rowWith(totpService.randomSecret(), enabled = true)))
        val recovery = AdminMfaRecoveryCode(userId = userId, codeHash = "hash")
        Mockito.`when`(adminMfaRecoveryCodeRepository.findByUserIdAndCodeHashAndUsedAtIsNull(anyArg(), anyArg()))
            .thenReturn(recovery)
        Mockito.`when`(adminMfaRecoveryCodeRepository.save(anyArg<AdminMfaRecoveryCode>())).thenAnswer { it.getArgument(0) }

        // 16 hex chars — not a 6-digit TOTP, so it falls through to the recovery path.
        assertTrue(service().verifyCode(userId, "abcdef0123456789"))
        assertTrue(recovery.usedAt != null)
    }

    @Test
    fun `verifyCode falls through to recovery when the totp key changed`() {
        val oldCipher = MfaSecretCipher("secret-A")
        val row = AdminMfa(
            userId = userId,
            secretEnc = oldCipher.encrypt(totpService.randomSecret()),
            enabled = true,
        )
        Mockito.`when`(adminMfaRepository.findById(userId)).thenReturn(Optional.of(row))
        val recovery = AdminMfaRecoveryCode(userId = userId, codeHash = "hash")
        Mockito.`when`(adminMfaRecoveryCodeRepository.findByUserIdAndCodeHashAndUsedAtIsNull(anyArg(), anyArg()))
            .thenReturn(recovery)
        Mockito.`when`(adminMfaRecoveryCodeRepository.save(anyArg<AdminMfaRecoveryCode>())).thenAnswer { it.getArgument(0) }

        assertTrue(service(MfaSecretCipher("secret-B")).verifyCode(userId, "abcdef0123456789"))
        assertTrue(recovery.usedAt != null)
    }
}
