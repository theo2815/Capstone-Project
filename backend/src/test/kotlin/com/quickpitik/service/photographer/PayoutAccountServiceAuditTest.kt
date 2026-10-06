package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.CreatePayoutRequest
import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.PayoutAccountAudit
import com.quickpitik.entity.PayoutMethod
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountAuditRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import java.util.Base64
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Phase 3a — every payout-account change writes an append-only audit row, which
// also drives the post-change payout cooldown. This covers the create path;
// patch/delete/set-primary/qr call the same recordChange().
class PayoutAccountServiceAuditTest {

    private lateinit var payoutAccountRepository: PayoutAccountRepository
    private lateinit var payoutAccountAuditRepository: PayoutAccountAuditRepository
    private lateinit var storageService: StorageService
    private lateinit var storageProperties: StorageProperties
    private lateinit var eventPublisher: ApplicationEventPublisher

    private val userId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        payoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)
        payoutAccountAuditRepository = Mockito.mock(PayoutAccountAuditRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        storageProperties = Mockito.mock(StorageProperties::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
    }

    private fun service() = PayoutAccountService(
        payoutAccountRepository,
        payoutAccountAuditRepository,
        storageService,
        storageProperties,
        eventPublisher,
        Mockito.mock(com.quickpitik.repository.PayoutCycleRepository::class.java),
        Mockito.mock(com.quickpitik.repository.UserRepository::class.java),
    )

    @Test
    fun `creating a payout account writes a masked audit row`() {
        Mockito.`when`(payoutAccountRepository.countByUserId(userId)).thenReturn(0L)
        Mockito.`when`(payoutAccountRepository.save(anyArg<PayoutAccount>())).thenAnswer { it.getArgument(0) }

        service().create(
            userId,
            CreatePayoutRequest(method = "gcash", accountNumber = "09171234567", accountName = "Juan Dela Cruz"),
        )

        val captor = ArgumentCaptor.forClass(PayoutAccountAudit::class.java)
        Mockito.verify(payoutAccountAuditRepository).save(captor.capture())
        assertEquals("create", captor.value.action)
        assertEquals("gcash", captor.value.method)
        assertEquals("••••4567", captor.value.accountMaskedNew)
    }

    @Test
    fun `creating a payout account publishes an owner-alert event`() {
        Mockito.`when`(payoutAccountRepository.countByUserId(userId)).thenReturn(0L)
        Mockito.`when`(payoutAccountRepository.save(anyArg<PayoutAccount>())).thenAnswer { it.getArgument(0) }

        service().create(
            userId,
            CreatePayoutRequest(method = "gcash", accountNumber = "09171234567", accountName = "Juan Dela Cruz"),
        )

        val captor = ArgumentCaptor.forClass(PayoutAccountChangedEvent::class.java)
        Mockito.verify(eventPublisher).publishEvent(captor.capture())
        assertEquals("create", captor.value.action)
        assertEquals("••••4567", captor.value.maskedAccount)
    }
    @Test
    fun `truncated WebP QR is rejected before storage or account writes`() {
        val account = PayoutAccount(userId = userId, method = PayoutMethod.GCASH, accountNumber = "09171234567", accountName = "Juan Dela Cruz", qrS3Key = "previous.png")
        Mockito.`when`(payoutAccountRepository.findByIdAndUserId(account.id, userId)).thenReturn(account)
        val bytes = Base64.getDecoder().decode("UklGRjIAAABXRUJQVlA4ICYAAAA=")

        val ex = assertFailsWith<ValidationException> {
            service().uploadQr(userId, account.id, bytes, "image/webp")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
        assertEquals("file", ex.field)
        assertEquals("previous.png", account.qrS3Key)
        Mockito.verifyNoInteractions(storageService, payoutAccountAuditRepository, eventPublisher)
        Mockito.verify(payoutAccountRepository, Mockito.never()).save(Mockito.any(PayoutAccount::class.java))
    }

    @Test
    fun `uploading a QR over the pixel budget is rejected before storage or save`() {
        val account = PayoutAccount(userId = userId, method = PayoutMethod.GCASH, accountNumber = "09171234567", accountName = "Juan Dela Cruz")
        Mockito.`when`(payoutAccountRepository.findByIdAndUserId(account.id, userId)).thenReturn(account)

        val error = assertFailsWith<ValidationException> {
            service().uploadQr(userId, account.id, overBudgetPngHeader(), "image/png")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
        assertEquals("Image dimensions exceed the supported maximum.", error.message)
        assertEquals("qr", error.field)
        Mockito.verifyNoInteractions(storageService)
        Mockito.verify(payoutAccountRepository, Mockito.never()).save(Mockito.any(PayoutAccount::class.java))
    }

    // 10,000 × 8,001 = 80,010,000 pixels, above ImagePixelGuard.MAX_PIXELS.
    private fun overBudgetPngHeader(): ByteArray = byteArrayOf(
        -119, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 39, 16, 0, 0, 31, 65, 8, 6, 0, 0, 0, -26, 16, 82, 64,
    )
}
