package com.quickpitik.service.photographer

import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.CreatePayoutRequest
import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.PayoutAccountAudit
import com.quickpitik.repository.PayoutAccountAuditRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import java.util.UUID
import kotlin.test.assertEquals

// Phase 3a — every payout-account change writes an append-only audit row, which
// also drives the post-change payout cooldown. This covers the create path;
// patch/delete/set-primary/qr call the same recordChange().
class PayoutAccountServiceAuditTest {

    private lateinit var payoutAccountRepository: PayoutAccountRepository
    private lateinit var payoutAccountAuditRepository: PayoutAccountAuditRepository
    private lateinit var storageService: StorageService
    private lateinit var storageProperties: StorageProperties

    private val userId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        payoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)
        payoutAccountAuditRepository = Mockito.mock(PayoutAccountAuditRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        storageProperties = Mockito.mock(StorageProperties::class.java)
    }

    private fun service() = PayoutAccountService(
        payoutAccountRepository,
        payoutAccountAuditRepository,
        storageService,
        storageProperties,
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
}
