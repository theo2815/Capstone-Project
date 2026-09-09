package com.quickpitik.service.earnings

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.PayoutAccountAudit
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.PayoutStatusHistory
import com.quickpitik.entity.PayoutMethod
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountAuditRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PayoutStatusHistoryRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.repository.TransactionRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// Phase 1 — the withdrawable balance is reservation- and clearing-aware:
//   available = max(0, cleared − paid − reserved)
// where `cleared` = sales aged past the clearing window plus ALL refunds,
// `reserved` = amounts tied up in open cycles (pending/held), `paid` = settled
// cycles. request() must key off this available amount, never lifetime earnings.
class PayoutRequestServiceBalanceTest {

    private lateinit var payoutCycleRepository: PayoutCycleRepository
    private lateinit var transactionRepository: TransactionRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var payoutAccountRepository: PayoutAccountRepository
    private lateinit var payoutAccountAuditRepository: PayoutAccountAuditRepository
    private lateinit var payoutStatusHistoryRepository: PayoutStatusHistoryRepository
    private lateinit var payoutStatusRecorder: PayoutStatusRecorder
    private val platformProperties = PlatformProperties()

    private val photographerId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        payoutCycleRepository = Mockito.mock(PayoutCycleRepository::class.java)
        transactionRepository = Mockito.mock(TransactionRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        payoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)
        payoutAccountAuditRepository = Mockito.mock(PayoutAccountAuditRepository::class.java)
        payoutStatusHistoryRepository = Mockito.mock(PayoutStatusHistoryRepository::class.java)
        payoutStatusRecorder = PayoutStatusRecorder(payoutStatusHistoryRepository)
    }

    private fun service() = PayoutRequestService(
        payoutCycleRepository,
        transactionRepository,
        photographerSettingsRepository,
        payoutAccountRepository,
        payoutAccountAuditRepository,
        payoutStatusRecorder,
        platformProperties,
        Mockito.mock(UserRepository::class.java),
    )

    // Stub the four money sums; each test sets the numbers it cares about.
    private fun stubSums(cleared: String, lifetime: String, paid: String, reserved: String) {
        Mockito.`when`(transactionRepository.sumCleared(anyArg(), anyArg()))
            .thenReturn(BigDecimal(cleared))
        Mockito.`when`(transactionRepository.sumLifetime(photographerId))
            .thenReturn(BigDecimal(lifetime))
        Mockito.`when`(payoutCycleRepository.sumPaidByPhotographer(photographerId))
            .thenReturn(BigDecimal(paid))
        Mockito.`when`(payoutCycleRepository.sumReservedByPhotographer(photographerId))
            .thenReturn(BigDecimal(reserved))
    }

    private fun noOpenRequest() {
        Mockito.`when`(payoutCycleRepository.findFirstByPhotographerIdAndStatusWireIn(anyArg(), anyArg()))
            .thenReturn(null)
    }

    private fun assertAmount(expected: String, actual: BigDecimal) =
        assertEquals(0, BigDecimal(expected).compareTo(actual), "expected $expected but was $actual")

    @Test
    fun `an account without a QR is incomplete and cannot request money`() {
        stubSums(cleared = "1000", lifetime = "1000", paid = "0", reserved = "0")
        noOpenRequest()
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)).thenReturn(
            PayoutAccount(userId = photographerId, method = PayoutMethod.GCASH,
                accountNumber = "09171234567", accountName = "Test", isPrimary = true),
        )
        assertEquals(false, service().getBalance(photographerId).primaryAccount?.hasQr)
        assertEquals(ErrorCodes.PAYOUT_QR_REQUIRED, assertFailsWith<ValidationException> { service().request(photographerId) }.code)
        Mockito.verify(payoutCycleRepository, Mockito.never()).saveAndFlush(anyArg<PayoutCycle>())
    }

    @Test
    fun `unpaidBalance excludes funds reserved by an open cycle`() {
        stubSums(cleared = "1000", lifetime = "1000", paid = "0", reserved = "1000")
        noOpenRequest()
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)).thenReturn(null)

        val balance = service().getBalance(photographerId)

        assertAmount("0", balance.unpaidBalance)
        assertAmount("1000", balance.reserved)
    }

    @Test
    fun `unpaidBalance excludes earnings still within the clearing window`() {
        stubSums(cleared = "600", lifetime = "1000", paid = "0", reserved = "0")
        noOpenRequest()
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)).thenReturn(null)

        val balance = service().getBalance(photographerId)

        assertAmount("600", balance.unpaidBalance)
        assertAmount("400", balance.clearing)
    }

    @Test
    fun `unpaidBalance never goes negative when payouts exceed cleared earnings`() {
        // e.g. a refund landed after a prior payout: paid > cleared.
        stubSums(cleared = "500", lifetime = "500", paid = "800", reserved = "0")
        noOpenRequest()
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)).thenReturn(null)

        val balance = service().getBalance(photographerId)

        assertAmount("0", balance.unpaidBalance)
    }

    // The clearing-window CUTOFF (that `cleared` really is "sales aged past
    // now − window") is verified against real dated rows in the Phase-9
    // integration suite, not here — asserting the exact mock argument is a
    // brittle mock-interaction test. These unit tests cover the balance MATH.

    @Test
    fun `request pays out the available amount, not lifetime earnings`() {
        stubSums(cleared = "600", lifetime = "1000", paid = "0", reserved = "0")
        noOpenRequest()
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId))
            .thenReturn(
                PayoutAccount(
                    userId = photographerId,
                    method = PayoutMethod.GCASH,
                    accountNumber = "09171234567",
                    accountName = "Juan Dela Cruz",
                    qrS3Key = "test/qr.png",
                ),
            )
        Mockito.`when`(photographerSettingsRepository.findById(photographerId)).thenReturn(Optional.empty())
        Mockito.`when`(payoutCycleRepository.countByPhotographerId(photographerId)).thenReturn(0L)
        Mockito.`when`(transactionRepository.countSalesWindow(anyArg(), anyArg(), anyArg())).thenReturn(3L)
        Mockito.`when`(payoutCycleRepository.saveAndFlush(anyArg<PayoutCycle>())).thenAnswer { it.getArgument(0) }
        val captor = ArgumentCaptor.forClass(PayoutCycle::class.java)

        service().request(photographerId)

        Mockito.verify(payoutCycleRepository).saveAndFlush(captor.capture())
        assertAmount("600", captor.value.amountPhp)
    }

    @Test
    fun `request is rejected when available is below the minimum even if lifetime is not`() {
        // ₱1000 lifetime but only ₱400 cleared → below the ₱500 minimum.
        stubSums(cleared = "400", lifetime = "1000", paid = "0", reserved = "0")
        noOpenRequest()

        val ex = assertFailsWith<ValidationException> { service().request(photographerId) }

        assertEquals(ErrorCodes.PAYOUT_BELOW_MINIMUM, ex.code)
    }

    @Test
    fun `request maps a duplicate-open constraint violation to a 409`() {
        // The findFirst check passes (no open cycle yet), but the DB partial
        // unique index rejects the insert — this request lost the race. It must
        // surface the same clean 409 as the pre-check, not a raw 500.
        stubSums(cleared = "1000", lifetime = "1000", paid = "0", reserved = "0")
        noOpenRequest()
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId))
            .thenReturn(
                PayoutAccount(
                    userId = photographerId,
                    method = PayoutMethod.GCASH,
                    accountNumber = "09171234567",
                    accountName = "Juan Dela Cruz",
                    qrS3Key = "test/qr.png",
                ),
            )
        Mockito.`when`(photographerSettingsRepository.findById(photographerId)).thenReturn(Optional.empty())
        Mockito.`when`(payoutCycleRepository.countByPhotographerId(photographerId)).thenReturn(0L)
        Mockito.`when`(transactionRepository.countSalesWindow(anyArg(), anyArg(), anyArg())).thenReturn(3L)
        Mockito.`when`(payoutCycleRepository.saveAndFlush(anyArg<PayoutCycle>()))
            .thenThrow(DataIntegrityViolationException("uq_payout_open_per_photographer"))

        val ex = assertFailsWith<ApiException> { service().request(photographerId) }

        assertEquals(ErrorCodes.PAYOUT_REQUEST_OPEN, ex.code)
        assertEquals(HttpStatus.CONFLICT, ex.status)
    }

    @Test
    fun `request is blocked during the payout-account cooldown`() {
        // Balance + account are fine; the recent payout-account change bars the
        // request (a compromised account can't change-bank-then-withdraw).
        stubSums(cleared = "1000", lifetime = "1000", paid = "0", reserved = "0")
        noOpenRequest()
        Mockito.`when`(payoutAccountAuditRepository.existsByUserIdAndCreatedAtAfter(anyArg(), anyArg()))
            .thenReturn(true)

        val ex = assertFailsWith<ApiException> { service().request(photographerId) }

        assertEquals(ErrorCodes.PAYOUT_ACCOUNT_COOLDOWN, ex.code)
    }

    @Test
    fun `getBalance surfaces the account cooldown-until after a recent change`() {
        stubSums(cleared = "1000", lifetime = "1000", paid = "0", reserved = "0")
        noOpenRequest()
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId)).thenReturn(null)
        Mockito.`when`(payoutAccountAuditRepository.findFirstByUserIdOrderByCreatedAtDesc(photographerId))
            .thenReturn(PayoutAccountAudit(userId = photographerId, action = "update", createdAt = OffsetDateTime.now()))

        val until = service().getBalance(photographerId).accountCooldownUntil

        assertTrue(until != null && until.isAfter(OffsetDateTime.now()), "expected a future cooldown-until, was $until")
    }

    @Test
    fun `request records a pending status-history entry`() {
        stubSums(cleared = "600", lifetime = "1000", paid = "0", reserved = "0")
        noOpenRequest()
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(photographerId))
            .thenReturn(
                PayoutAccount(
                    userId = photographerId,
                    method = PayoutMethod.GCASH,
                    accountNumber = "09171234567",
                    accountName = "Juan Dela Cruz",
                    qrS3Key = "test/qr.png",
                ),
            )
        Mockito.`when`(photographerSettingsRepository.findById(photographerId)).thenReturn(Optional.empty())
        Mockito.`when`(payoutCycleRepository.countByPhotographerId(photographerId)).thenReturn(0L)
        Mockito.`when`(transactionRepository.countSalesWindow(anyArg(), anyArg(), anyArg())).thenReturn(3L)
        Mockito.`when`(payoutCycleRepository.saveAndFlush(anyArg<PayoutCycle>())).thenAnswer { it.getArgument(0) }

        service().request(photographerId)

        val captor = ArgumentCaptor.forClass(PayoutStatusHistory::class.java)
        Mockito.verify(payoutStatusHistoryRepository).save(captor.capture())
        assertEquals(PayoutCycleStatus.PENDING.wire, captor.value.toStatus)
    }
}
