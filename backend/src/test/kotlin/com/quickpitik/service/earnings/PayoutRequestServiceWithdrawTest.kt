package com.quickpitik.service.earnings

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.TransactionRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// withdraw() hard-deletes the cycle, so any client retry (double-click, flaky
// network) hits a row that is already gone. It must be idempotent — matching
// SelfieService.delete / SocialLinkService.delete / PayoutAccountService.delete,
// which all no-op rather than 404. A non-HELD cycle is still a genuine state
// error and must keep its 409.
class PayoutRequestServiceWithdrawTest {

    private lateinit var payoutCycleRepository: PayoutCycleRepository
    private lateinit var transactionRepository: TransactionRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var payoutAccountRepository: PayoutAccountRepository

    private val photographerId = UUID.randomUUID()
    private val cycleId = "PAY-JUANDC-001"

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        payoutCycleRepository = Mockito.mock(PayoutCycleRepository::class.java)
        transactionRepository = Mockito.mock(TransactionRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        payoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)
    }

    private fun service() = PayoutRequestService(
        payoutCycleRepository,
        transactionRepository,
        photographerSettingsRepository,
        payoutAccountRepository,
    )

    private fun cycle(status: PayoutCycleStatus): PayoutCycle {
        val c = PayoutCycle(
            id = cycleId,
            photographerId = photographerId,
            weekOf = LocalDate.of(2026, 8, 10),
            methodWire = "gcash",
        )
        c.status = status
        return c
    }

    @Test
    fun `withdrawing an already-withdrawn cycle is a silent no-op, not a 404`() {
        Mockito.`when`(payoutCycleRepository.findByIdAndPhotographerId(cycleId, photographerId))
            .thenReturn(null)

        // The assertion is simply that this does not throw.
        service().withdraw(photographerId, cycleId)

        Mockito.verify(payoutCycleRepository, Mockito.never()).delete(anyArg())
    }

    @Test
    fun `withdrawing a held cycle deletes it`() {
        Mockito.`when`(payoutCycleRepository.findByIdAndPhotographerId(cycleId, photographerId))
            .thenReturn(cycle(PayoutCycleStatus.HELD))

        service().withdraw(photographerId, cycleId)

        Mockito.verify(payoutCycleRepository).delete(anyArg())
    }

    @Test
    fun `withdrawing a non-held cycle still conflicts`() {
        Mockito.`when`(payoutCycleRepository.findByIdAndPhotographerId(cycleId, photographerId))
            .thenReturn(cycle(PayoutCycleStatus.PENDING))

        val ex = assertFailsWith<ApiException> { service().withdraw(photographerId, cycleId) }

        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        Mockito.verify(payoutCycleRepository, Mockito.never()).delete(anyArg())
    }
}
