package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.MarkPayoutPaidRequest
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.PayoutStatusHistory
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PayoutStatusHistoryRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.earnings.PayoutStatusRecorder
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Phase 4b — the admin payout state machine. Each transition flips the real
// status, records a payout_status_history row, and guards invalid source states.
// Uses a real PayoutStatusRecorder + AdminDecisionLogService over mock repos so
// the history write is asserted via a single-arg save captor (avoids the
// Mockito+Kotlin non-null-enum matcher NPE).
class AdminPayoutServiceTransitionTest {

    private lateinit var payoutCycleRepository: PayoutCycleRepository
    private lateinit var payoutAccountRepository: PayoutAccountRepository
    private lateinit var userRepository: UserRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var transactionRepository: TransactionRepository
    private lateinit var storageService: StorageService
    private lateinit var storageProperties: StorageProperties
    private lateinit var adminDecisionLogRepository: AdminDecisionLogRepository
    private lateinit var photographerMessageRepository: PhotographerMessageRepository
    private lateinit var eventPublisher: ApplicationEventPublisher
    private lateinit var payoutStatusHistoryRepository: PayoutStatusHistoryRepository

    private val adminId = UUID.randomUUID()
    private val photographerId = UUID.randomUUID()
    private val cycleId = "PAY-JUANDC-001"

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        payoutCycleRepository = Mockito.mock(PayoutCycleRepository::class.java)
        payoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        transactionRepository = Mockito.mock(TransactionRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        storageProperties = Mockito.mock(StorageProperties::class.java)
        adminDecisionLogRepository = Mockito.mock(AdminDecisionLogRepository::class.java)
        photographerMessageRepository = Mockito.mock(PhotographerMessageRepository::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
        payoutStatusHistoryRepository = Mockito.mock(PayoutStatusHistoryRepository::class.java)

        // Persistence stubs shared by every transition.
        Mockito.`when`(payoutCycleRepository.save(anyArg<PayoutCycle>())).thenAnswer { it.getArgument(0) }
        Mockito.`when`(adminDecisionLogRepository.save(anyArg())).thenAnswer { it.getArgument(0) }
        Mockito.`when`(photographerMessageRepository.save(anyArg())).thenAnswer { it.getArgument(0) }
        Mockito.`when`(userRepository.findById(anyArg())).thenReturn(Optional.empty())
        Mockito.`when`(photographerSettingsRepository.findById(anyArg())).thenReturn(Optional.empty())
        Mockito.`when`(payoutAccountRepository.findByUserIdAndIsPrimaryTrue(anyArg())).thenReturn(null)
        Mockito.`when`(payoutAccountRepository.findAllByUserIdOrderByCreatedAtAsc(anyArg())).thenReturn(emptyList())
    }

    private fun service(): AdminPayoutService {
        val adminDecisionLogService = AdminDecisionLogService(
            adminDecisionLogRepository, photographerMessageRepository, eventPublisher,
        )
        val recorder = PayoutStatusRecorder(payoutStatusHistoryRepository)
        return AdminPayoutService(
            payoutCycleRepository,
            payoutAccountRepository,
            userRepository,
            photographerSettingsRepository,
            transactionRepository,
            storageService,
            storageProperties,
            adminDecisionLogService,
            adminDecisionLogRepository,
            recorder,
        )
    }

    private fun cycle(status: PayoutCycleStatus): PayoutCycle {
        val c = PayoutCycle(
            id = cycleId,
            photographerId = photographerId,
            weekOf = LocalDate.of(2026, 9, 1),
            methodWire = "gcash",
        )
        c.status = status
        c.amountPhp = java.math.BigDecimal("1000")
        return c
    }

    private fun given(status: PayoutCycleStatus) {
        Mockito.`when`(payoutCycleRepository.findById(cycleId)).thenReturn(Optional.of(cycle(status)))
    }

    private fun recordedToStatus(): String {
        val captor = ArgumentCaptor.forClass(PayoutStatusHistory::class.java)
        Mockito.verify(payoutStatusHistoryRepository).save(captor.capture())
        return captor.value.toStatus
    }

    @Test
    fun `approve moves pending to approved and records history`() {
        given(PayoutCycleStatus.PENDING)
        val dto = service().approve(adminId, cycleId)
        assertEquals("approved", dto.status)
        assertEquals(PayoutCycleStatus.APPROVED.wire, recordedToStatus())
    }

    @Test
    fun `reject moves pending to rejected with a reason`() {
        given(PayoutCycleStatus.PENDING)
        val dto = service().reject(adminId, cycleId, "duplicate request")
        assertEquals("rejected", dto.status)
        assertEquals(PayoutCycleStatus.REJECTED.wire, recordedToStatus())
    }

    @Test
    fun `mark-processing moves approved to processing`() {
        given(PayoutCycleStatus.APPROVED)
        service().markProcessing(adminId, cycleId)
        assertEquals(PayoutCycleStatus.PROCESSING.wire, recordedToStatus())
    }

    @Test
    fun `mark-failed moves processing to failed`() {
        given(PayoutCycleStatus.PROCESSING)
        service().markFailed(adminId, cycleId, "instapay rejected")
        assertEquals(PayoutCycleStatus.FAILED.wire, recordedToStatus())
    }

    @Test
    fun `mark-returned moves paid to returned`() {
        given(PayoutCycleStatus.PAID)
        service().markReturned(adminId, cycleId, "account closed")
        assertEquals(PayoutCycleStatus.RETURNED.wire, recordedToStatus())
    }

    @Test
    fun `mark-paid moves approved to paid`() {
        given(PayoutCycleStatus.APPROVED)
        val dto = service().markPaid(adminId, cycleId, MarkPayoutPaidRequest(paymentReference = "GC-123456"))
        assertEquals("paid", dto.status)
        assertEquals(PayoutCycleStatus.PAID.wire, recordedToStatus())
    }

    @Test
    fun `approve is rejected from a terminal state`() {
        given(PayoutCycleStatus.PAID)
        val ex = assertFailsWith<ApiException> { service().approve(adminId, cycleId) }
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
    }

    @Test
    fun `mark-processing is rejected when not approved`() {
        given(PayoutCycleStatus.PENDING)
        val ex = assertFailsWith<ApiException> { service().markProcessing(adminId, cycleId) }
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
    }
}
