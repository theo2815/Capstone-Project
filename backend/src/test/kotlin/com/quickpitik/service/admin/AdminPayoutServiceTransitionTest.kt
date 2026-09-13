package com.quickpitik.service.admin

import ch.qos.logback.classic.Level
import ch.qos.logback.classic.Logger
import ch.qos.logback.classic.spi.ILoggingEvent
import ch.qos.logback.classic.spi.ThrowableProxy
import ch.qos.logback.core.read.ListAppender
import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.MarkPayoutPaidRequest
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.PayoutStatusHistory
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.Transaction
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PayoutStatusHistoryRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.disbursement.DisbursementRequest
import com.quickpitik.service.disbursement.DisbursementResult
import com.quickpitik.service.disbursement.DisbursementService
import com.quickpitik.service.disbursement.ManualDisbursementProvider
import com.quickpitik.service.earnings.PayoutCycleTransitions
import com.quickpitik.service.earnings.PayoutRequestService
import com.quickpitik.service.earnings.PayoutStatusRecorder
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.PostgresIntegrationTest
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean
import org.springframework.test.util.AopTestUtils
import org.springframework.transaction.support.TransactionSynchronizationManager
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

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

    private fun service(
        disbursementService: DisbursementService = DisbursementService(
            listOf(ManualDisbursementProvider()), PlatformProperties(),
        ),
        payoutRequestService: PayoutRequestService = Mockito.mock(PayoutRequestService::class.java),
    ): AdminPayoutService {
        val adminDecisionLogService = AdminDecisionLogService(
            adminDecisionLogRepository, photographerMessageRepository, eventPublisher,
        )
        val recorder = PayoutStatusRecorder(payoutStatusHistoryRepository)
        val props = PlatformProperties()
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
            PayoutCycleTransitions(payoutCycleRepository, recorder),
            props,
            disbursementService,
            payoutRequestService,
            testTransactionTemplate(),
        )
    }

    private fun cycle(status: PayoutCycleStatus, amount: String = "1000", approvedBy: UUID? = null): PayoutCycle {
        val c = PayoutCycle(
            id = cycleId,
            photographerId = photographerId,
            weekOf = LocalDate.of(2026, 9, 1),
            methodWire = "gcash",
        )
        c.status = status
        c.amountPhp = BigDecimal(amount)
        c.approvedBy = approvedBy
        return c
    }

    private fun given(status: PayoutCycleStatus, amount: String = "1000", approvedBy: UUID? = null) {
        Mockito.`when`(payoutCycleRepository.lockById(cycleId))
            .thenReturn(cycle(status, amount, approvedBy))
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
    fun `reject refuses a provider claimed payout before mutating its reasons`() {
        val cycle = cycle(PayoutCycleStatus.APPROVED).apply {
            provider = "paymongo"
            holdReason = "original hold"
            rejectedReason = "original rejection"
        }
        Mockito.`when`(payoutCycleRepository.lockById(cycleId)).thenReturn(cycle)

        val ex = assertFailsWith<ApiException> { service().reject(adminId, cycleId, "new rejection") }

        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertEquals(PayoutCycleStatus.APPROVED, cycle.status)
        assertEquals("original hold", cycle.holdReason)
        assertEquals("original rejection", cycle.rejectedReason)
        Mockito.verify(payoutCycleRepository, Mockito.never()).save(anyArg<PayoutCycle>())
        Mockito.verifyNoInteractions(payoutStatusHistoryRepository, adminDecisionLogRepository, photographerMessageRepository)
    }

    @Test
    fun `mark-processing moves approved to processing`() {
        given(PayoutCycleStatus.APPROVED)
        service().markProcessing(adminId, cycleId)
        assertEquals(PayoutCycleStatus.PROCESSING.wire, recordedToStatus())
    }

    @Test
    fun `mark-processing routes through the disbursement seam and stamps the manual provider`() {
        val cycle = cycle(PayoutCycleStatus.APPROVED).apply {
            accountSnapshotMethod = "gcash"
            accountSnapshotNumber = "09123456789"
            accountSnapshotName = "Test recipient"
        }
        Mockito.`when`(payoutCycleRepository.lockById(cycleId)).thenReturn(cycle)
        service().markProcessing(adminId, cycleId)
        val captor = ArgumentCaptor.forClass(PayoutCycle::class.java)
        Mockito.verify(payoutCycleRepository, Mockito.times(2)).save(captor.capture())
        assertEquals("manual", captor.value.provider)
        // Manual disbursement creates no external transfer, so no id is recorded.
        assertEquals(null, captor.value.providerTransferId)
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
    fun `mark-paid refuses a no-longer-covered cycle after refunds and saves nothing`() {
        given(PayoutCycleStatus.APPROVED)
        val payoutRequestService = Mockito.mock(PayoutRequestService::class.java)
        Mockito.doThrow(ApiException(HttpStatus.CONFLICT, ErrorCodes.PAYOUT_NOT_COVERED, "refund"))
            .`when`(payoutRequestService).assertCovered(anyArg())

        val ex = assertFailsWith<ApiException> {
            service(payoutRequestService = payoutRequestService)
                .markPaid(adminId, cycleId, MarkPayoutPaidRequest(paymentReference = "GC-123456"))
        }

        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.PAYOUT_NOT_COVERED, ex.code)
        Mockito.verify(payoutCycleRepository, Mockito.never()).save(anyArg<PayoutCycle>())
    }

    @Test
    fun `mark-paid on paid is rejected without saving, logging, or messaging`() {
        given(PayoutCycleStatus.PAID)

        val ex = assertFailsWith<ApiException> {
            service().markPaid(adminId, cycleId, MarkPayoutPaidRequest(paymentReference = "GC-123456"))
        }

        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        Mockito.verify(payoutCycleRepository, Mockito.never()).save(anyArg<PayoutCycle>())
        Mockito.verifyNoInteractions(payoutStatusHistoryRepository)
        Mockito.verifyNoInteractions(adminDecisionLogRepository)
        Mockito.verifyNoInteractions(photographerMessageRepository)
    }

    @Test
    fun `mark-processing on processing is rejected before the disbursement service`() {
        given(PayoutCycleStatus.PROCESSING)
        val disbursementService = Mockito.mock(DisbursementService::class.java)
        Mockito.`when`(disbursementService.activeName).thenReturn("manual")

        val ex = assertFailsWith<ApiException> { service(disbursementService).markProcessing(adminId, cycleId) }

        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        Mockito.verifyNoInteractions(disbursementService)
    }

    @Test
    fun `approve on approved is rejected without saving the cycle`() {
        given(PayoutCycleStatus.APPROVED)

        val ex = assertFailsWith<ApiException> { service().approve(adminId, cycleId) }

        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        Mockito.verify(payoutCycleRepository, Mockito.never()).save(anyArg<PayoutCycle>())
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
        val disbursementService = Mockito.mock(DisbursementService::class.java)
        val ex = assertFailsWith<ApiException> { service(disbursementService).markProcessing(adminId, cycleId) }
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        Mockito.verifyNoInteractions(disbursementService)
    }

    @Test
    fun `mark-processing refuses an uncovered cycle before initiating or saving`() {
        given(PayoutCycleStatus.APPROVED)
        val disbursementService = Mockito.mock(DisbursementService::class.java)
        val requests = Mockito.mock(PayoutRequestService::class.java)
        Mockito.doThrow(ApiException(HttpStatus.CONFLICT, ErrorCodes.PAYOUT_NOT_COVERED, "refund"))
            .`when`(requests).assertCovered(anyArg())

        val ex = assertFailsWith<ApiException> { service(disbursementService, requests).markProcessing(adminId, cycleId) }

        assertEquals(ErrorCodes.PAYOUT_NOT_COVERED, ex.code)
        Mockito.verifyNoInteractions(disbursementService, payoutStatusHistoryRepository, adminDecisionLogRepository)
        Mockito.verify(payoutCycleRepository, Mockito.never()).save(anyArg<PayoutCycle>())
    }

    // ─── Phase 4c — two-person approval ────────────────────────────────────

    @Test
    fun `large payout first approval stays pending awaiting a second admin`() {
        Mockito.`when`(userRepository.countByRole(Role.ADMIN)).thenReturn(2L)
        given(PayoutCycleStatus.PENDING, amount = "15000")
        val dto = service().approve(adminId, cycleId)
        assertEquals("pending_review", dto.status)
        assertEquals(adminId.toString(), dto.approvedBy)
    }

    @Test
    fun `large payout second approval by a different admin approves it`() {
        Mockito.`when`(userRepository.countByRole(Role.ADMIN)).thenReturn(2L)
        val firstAdmin = UUID.randomUUID()
        given(PayoutCycleStatus.PENDING, amount = "15000", approvedBy = firstAdmin)
        val dto = service().approve(adminId, cycleId)
        assertEquals("approved", dto.status)
        assertEquals(adminId.toString(), dto.secondApprovedBy)
    }

    @Test
    fun `the same admin cannot give both approvals`() {
        Mockito.`when`(userRepository.countByRole(Role.ADMIN)).thenReturn(2L)
        given(PayoutCycleStatus.PENDING, amount = "15000", approvedBy = adminId)
        val ex = assertFailsWith<ApiException> { service().approve(adminId, cycleId) }
        assertEquals(ErrorCodes.PAYOUT_SECOND_APPROVER_REQUIRED, ex.code)
    }

    @Test
    fun `two-person is dormant with a single admin`() {
        Mockito.`when`(userRepository.countByRole(Role.ADMIN)).thenReturn(1L)
        given(PayoutCycleStatus.PENDING, amount = "15000")
        val dto = service().approve(adminId, cycleId)
        assertEquals("approved", dto.status)
    }

    @Test
    fun `a large pending payout cannot be paid before it is approved`() {
        given(PayoutCycleStatus.PENDING, amount = "15000")
        val ex = assertFailsWith<ApiException> {
            service().markPaid(adminId, cycleId, MarkPayoutPaidRequest(paymentReference = "GC-1"))
        }
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
    }
}

class AdminPayoutProcessingIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var admin: AdminPayoutService
    @Autowired lateinit var cycles: PayoutCycleRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var ledger: TransactionRepository
    @Autowired lateinit var history: PayoutStatusHistoryRepository
    @Autowired lateinit var decisions: AdminDecisionLogRepository
    @Autowired lateinit var tx: TransactionTemplate
    @Autowired lateinit var dataSource: javax.sql.DataSource
    @MockitoBean lateinit var disbursements: DisbursementService
    @MockitoSpyBean lateinit var decisionService: AdminDecisionLogService

    private fun <T> anyArg(): T = Mockito.any()

    private fun user(role: Role) = users.saveAndFlush(User(
        email = "processing-${UUID.randomUUID()}@test.local", passwordHash = "unused",
        name = "Processing fixture", role = role, emailVerifiedAt = OffsetDateTime.now(),
    ))

    private fun coveredCycle(): PayoutCycle {
        val photographer = user(Role.PHOTOGRAPHER)
        val event = events.saveAndFlush(Event(
            slug = "processing-${UUID.randomUUID()}", name = "Processing", date = LocalDate.now(), location = "Cebu",
            status = EventStatus.ACTIVE,
        ))
        val photo = photos.saveAndFlush(Photo(
            eventId = event.id, photographerId = photographer.id, s3Key = "test/photo", pricePhp = BigDecimal("1000.00"),
        ))
        val order = orders.saveAndFlush(Order(
            eventId = event.id, recipientEmail = "processing@test.local", paymentMethodWire = "gcash",
            totalPhp = BigDecimal("1000.00"),
        ))
        ledger.saveAndFlush(Transaction(
            photographerId = photographer.id, eventId = event.id, photoId = photo.id, orderId = order.id,
            paidAt = OffsetDateTime.now().minusDays(30), amountKeptPhp = BigDecimal("1000.00"),
        ))
        return cycles.saveAndFlush(PayoutCycle(
            id = "PAY-PROCESSING-${UUID.randomUUID()}", photographerId = photographer.id, weekOf = LocalDate.now(),
            amountPhp = BigDecimal("1000.00"), statusWire = PayoutCycleStatus.APPROVED.wire,
            accountSnapshotMethod = "gcash", accountSnapshotNumber = "09123456789", accountSnapshotName = "Test recipient",
        ))
    }

    @Test
    fun `provider runs without a transaction or bound connection after the provider stamp commits`() {
        val cycle = coveredCycle()
        val adminId = user(Role.ADMIN).id
        Mockito.`when`(disbursements.activeName).thenReturn("paymongo")
        Mockito.`when`(disbursements.initiate(anyArg())).thenAnswer { call ->
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            assertFalse(TransactionSynchronizationManager.hasResource(dataSource))
            val committed = cycles.findById(cycle.id).orElseThrow()
            assertEquals("paymongo", committed.provider)
            assertEquals(PayoutCycleStatus.APPROVED, committed.status)
            assertEquals(null, committed.processingAt)
            assertEquals(null, committed.providerTransferId)
            assertEquals(DisbursementRequest(cycle.id, BigDecimal("1000.00"), "gcash", "09123456789", "Test recipient"),
                call.getArgument(0))
            DisbursementResult("tr_processing_success")
        }

        // Also proves the method suspends a caller's existing transaction.
        val dto = tx.execute {
            assertTrue(TransactionSynchronizationManager.isActualTransactionActive())
            admin.markProcessing(adminId, cycle.id).also {
                assertTrue(TransactionSynchronizationManager.isActualTransactionActive())
            }
        }!!

        assertEquals("processing", dto.status)
        assertEquals("paymongo", dto.provider)
        val persisted = cycles.findById(cycle.id).orElseThrow()
        assertEquals("tr_processing_success", persisted.providerTransferId)
        assertNotNull(persisted.processingAt)
        assertEquals("processing", history.findByPayoutCycleIdOrderByCreatedAtAsc(cycle.id).single().toStatus)
        assertEquals("processing", decisions.findAll().single { it.targetPayoutId == cycle.id }.decision)
    }

    @Test
    fun `reject during provider initiation keeps the reservation and lets tx2 link the transfer`() {
        val cycle = coveredCycle().apply { holdReason = "original hold" }
        cycles.saveAndFlush(cycle)
        val processingAdmin = user(Role.ADMIN).id
        val rejectingAdmin = user(Role.ADMIN).id
        Mockito.`when`(disbursements.activeName).thenReturn("paymongo")
        Mockito.`when`(disbursements.initiate(anyArg())).thenAnswer {
            assertFalse(TransactionSynchronizationManager.isActualTransactionActive())
            val refusal = runCatching { admin.reject(rejectingAdmin, cycle.id, "cancel transfer") }.exceptionOrNull()

            // A separate admin request runs between the two processing transactions.
            assertEquals(BigDecimal("1000.00"), cycles.sumReservedByPhotographer(cycle.photographerId),
                "Reject must not release a provider-claimed payout's reservation")
            assertTrue(refusal is ApiException)
            assertEquals(HttpStatus.CONFLICT, refusal.status)
            assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, refusal.code)
            val duringInitiation = cycles.findById(cycle.id).orElseThrow()
            assertEquals(PayoutCycleStatus.APPROVED, duringInitiation.status)
            assertEquals("paymongo", duringInitiation.provider)
            assertEquals("original hold", duringInitiation.holdReason)
            assertEquals(null, duringInitiation.rejectedReason)
            assertTrue(history.findByPayoutCycleIdOrderByCreatedAtAsc(cycle.id).isEmpty())
            assertTrue(decisions.findAll().none { it.targetPayoutId == cycle.id })
            DisbursementResult("tr_reject_race")
        }

        assertEquals("processing", admin.markProcessing(processingAdmin, cycle.id).status)

        val persisted = cycles.findById(cycle.id).orElseThrow()
        assertEquals(PayoutCycleStatus.PROCESSING, persisted.status)
        assertEquals("tr_reject_race", persisted.providerTransferId)
        assertEquals(BigDecimal("1000.00"), cycles.sumReservedByPhotographer(cycle.photographerId))
        assertEquals("processing", history.findByPayoutCycleIdOrderByCreatedAtAsc(cycle.id).single().toStatus)
        assertEquals("processing", decisions.findAll().single { it.targetPayoutId == cycle.id }.decision)
    }

    @Test
    fun `tx2 failure rolls back processing but keeps tx1 and logs the orphan transfer with the original failure`() {
        val cycle = coveredCycle()
        val adminId = user(Role.ADMIN).id
        val failure = IllegalStateException("decision store unavailable")
        Mockito.`when`(disbursements.activeName).thenReturn("paymongo")
        Mockito.`when`(disbursements.initiate(anyArg())).thenReturn(DisbursementResult("tr_processing_orphan"))
        Mockito.doThrow(failure).`when`(AopTestUtils.getUltimateTargetObject<AdminDecisionLogService>(decisionService))
            .logPayoutDecision(adminId, cycle.id, "processing")
        val logger = LoggerFactory.getLogger(AdminPayoutService::class.java) as Logger
        val appender = ListAppender<ILoggingEvent>().apply { start() }
        logger.addAppender(appender)
        try {
            assertSame(failure, assertFailsWith<IllegalStateException> { admin.markProcessing(adminId, cycle.id) })
            val persisted = cycles.findById(cycle.id).orElseThrow()
            assertEquals(PayoutCycleStatus.APPROVED, persisted.status)
            assertEquals("paymongo", persisted.provider)
            assertEquals(null, persisted.providerTransferId)
            assertEquals(null, persisted.processingAt)
            assertTrue(history.findByPayoutCycleIdOrderByCreatedAtAsc(cycle.id).isEmpty())
            assertTrue(decisions.findAll().none { it.targetPayoutId == cycle.id })
            val error = appender.list.single { it.level == Level.ERROR }
            assertTrue(error.formattedMessage.contains("tr_processing_orphan"))
            assertTrue(error.formattedMessage.contains(cycle.id))
            assertSame(failure, (error.throwableProxy as ThrowableProxy).throwable)
        } finally {
            logger.detachAppender(appender)
            appender.stop()
        }
    }
}
