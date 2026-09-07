package com.quickpitik.service.disbursement

import com.quickpitik.dto.orders.PaymongoTransferNode
import com.quickpitik.dto.orders.PaymongoTransferWebhookAttributes
import com.quickpitik.dto.orders.PaymongoTransferWebhookEvent
import com.quickpitik.dto.orders.PaymongoTransferWebhookOuter
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PayoutStatusHistoryRepository
import com.quickpitik.repository.ProcessedWebhookEventRepository
import com.quickpitik.service.earnings.PayoutStatusRecorder
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals

// Provider is the source of truth for whether money moved. Event id dedupe +
// state-machine mapping are the logic worth proving.
class PayoutDisbursementWebhookServiceTest {

    private lateinit var payoutCycleRepository: PayoutCycleRepository
    private lateinit var processedWebhookEventRepository: ProcessedWebhookEventRepository
    private lateinit var payoutStatusHistoryRepository: PayoutStatusHistoryRepository

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        payoutCycleRepository = Mockito.mock(PayoutCycleRepository::class.java)
        processedWebhookEventRepository = Mockito.mock(ProcessedWebhookEventRepository::class.java)
        payoutStatusHistoryRepository = Mockito.mock(PayoutStatusHistoryRepository::class.java)
        Mockito.`when`(payoutCycleRepository.save(anyArg<PayoutCycle>())).thenAnswer { it.getArgument(0) }
    }

    private fun service() = PayoutDisbursementWebhookService(
        payoutCycleRepository,
        processedWebhookEventRepository,
        PayoutStatusRecorder(payoutStatusHistoryRepository),
    )

    private fun cycle(status: PayoutCycleStatus, transferId: String = "tr_1"): PayoutCycle {
        val c = PayoutCycle(
            id = "PAY-X-001",
            photographerId = UUID.randomUUID(),
            weekOf = LocalDate.of(2026, 9, 1),
            methodWire = "gcash",
        )
        c.status = status
        c.amountPhp = BigDecimal("500")
        c.provider = "paymongo"
        c.providerTransferId = transferId
        return c
    }

    private fun event(type: String, transferId: String = "tr_1", eventId: String = "evt_1") =
        PaymongoTransferWebhookEvent(
            PaymongoTransferWebhookOuter(
                id = eventId,
                attributes = PaymongoTransferWebhookAttributes(
                    type = type,
                    data = PaymongoTransferNode(id = transferId, attributes = null),
                ),
            ),
        )

    @Test
    fun `successful transfer moves processing to paid`() {
        Mockito.`when`(payoutCycleRepository.findByProviderTransferId("tr_1"))
            .thenReturn(cycle(PayoutCycleStatus.PROCESSING))
        assertEquals("paid", service().handle(event("transfer.outward.successful"))["status"])
    }

    @Test
    fun `failed transfer moves processing to failed`() {
        Mockito.`when`(payoutCycleRepository.findByProviderTransferId("tr_1"))
            .thenReturn(cycle(PayoutCycleStatus.PROCESSING))
        assertEquals("failed", service().handle(event("transfer.outward.failed"))["status"])
    }

    @Test
    fun `returned transfer moves paid to returned`() {
        Mockito.`when`(payoutCycleRepository.findByProviderTransferId("tr_1"))
            .thenReturn(cycle(PayoutCycleStatus.PAID))
        assertEquals("returned", service().handle(event("transfer.outward.returned"))["status"])
    }

    @Test
    fun `a redelivered event is deduped before any work`() {
        Mockito.`when`(
            processedWebhookEventRepository.existsByProviderAndEventId("paymongo", "evt_1"),
        ).thenReturn(true)
        val res = service().handle(event("transfer.outward.successful"))
        assertEquals(true, res["deduped"])
        Mockito.verify(payoutCycleRepository, Mockito.never()).findByProviderTransferId(anyArg())
    }

    @Test
    fun `an unknown transfer id is ignored`() {
        Mockito.`when`(payoutCycleRepository.findByProviderTransferId(anyArg())).thenReturn(null)
        assertEquals(
            "unknown-transfer",
            service().handle(event("transfer.outward.successful", transferId = "tr_x"))["ignored"],
        )
    }

    @Test
    fun `an out-of-order outcome is skipped, not re-applied`() {
        // Already PAID; a late "successful" (which expects PROCESSING) is a no-op.
        Mockito.`when`(payoutCycleRepository.findByProviderTransferId("tr_1"))
            .thenReturn(cycle(PayoutCycleStatus.PAID))
        assertEquals("invalid-from", service().handle(event("transfer.outward.successful"))["skipped"])
    }
}
