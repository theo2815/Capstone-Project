package com.quickpitik.service.admin

import com.quickpitik.config.StorageProperties
import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginationParams
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.DisputeStatus
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.orders.PaymongoRefundService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.service.storage.StorageService
import com.quickpitik.exception.ApiException
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import org.springframework.data.domain.Pageable
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.util.UUID
import java.util.Optional
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

class AdminDisputeServiceTest {
    private val disputeRepository = Mockito.mock(DisputeRepository::class.java)
    private val orderRepository = Mockito.mock(OrderRepository::class.java)
    private val orderItemRepository = Mockito.mock(OrderItemRepository::class.java)
    private val photoRepository = Mockito.mock(PhotoRepository::class.java)
    private val eventRepository = Mockito.mock(EventRepository::class.java)
    private val userRepository = Mockito.mock(UserRepository::class.java)
    private val settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
    private val decisionLogRepository = Mockito.mock(AdminDecisionLogRepository::class.java)
    private val decisionLogService = Mockito.mock(AdminDecisionLogService::class.java)
    private val service = AdminDisputeService(
        disputeRepository,
        orderRepository,
        orderItemRepository,
        photoRepository,
        eventRepository,
        userRepository,
        settingsRepository,
        Mockito.mock(StorageService::class.java),
        StorageProperties(),
        decisionLogService,
        decisionLogRepository,
        Mockito.mock(RunnerMessagesService::class.java),
        Mockito.mock(PaymongoRefundService::class.java),
        Mockito.mock(TransactionTemplate::class.java),
    )

    @Test
    fun `deny refuses a dispute while its refund is pending`() {
        val adminId = UUID.randomUUID()
        val dispute = Dispute(orderId = UUID.randomUUID(), photoId = UUID.randomUUID(), reasonWire = "other")
        dispute.refundStatus = "pending"
        Mockito.`when`(disputeRepository.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        Mockito.`when`(disputeRepository.save(dispute)).thenReturn(dispute)
        Mockito.`when`(photoRepository.findById(dispute.photoId)).thenReturn(Optional.empty())
        Mockito.`when`(decisionLogService.logDisputeDecision(adminId, dispute.id, "denied", null))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetDisputeId = dispute.id, decision = "denied"))

        val error = assertFailsWith<ApiException> { service.deny(adminId, dispute.id, null) }

        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, error.code)
        assertEquals(DisputeStatus.OPEN, dispute.status)
        Mockito.verify(disputeRepository, Mockito.never()).save(dispute)
    }

    @Test
    fun `deny refuses a dispute that needs manual refund reconciliation`() {
        val adminId = UUID.randomUUID()
        val dispute = Dispute(orderId = UUID.randomUUID(), photoId = UUID.randomUUID(), reasonWire = "other")
        dispute.status = DisputeStatus.ESCALATED
        dispute.refundStatus = "manual_review"
        Mockito.`when`(disputeRepository.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        Mockito.`when`(disputeRepository.save(dispute)).thenReturn(dispute)
        Mockito.`when`(photoRepository.findById(dispute.photoId)).thenReturn(Optional.empty())
        Mockito.`when`(decisionLogService.logDisputeDecision(adminId, dispute.id, "denied", null))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetDisputeId = dispute.id, decision = "denied"))

        val error = assertFailsWith<ApiException> { service.deny(adminId, dispute.id, null) }

        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, error.code)
        assertEquals(DisputeStatus.ESCALATED, dispute.status)
        Mockito.verify(disputeRepository, Mockito.never()).save(dispute)
    }

    @Test
    fun `escalate locks the dispute before changing its state`() {
        val adminId = UUID.randomUUID()
        val dispute = Dispute(orderId = UUID.randomUUID(), photoId = UUID.randomUUID(), reasonWire = "other")
        Mockito.`when`(disputeRepository.findById(dispute.id)).thenReturn(Optional.of(dispute))
        Mockito.`when`(disputeRepository.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        Mockito.`when`(photoRepository.findById(dispute.photoId)).thenReturn(Optional.empty())
        Mockito.`when`(orderRepository.findById(dispute.orderId)).thenReturn(Optional.empty())
        Mockito.`when`(orderItemRepository.findById(OrderItemId(dispute.orderId, dispute.photoId)))
            .thenReturn(Optional.empty())
        Mockito.`when`(decisionLogService.logDisputeDecision(adminId, dispute.id, "escalated", null))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetDisputeId = dispute.id, decision = "escalated"))

        service.escalate(adminId, dispute.id, null)

        Mockito.verify(disputeRepository).findByIdForUpdate(dispute.id)
        Mockito.verify(disputeRepository, Mockito.never()).findById(dispute.id)
    }

    @Test
    fun `hydrate uses only the disputed item's historical charge`() {
        val order = order(BigDecimal("300.00"))
        val disputed = Dispute(orderId = order.id, photoId = UUID.randomUUID(), reasonWire = "other")
        val items = listOf(
            item(disputed, "100.00", "0.00"),
            OrderItem(OrderItemId(order.id, UUID.randomUUID()), BigDecimal("100.00")),
            OrderItem(OrderItemId(order.id, UUID.randomUUID()), BigDecimal("100.00")),
        )
        Mockito.`when`(orderItemRepository.findById(OrderItemId(order.id, disputed.photoId))).thenReturn(Optional.of(items.first()))
        val dto = service.hydrateOne(disputed, activity = emptyList(), order = order, photo = null)
        assertEquals(BigDecimal("300.00"), dto.orderSnapshot.total)
        assertEquals(BigDecimal("100.00"), dto.itemChargeAmount)
    }

    @Test
    fun `hydrate keeps discounted zero and missing item charges distinct`() {
        val order = order(BigDecimal("150.00"))
        val dispute = Dispute(orderId = order.id, photoId = UUID.randomUUID(), reasonWire = "other")
        Mockito.`when`(orderItemRepository.findById(OrderItemId(order.id, dispute.photoId)))
            .thenReturn(Optional.of(item(dispute, "150.00", "22.50")))
            .thenReturn(Optional.of(item(dispute, "100.00", "100.00")))
            .thenReturn(Optional.empty())

        assertEquals(BigDecimal("127.50"), service.hydrateOne(dispute, activity = emptyList(), order = order, photo = null).itemChargeAmount)
        assertEquals(BigDecimal("0.00"), service.hydrateOne(dispute, activity = emptyList(), order = order, photo = null).itemChargeAmount)
        assertNull(service.hydrateOne(dispute, activity = emptyList(), order = order, photo = null).itemChargeAmount)
    }

    @Test
    fun `hydrate exposes the refund state needed to distinguish manual review from in-flight`() {
        val dispute = Dispute(orderId = UUID.randomUUID(), photoId = UUID.randomUUID(), reasonWire = "other")
        dispute.refundStatus = "manual_review"
        assertEquals("manual_review", service.hydrateOne(dispute, activity = emptyList(), order = null, photo = null).refundStatus)
        dispute.refundStatus = "requesting"
        assertEquals("requesting", service.hydrateOne(dispute, activity = emptyList(), order = null, photo = null).refundStatus)
    }

    @Test
    fun `list batches historical items for the dispute page`() {
        val order = order(BigDecimal("300.00"))
        val first = Dispute(orderId = order.id, photoId = UUID.randomUUID(), reasonWire = "other")
        val second = Dispute(orderId = order.id, photoId = UUID.randomUUID(), reasonWire = "other")
        Mockito.`when`(disputeRepository.pageForAdmin(
            Mockito.isNull(),
            Mockito.any(Pageable::class.java) ?: OffsetLimitPageable(PaginationParams.of(0, 10)),
        ))
            .thenReturn(PageImpl(listOf(first, second)))
        Mockito.`when`(orderRepository.findAllById(setOf(order.id))).thenReturn(listOf(order))
        Mockito.`when`(orderItemRepository.findByIdOrderIdIn(setOf(order.id)))
            .thenReturn(listOf(item(first, "100.00", "0.00"), item(second, "150.00", "22.50")))
        Mockito.`when`(photoRepository.findAllById(setOf(first.photoId, second.photoId))).thenReturn(emptyList())
        Mockito.`when`(userRepository.findAllById(emptySet())).thenReturn(emptyList())
        Mockito.`when`(settingsRepository.findAllById(emptySet())).thenReturn(emptyList())
        Mockito.`when`(eventRepository.findAllById(setOf(order.eventId))).thenReturn(emptyList())
        Mockito.`when`(decisionLogRepository.findByTargetDisputeIdInOrderByDecidedAtDesc(listOf(first.id, second.id)))
            .thenReturn(emptyList())

        val page = service.list(null, PaginationParams.of(0, 10))
        assertEquals(listOf(BigDecimal("100.00"), BigDecimal("127.50")), page.items.map { it.itemChargeAmount })
        Mockito.verify(orderItemRepository).findByIdOrderIdIn(setOf(order.id))
        Mockito.verify(orderItemRepository, Mockito.never()).findById(Mockito.any(OrderItemId::class.java))
    }

    @Test
    fun `a second escalation conflicts without another decision or notice`() {
        val adminId = UUID.randomUUID()
        val dispute = Dispute(orderId = UUID.randomUUID(), photoId = UUID.randomUUID(), reasonWire = "other")
        Mockito.`when`(disputeRepository.findByIdForUpdate(dispute.id)).thenReturn(dispute)
        Mockito.`when`(photoRepository.findById(dispute.photoId)).thenReturn(Optional.empty())
        Mockito.`when`(decisionLogService.logDisputeDecision(adminId, dispute.id, "escalated", null))
            .thenReturn(AdminDecisionLog(adminId = adminId, targetDisputeId = dispute.id, decision = "escalated"))
        service.escalate(adminId, dispute.id, null)
        val error = assertFailsWith<ApiException> { service.escalate(adminId, dispute.id, null) }
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, error.code)
        assertEquals(DisputeStatus.ESCALATED, dispute.status)
        Mockito.verify(decisionLogService, Mockito.times(1)).logDisputeDecision(adminId, dispute.id, "escalated", null)
    }


    @Test
    fun `withdrawn dispute rejects escalation without writing a decision`() {
        val admin = UUID.randomUUID()
        val d = Dispute(orderId = UUID.randomUUID(), photoId = UUID.randomUUID(), reasonWire = "other",
            statusWire = DisputeStatus.WITHDRAWN.wire)
        Mockito.`when`(disputeRepository.findByIdForUpdate(d.id)).thenReturn(d)
        Mockito.`when`(photoRepository.findById(d.photoId)).thenReturn(Optional.empty())
        Mockito.`when`(decisionLogService.logDisputeDecision(admin, d.id, "escalated", null))
            .thenReturn(AdminDecisionLog(adminId = admin, targetDisputeId = d.id, decision = "escalated"))
        val error = assertFailsWith<ApiException> { service.escalate(admin, d.id, null) }
        assertEquals(409, error.status.value())
        assertEquals(DisputeStatus.WITHDRAWN, d.status)
        Mockito.verifyNoInteractions(decisionLogService)
    }

    private fun order(total: BigDecimal) = Order(
        eventId = UUID.randomUUID(),
        recipientEmail = "runner@example.test",
        paymentMethodWire = "gcash",
        totalPhp = total,
    )

    private fun item(dispute: Dispute, price: String, discount: String) = OrderItem(
        OrderItemId(dispute.orderId, dispute.photoId),
        BigDecimal(price),
        BigDecimal(discount),
    )
}
