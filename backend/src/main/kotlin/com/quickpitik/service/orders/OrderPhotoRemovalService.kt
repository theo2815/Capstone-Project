package com.quickpitik.service.orders

import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.time.Instant
import java.util.UUID
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.removal.*
import com.quickpitik.service.runner.RunnerMessagesService

@Service
class OrderPhotoRemovalService(
    private val photos: PhotoRepository,
    private val orders: OrderRepository,
    private val orderItems: OrderItemRepository,
    private val removalItems: PhotoRemovalItemRepository,
    private val history: PhotoRemovalHistoryRepository,
    private val recorder: PhotoRemovalHistoryRecorder,
    private val inbox: RunnerMessagesService,
) {
    @Transactional(propagation = Propagation.MANDATORY)
    fun onRemovalAuthorized(photoId: UUID, requestId: UUID, occurredAt: Instant) {
        photos.findAllByIdForUpdate(listOf(photoId))
        if (!photos.isRemoved(photoId)) return
        val item = removalItems.findByRequestIdOrderByIdAsc(requestId).firstOrNull {
            it.photoIdSnapshot == photoId && it.state in authorizedStates
        } ?: removalConflict()
        // Settlement already holds payment/order locks. Authorization only reads
        // orders after the photo lock, avoiding the inverse lock order.
        orders.findReferencingPhoto(photoId).filter { it.paidAt != null }.forEach { notify(it, item, occurredAt) }
    }

    @Transactional(propagation = Propagation.MANDATORY)
    fun onOrderSettled(orderId: UUID, occurredAt: Instant) {
        lockSettlementPhotos(listOf(orderId))
        val order = orders.findById(orderId).orElse(null) ?: return
        if (order.paidAt == null) return
        orderItems.findByIdOrderId(orderId).forEach { line ->
            if (photos.isRemoved(line.id.photoId)) {
                val item = removalItems.findByPhotoIdSnapshotAndStateInOrderByAdminDecidedAtAscIdAsc(line.id.photoId,
                    authorizedStates).firstOrNull() ?: removalConflict()
                notify(order, item, occurredAt)
            }
        }
    }

    @Transactional(propagation = Propagation.MANDATORY)
    internal fun lockSettlementPhotos(orderIds: Collection<UUID>) {
        val ids = orderItems.findByIdOrderIdIn(orderIds).map { it.id.photoId }.distinct().sorted()
        if (ids.isNotEmpty()) photos.findAllByIdForUpdate(ids)
    }

    private fun notify(order: Order, item: PhotoRemovalItem, occurredAt: Instant) {
        val photoId = item.photoIdSnapshot
        order.userId?.let {
            inbox.pushRemovalNotice(it, order.id, photoId, item.requestId, occurredAt)
            return
        }
        if (history.hasGuestBuyerNotice(photoId, order.id.toString())) return
        val commandId = UUID.nameUUIDFromBytes("guest_buyer_removal:${order.id}:$photoId".toByteArray(Charsets.UTF_8))
        recorder.record(RemovalHistoryEvent(requestId = item.requestId, itemId = item.id, photoIdSnapshot = photoId,
            actorType = RemovalActorType.SYSTEM, action = RemovalHistoryActions.NOTICE_STORED, occurredAt = occurredAt,
            commandId = commandId, metadata = mapOf("kind" to "guest_buyer_removal", "orderId" to order.id.toString())))
    }

    private val authorizedStates = setOf(RemovalItemState.REMOVAL_AUTHORIZED, RemovalItemState.REMOVED)
}
