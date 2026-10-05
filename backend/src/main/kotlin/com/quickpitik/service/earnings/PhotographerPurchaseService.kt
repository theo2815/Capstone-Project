package com.quickpitik.service.earnings

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.earnings.PhotographerPhotoPurchasesResponse
import com.quickpitik.dto.earnings.PhotographerPurchaseDto
import com.quickpitik.dto.earnings.PhotographerPurchasedPhotoDto
import com.quickpitik.dto.earnings.PurchaseEventDto
import com.quickpitik.dto.earnings.PurchaseTransactionDto
import com.quickpitik.dto.earnings.privacyBuyerDisplay
import com.quickpitik.dto.earnings.privacyGuestBuyerDisplay
import com.quickpitik.entity.Dispute
import com.quickpitik.entity.Order
import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.Transaction
import com.quickpitik.entity.refundNeedsReconciliation
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderItemRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotographerEventService
import com.quickpitik.service.photographer.PhotographerPhotoDtoMapper
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.util.UUID

@Service
@Transactional(readOnly = true)
class PhotographerPurchaseService(
    private val orderItems: OrderItemRepository,
    private val orders: OrderRepository,
    private val photos: PhotoRepository,
    private val events: EventRepository,
    private val transactions: TransactionRepository,
    private val disputes: DisputeRepository,
    private val users: UserRepository,
    private val eventService: PhotographerEventService,
    private val photoDtoMapper: PhotographerPhotoDtoMapper,
) {
    fun listPurchasedPhotos(
        photographerId: UUID,
        eventId: UUID,
        params: PaginationParams,
    ): PaginatedResponse<PhotographerPurchasedPhotoDto> {
        eventService.getEventDetail(photographerId, eventId, includeRemoved = true)
        val page = orderItems.pagePurchasedPhotosForPhotographer(photographerId, eventId, OffsetLimitPageable(params))
        if (page.isEmpty) return PaginatedResponse.of(emptyList(), page.totalElements, params)
        val photosById = photos.findAllById(page.content.map { it.photoId }).associateBy { it.id }
        val items = page.content.mapNotNull { row ->
            val photo = photosById[row.photoId] ?: return@mapNotNull null
            PhotographerPurchasedPhotoDto(
                photo = photoDtoMapper.toLibraryDto(photo, row.purchaseCount),
                purchaseCount = row.purchaseCount,
                lastPurchasedAt = row.lastPurchasedAt,
            )
        }
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    fun listPhotoPurchases(
        photographerId: UUID,
        photoId: UUID,
        orderId: UUID?,
        params: PaginationParams,
    ): PhotographerPhotoPurchasesResponse {
        val photo = photos.findFirstByIdAndPhotographerId(photoId, photographerId)
            ?: throw NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Photo not found")
        val event = events.findById(photo.eventId).orElse(null)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val page = orderItems.pagePurchasesForPhotographerPhoto(photographerId, photoId, OffsetLimitPageable(params))
        val selected = orderId?.let {
            orderItems.findPurchaseForPhotographerPhoto(photographerId, photoId, it)
                ?: throw NotFoundException(code = ErrorCodes.PHOTO_NOT_FOUND, message = "Purchase not found")
        } ?: page.content.firstOrNull()
        val rows = (page.content + listOfNotNull(selected)).distinctBy { it.id.orderId }
        val orderIds = rows.map { it.id.orderId }
        val ordersById = orders.findAllById(orderIds).associateBy { it.id }
        val transactionsByOrder = if (orderIds.isEmpty()) emptyMap() else
            transactions.findByPhotographerIdAndPhotoIdAndOrderIdIn(photographerId, photoId, orderIds)
                .groupBy { it.orderId }
        val disputesByOrder = if (orderIds.isEmpty()) emptyMap() else
            disputes.findByOrderIdIn(orderIds).filter { it.photoId == photoId }.groupBy { it.orderId }
        val buyerIds = ordersById.values.mapNotNull { it.userId }.toSet()
        val buyersById = users.findAllById(buyerIds).associate { it.id to privacyBuyerDisplay(it.name) }
        val purchasesByOrder = rows.associate { row ->
            val order = ordersById.getValue(row.id.orderId)
            order.id to purchaseDto(
                row, order, transactionsByOrder[order.id].orEmpty(), disputesByOrder[order.id].orEmpty(),
                order.userId?.let { buyersById[it] },
            )
        }
        return PhotographerPhotoPurchasesResponse(
            photo = photoDtoMapper.toLibraryDto(photo, page.totalElements),
            event = PurchaseEventDto(event.id, event.name, event.date, event.deletedAt),
            items = page.content.map { purchasesByOrder.getValue(it.id.orderId) },
            total = page.totalElements,
            offset = params.offset,
            limit = params.limit,
            selectedPurchase = selected?.let { purchasesByOrder[it.id.orderId] },
        )
    }

    private fun purchaseDto(
        item: OrderItem,
        order: Order,
        ledger: List<Transaction>,
        refundFacts: List<Dispute>,
        liveBuyer: String?,
    ): PhotographerPurchaseDto {
        val price = item.pricePhpAtPurchase.setScale(2)
        val discount = item.discountPhp.setScale(2)
        val paid = price.subtract(discount)
        val completed = refundFacts.filter {
            it.refundStatus == "succeeded" ||
                (it.statusWire == "resolved" && it.refundedAt != null && it.resolutionWire in REFUND_RESOLUTIONS)
        }
        val refundRows = ledger.filter { it.isRefund }
        val refundPending = refundFacts.any {
            it.statusWire !in setOf("denied", "withdrawn") &&
                it.refundStatus != "failed" && it.refundNeedsReconciliation()
        }
        val knownRefund = completed.takeIf { it.isNotEmpty() && it.all { fact -> fact.refundAmountPhp != null } }
            ?.mapNotNull { it.refundAmountPhp }
            ?.fold(ZERO, BigDecimal::add)
        val refundStatus = when {
            completed.isNotEmpty() &&
                (completed.any { it.resolutionWire == "refund_full" } || knownRefund != null && knownRefund >= paid) -> "full"
            completed.isNotEmpty() -> "partial"
            refundRows.isNotEmpty() -> "recorded"
            refundPending -> "pending"
            else -> "none"
        }
        val sale = ledger.firstOrNull { !it.isRefund }
        val buyer = sale?.buyerDisplayName?.takeIf { it.isNotBlank() }
            ?: liveBuyer?.takeIf { it.isNotBlank() }
            ?: privacyGuestBuyerDisplay(order.recipientEmail).takeIf { it.isNotBlank() }
            ?: "Buyer unavailable"
        return PhotographerPurchaseDto(
            orderId = order.id,
            photoId = item.id.photoId,
            buyer = buyer,
            purchasedAt = order.paidAt,
            orderCreatedAt = order.createdAt,
            paymentMethod = order.paymentMethodWire.takeIf { it in PAYMENT_METHODS } ?: "other",
            pricePhp = price,
            discountPhp = discount,
            paidPhp = paid,
            kind = if (paid.signum() == 0) "free" else "paid",
            refundStatus = refundStatus,
            refundPending = refundPending,
            refundedPhp = if (completed.isNotEmpty() || refundRows.isNotEmpty()) knownRefund else ZERO,
            refundedAt = completed.mapNotNull { it.refundedAt }.maxOrNull(),
            netKeptPhp = when {
                sale != null -> ledger.fold(ZERO) { sum, tx -> sum.add(tx.amountKeptPhp) }
                paid.signum() == 0 -> ZERO
                else -> null
            },
            transactions = ledger.sortedWith(compareBy<Transaction> { it.createdAt }.thenBy { it.id })
                .map {
                    PurchaseTransactionDto(
                        it.id, it.isRefund, it.refundOf, it.amountKeptPhp, it.discountPhp, it.paidAt, it.createdAt,
                    )
                },
        )
    }

    private companion object {
        val ZERO: BigDecimal = BigDecimal("0.00")
        val REFUND_RESOLUTIONS = setOf("refund_full", "refund_partial")
        val PAYMENT_METHODS = setOf("gcash", "maya", "card", "qrph")
    }
}
