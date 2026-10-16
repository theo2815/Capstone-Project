package com.quickpitik.dto.earnings

import com.quickpitik.dto.photographer.PhotographerLibraryPhotoDto
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID

data class PhotographerPurchasedPhotoDto(
    val photo: PhotographerLibraryPhotoDto,
    val purchaseCount: Long,
    val lastPurchasedAt: OffsetDateTime?,
)

data class PurchaseEventDto(
    val id: UUID,
    val name: String,
    val date: LocalDate,
    val deletedAt: OffsetDateTime?,
)

data class PurchaseTransactionDto(
    val id: UUID,
    val isRefund: Boolean,
    val refundOf: UUID?,
    val amountKept: BigDecimal,
    val discountPhp: BigDecimal,
    val paidAt: OffsetDateTime,
    val recordedAt: OffsetDateTime,
)

data class PhotographerPurchaseDto(
    val orderId: UUID,
    val photoId: UUID,
    val buyer: String,
    val purchasedAt: OffsetDateTime?,
    val orderCreatedAt: OffsetDateTime,
    val paymentMethod: String,
    val pricePhp: BigDecimal,
    val discountPhp: BigDecimal,
    val paidPhp: BigDecimal,
    val kind: String,
    val refundStatus: String,
    val refundPending: Boolean,
    val refundedPhp: BigDecimal?,
    val refundedAt: OffsetDateTime?,
    val netKeptPhp: BigDecimal?,
    val transactions: List<PurchaseTransactionDto>,
)

data class PhotographerPhotoPurchasesResponse(
    val photo: PhotographerLibraryPhotoDto,
    val event: PurchaseEventDto,
    val items: List<PhotographerPurchaseDto>,
    val total: Long,
    val offset: Int,
    val limit: Int,
    val selectedPurchase: PhotographerPurchaseDto?,
)
