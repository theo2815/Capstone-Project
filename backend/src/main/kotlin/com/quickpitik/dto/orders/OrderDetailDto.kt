package com.quickpitik.dto.orders

import com.quickpitik.entity.OrderStatus
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

// GET /me/orders/{id} — extends OrderListItemDto with photos[] + downloadBundleUrl.
// Mirrors website/src/lib/api-orders.ts OrderDetail (= MockOrder & { photos, downloadBundleUrl }).
data class OrderDetailDto(
    val id: UUID,
    val eventId: UUID,
    val photoIds: List<UUID>,
    val total: BigDecimal,
    val paymentMethod: String,
    val paidAt: OffsetDateTime?,
    val eventName: String?,
    val eventSlug: String?,
    val status: OrderStatus?,
    val photos: List<OrderPhotoDetailDto>,
    val downloadBundleUrl: String? = null,
)

data class OrderPhotoDetailDto(
    val id: UUID,
    val bib: String?,
    val time: String,
    val tone: Int,
    val thumbnailUrl: String? = null,
    val previewUrl: String? = null,
    val downloadUrl: String? = null,
)
