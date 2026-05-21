package com.quickpitik.mobile.data.remote

data class AddCartItemRequest(
    val photoId: String,
    val eventId: String
)

data class CartItemDto(
    val photoId: String,
    val eventId: String,
    val thumbnailUrl: String?,
    val price: Double,
    val bib: String?,
    val eventName: String?,
    val eventSlug: String?,
    val tone: Int?,
    val time: String?
)

data class MergeCartRequest(
    val items: List<MergeCartItem>
)

data class MergeCartItem(
    val photoId: String,
    val eventId: String
)

data class ClearedResponse(
    val cleared: Boolean
)

data class RemovedResponse(
    val removed: Boolean
)

data class CreateOrderRequest(
    val items: List<CreateOrderItem>,
    val paymentMethod: String,
    val recipientEmail: String?
)

data class CreateOrderItem(
    val photoId: String,
    val eventId: String
)

data class OrderResponse(
    val id: String,
    val status: String, // "PENDING", "PAID", etc.
    val items: List<OrderResponseItem>,
    val totalAmount: Double,
    val paymentMethod: String,
    val createdAt: String,
    val redirectUrl: String? = null
)

data class OrderResponseItem(
    val photoId: String,
    val price: Double,
    val downloadUrl: String? = null
)

data class OrderListItemDto(
    val id: String,
    val eventId: String,
    val photoIds: List<String>,
    val total: Double,
    val paymentMethod: String,
    val paidAt: String?,
    val eventName: String?,
    val eventSlug: String?,
    val eventDate: String?,
    val eventState: String?,
    val status: String?
)

data class OrderDetailDto(
    val id: String,
    val eventId: String,
    val photoIds: List<String>,
    val total: Double,
    val paymentMethod: String,
    val paidAt: String?,
    val eventName: String?,
    val eventSlug: String?,
    val status: String?,
    val photos: List<OrderPhotoDetailDto>,
    val downloadBundleUrl: String? = null,
    val recipientEmail: String = "",
    val shareToken: String? = null
)

data class OrderPhotoDetailDto(
    val id: String,
    val bib: String?,
    val time: String,
    val tone: Int,
    val thumbnailUrl: String? = null,
    val previewUrl: String? = null,
    val downloadUrl: String? = null
)
