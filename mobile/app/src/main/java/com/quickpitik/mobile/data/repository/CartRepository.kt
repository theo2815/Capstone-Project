package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.CartItemDto
import com.quickpitik.mobile.data.remote.CouponPreviewDto
import com.quickpitik.mobile.data.remote.OrderDetailDto
import com.quickpitik.mobile.data.remote.OrderStatusDto
import com.quickpitik.mobile.data.remote.OrderListItemDto
import com.quickpitik.mobile.data.remote.OrderResponse
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RefundResponse
import com.quickpitik.mobile.data.remote.RunnerDisputeDto
import kotlinx.coroutines.flow.StateFlow

interface CartRepository {
    val cartItems: StateFlow<List<CartItemDto>>
    
    suspend fun fetchCart(token: String): Result<List<CartItemDto>>
    suspend fun addToCart(token: String?, photoDto: PhotoDto, eventId: String, eventSlug: String, eventName: String): Result<CartItemDto>
    suspend fun removeFromCart(token: String?, photoId: String): Result<Boolean>
    suspend fun clearCart(token: String?): Result<Boolean>
    /** Drops [photoIds] from the cart on this phone only; the server cart is not called. */
    fun dropLocal(photoIds: Collection<String>)
    suspend fun mergeCart(token: String): Result<List<CartItemDto>>
    
    // code == null/blank → the auto-applied photographer coupons for these
    // photos; a typed code validates + overrides. Server prices it; mobile just
    // displays the result. Public (guest checkout allowed).
    suspend fun previewCoupon(
        token: String?,
        photoIds: List<String>,
        code: String?
    ): Result<CouponPreviewDto>

    suspend fun checkout(
        token: String?,
        recipientEmail: String?,
        paymentMethod: String,
        couponCode: String?
    ): Result<OrderResponse>
    
    suspend fun getOrders(token: String): Result<List<OrderListItemDto>>
    suspend fun getOrderDetail(token: String, orderId: String): Result<OrderDetailDto>
    suspend fun getGuestOrderDetail(orderId: String, shareToken: String): Result<OrderDetailDto>

    // V39 status-only poll (lighter than full detail). verify=true asks PayMongo
    // to settle/expire a stuck pending intent — used on the last poll attempt.
    suspend fun getOrderStatus(token: String, orderId: String, verify: Boolean): Result<OrderStatusDto>
    suspend fun getGuestOrderStatus(orderId: String, shareToken: String, verify: Boolean): Result<OrderStatusDto>

    // Cancel a live QR Ph payment; the answer is the order's status afterwards
    // (PAID/FULFILLED if the payment won the race).
    suspend fun cancelOrder(token: String, orderId: String): Result<OrderStatusDto>
    suspend fun cancelGuestOrder(orderId: String, shareToken: String): Result<OrderStatusDto>

    suspend fun submitRefund(
        token: String,
        orderId: String,
        photoIds: List<String>,
        reason: String,
        note: String
    ): Result<RefundResponse>
    suspend fun withdrawDispute(token: String, disputeId: String): Result<RunnerDisputeDto>
}
