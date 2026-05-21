package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.*
import kotlinx.coroutines.flow.StateFlow

interface CartRepository {
    val cartItems: StateFlow<List<CartItemDto>>
    
    suspend fun fetchCart(token: String): Result<List<CartItemDto>>
    suspend fun addToCart(token: String?, photoDto: PhotoDto, eventId: String, eventSlug: String, eventName: String): Result<CartItemDto>
    suspend fun removeFromCart(token: String?, photoId: String): Result<Boolean>
    suspend fun clearCart(token: String?): Result<Boolean>
    suspend fun mergeCart(token: String): Result<List<CartItemDto>>
    
    suspend fun checkout(
        token: String?,
        recipientEmail: String?,
        paymentMethod: String
    ): Result<OrderResponse>
    
    suspend fun getOrders(token: String): Result<List<OrderListItemDto>>
    suspend fun getOrderDetail(token: String, orderId: String): Result<OrderDetailDto>
}
