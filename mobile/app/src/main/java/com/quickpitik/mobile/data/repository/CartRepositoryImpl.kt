package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.UUID

class CartRepositoryImpl : CartRepository {
    private val api get() = RetrofitClient.apiService
    private val _cartItems = MutableStateFlow<List<CartItemDto>>(emptyList())
    override val cartItems: StateFlow<List<CartItemDto>> = _cartItems.asStateFlow()

    // (fingerprint of cart+method+email) -> Idempotency-Key. See checkout().
    private var idempotency: Pair<String, String>? = null

    override suspend fun fetchCart(token: String): Result<List<CartItemDto>> {
        return try {
            val response = api.getCart("Bearer $token")
            if (response.success && response.data != null) {
                _cartItems.value = response.data
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to fetch cart"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun addToCart(
        token: String?,
        photoDto: PhotoDto,
        eventId: String,
        eventSlug: String,
        eventName: String
    ): Result<CartItemDto> {
        val newItem = CartItemDto(
            photoId = photoDto.id,
            eventId = eventId,
            thumbnailUrl = photoDto.imageUrl,
            price = photoDto.price,
            bib = photoDto.bib,
            eventName = eventName,
            eventSlug = eventSlug,
            tone = photoDto.tone,
            time = photoDto.time
        )

        // Optimistically add to local state
        val currentList = _cartItems.value.toMutableList()
        if (currentList.none { it.photoId == photoDto.id }) {
            currentList.add(newItem)
            _cartItems.value = currentList
        }

        if (token != null) {
            return try {
                val response = api.addCartItem("Bearer $token", AddCartItemRequest(photoDto.id, eventId))
                if (response.success && response.data != null) {
                    // Update list with the server-returned item (price matching)
                    val list = _cartItems.value.toMutableList()
                    val idx = list.indexOfFirst { it.photoId == photoDto.id }
                    if (idx != -1) {
                        list[idx] = response.data
                        _cartItems.value = list
                    }
                    Result.success(response.data)
                } else {
                    // Revert optimistic add
                    _cartItems.value = _cartItems.value.filter { it.photoId != photoDto.id }
                    Result.failure(Exception(response.error ?: "Server refused to add item"))
                }
            } catch (e: Exception) {
                // Revert optimistic add
                _cartItems.value = _cartItems.value.filter { it.photoId != photoDto.id }
                Result.failure(e)
            }
        }

        return Result.success(newItem)
    }

    override suspend fun removeFromCart(token: String?, photoId: String): Result<Boolean> {
        // Optimistically remove from local state
        val originalList = _cartItems.value
        _cartItems.value = originalList.filter { it.photoId != photoId }

        if (token != null) {
            return try {
                val response = api.removeCartItem("Bearer $token", photoId)
                if (response.success && response.data != null) {
                    Result.success(response.data.removed)
                } else {
                    // Revert optimistic delete
                    _cartItems.value = originalList
                    Result.failure(Exception(response.error ?: "Server refused to delete item"))
                }
            } catch (e: Exception) {
                // Revert optimistic delete
                _cartItems.value = originalList
                Result.failure(e)
            }
        }

        return Result.success(true)
    }

    override suspend fun clearCart(token: String?): Result<Boolean> {
        val originalList = _cartItems.value
        _cartItems.value = emptyList()

        if (token != null) {
            return try {
                val response = api.clearCart("Bearer $token")
                if (response.success) {
                    // `cleared` is the count of rows deleted; zero is still
                    // success (the BE removes cart rows during order creation,
                    // so a follow-up DELETE legitimately finds nothing left).
                    Result.success(true)
                } else {
                    _cartItems.value = originalList
                    Result.failure(Exception(response.error ?: "Server refused to clear cart"))
                }
            } catch (e: Exception) {
                _cartItems.value = originalList
                Result.failure(e)
            }
        }

        return Result.success(true)
    }

    override suspend fun mergeCart(token: String): Result<List<CartItemDto>> {
        val currentItems = _cartItems.value
        if (currentItems.isEmpty()) {
            return fetchCart(token)
        }

        return try {
            val mergeRequest = MergeCartRequest(
                items = currentItems.map { MergeCartItem(it.photoId, it.eventId) }
            )
            val response = api.mergeCart("Bearer $token", mergeRequest)
            if (response.success && response.data != null) {
                _cartItems.value = response.data
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to merge cart"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun checkout(
        token: String?,
        recipientEmail: String?,
        paymentMethod: String
    ): Result<OrderResponse> {
        val currentItems = _cartItems.value
        if (currentItems.isEmpty()) {
            return Result.failure(Exception("Cart is empty"))
        }

        val formattedToken = token?.let { "Bearer $it" }
        // One key per (cart, method, email) attempt: the header is the
        // backend's dedupe handle, so a re-tap after a timeout must RESEND the
        // same key — the old fresh-UUID-per-call meant a retry could mint a
        // second order. Changing the method/cart/email is a genuinely new
        // order and rotates the key, mirroring checkout-modal.tsx (key
        // regenerated on method change, reused across retries).
        val fingerprint = currentItems.map { it.photoId }.sorted()
            .joinToString(",") + "|" + paymentMethod + "|" + (recipientEmail ?: "")
        val idempotencyKey = idempotency?.takeIf { it.first == fingerprint }?.second
            ?: UUID.randomUUID().toString().also { idempotency = fingerprint to it }

        val request = CreateOrderRequest(
            items = currentItems.map { CreateOrderItem(it.photoId, it.eventId) },
            paymentMethod = paymentMethod,
            recipientEmail = recipientEmail
        )

        return try {
            val response = api.createOrder(formattedToken, idempotencyKey, request)
            if (response.success && response.data != null) {
                // Order minted — the key has served its purpose; the next
                // checkout is a new order.
                idempotency = null
                // Match website: cart is cleared on PAID confirmation in
                // /orders/return, NOT on order creation. Pre-clearing here
                // strands the local cart empty if the user cancels PayMongo
                // and bounces back to retry — and the website doesn't do it.
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Checkout failed"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun getOrders(token: String): Result<List<OrderListItemDto>> {
        return try {
            // Backend MAX_LIMIT is 200 — take it all; there's no paging loop
            // here, so a smaller limit silently truncates the receipt list.
            val response = api.getOrders("Bearer $token", limit = 200)
            if (response.success && response.data != null) {
                Result.success(response.data.items)
            } else {
                Result.failure(Exception(response.error ?: "Failed to load orders"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun getOrderDetail(token: String, orderId: String): Result<OrderDetailDto> {
        return try {
            val response = api.getOrderDetail("Bearer $token", orderId)
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to load order detail"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun submitRefund(
        token: String,
        orderId: String,
        photoIds: List<String>,
        reason: String,
        note: String
    ): Result<RefundResponse> {
        return try {
            val response = api.submitRefund(
                "Bearer $token",
                orderId,
                RefundRequest(photoIds = photoIds, reason = reason, note = note)
            )
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to submit refund request"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun withdrawDispute(token: String, disputeId: String): Result<RunnerDisputeDto> {
        return try {
            val response = api.withdrawDispute("Bearer $token", disputeId)
            if (response.success && response.data != null) {
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to cancel refund request"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }
}
