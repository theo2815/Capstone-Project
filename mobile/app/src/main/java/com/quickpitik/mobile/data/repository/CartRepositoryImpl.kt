package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.AddCartItemRequest
import com.quickpitik.mobile.data.remote.CartItemDto
import com.quickpitik.mobile.data.remote.CouponPreviewDto
import com.quickpitik.mobile.data.remote.CouponPreviewRequest
import com.quickpitik.mobile.data.remote.CreateOrderItem
import com.quickpitik.mobile.data.remote.CreateOrderRequest
import com.quickpitik.mobile.data.remote.MergeCartItem
import com.quickpitik.mobile.data.remote.MergeCartRequest
import com.quickpitik.mobile.data.remote.OrderDetailDto
import com.quickpitik.mobile.data.remote.OrderListItemDto
import com.quickpitik.mobile.data.remote.OrderStatusDto
import com.quickpitik.mobile.data.remote.OrderResponse
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RefundRequest
import com.quickpitik.mobile.data.remote.RefundResponse
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.RunnerDisputeDto
import com.quickpitik.mobile.data.remote.apiCall
import com.quickpitik.mobile.data.remote.toResult
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import retrofit2.HttpException
import java.util.UUID

class CartRepositoryImpl : CartRepository {
    private val api get() = RetrofitClient.apiService
    private val _cartItems = MutableStateFlow<List<CartItemDto>>(emptyList())
    override val cartItems: StateFlow<List<CartItemDto>> = _cartItems.asStateFlow()

    // (fingerprint of cart+method+email) -> Idempotency-Key. See checkout().
    private var idempotency: Pair<String, String>? = null

    override suspend fun fetchCart(token: String): Result<List<CartItemDto>> {
        return try {
            api.getCart("Bearer $token").toResult("Failed to fetch cart")
                .onSuccess { _cartItems.value = it }
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
                api.addCartItem("Bearer $token", AddCartItemRequest(photoDto.id, eventId))
                    .toResult("Server refused to add item")
                    .onSuccess { served ->
                        // Update list with the server-returned item (price matching)
                        val list = _cartItems.value.toMutableList()
                        val idx = list.indexOfFirst { it.photoId == photoDto.id }
                        if (idx != -1) {
                            list[idx] = served
                            _cartItems.value = list
                        }
                    }
                    // Revert optimistic add
                    .onFailure { _cartItems.value = _cartItems.value.filter { it.photoId != photoDto.id } }
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
                api.removeCartItem("Bearer $token", photoId)
                    .toResult("Server refused to delete item")
                    .map { it.removed }
                    // Revert optimistic delete
                    .onFailure { _cartItems.value = originalList }
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

    override fun dropLocal(photoIds: Collection<String>) {
        _cartItems.value = _cartItems.value.filterNot { it.photoId in photoIds }
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
            api.mergeCart("Bearer $token", mergeRequest).toResult("Failed to merge cart")
                .onSuccess { _cartItems.value = it }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun previewCoupon(
        token: String?,
        photoIds: List<String>,
        code: String?
    ): Result<CouponPreviewDto> {
        if (photoIds.isEmpty()) return Result.failure(Exception("Cart is empty"))
        // apiCall surfaces the backend envelope (COUPON_INVALID / EXPIRED /
        // USAGE_LIMIT_REACHED / NOT_APPLICABLE) instead of a raw HTTP code.
        return apiCall("Couldn't apply that coupon") {
            api.previewCoupon(
                token?.let { "Bearer $it" },
                CouponPreviewRequest(photoIds = photoIds, code = code),
            )
        }
    }

    override suspend fun checkout(
        token: String?,
        recipientEmail: String?,
        paymentMethod: String,
        couponCode: String?
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
            .joinToString(",") + "|" + paymentMethod + "|" + (recipientEmail ?: "") +
            "|" + (couponCode ?: "")
        val idempotencyKey = idempotency?.takeIf { it.first == fingerprint }?.second
            ?: UUID.randomUUID().toString().also { idempotency = fingerprint to it }

        val request = CreateOrderRequest(
            items = currentItems.map { CreateOrderItem(it.photoId, it.eventId) },
            paymentMethod = paymentMethod,
            recipientEmail = recipientEmail,
            couponCode = couponCode,
        )

        // apiCall surfaces the backend envelope message — order-create is
        // rate-limited (10/min/IP), so a 429 here must read as the backend's
        // "slow down" copy, not "HTTP 429" (sibling pattern: submitRefund /
        // withdrawDispute below).
        //
        // Nothing clears the cart here: the website clears it on PAID
        // confirmation in /orders/return, and pre-clearing would strand the
        // local cart empty if the user cancels PayMongo and bounces back.
        return apiCall("Checkout failed") {
            try {
                api.createOrder(formattedToken, idempotencyKey, request)
            } catch (e: HttpException) {
                // A refused key (its checkout expired, or another checkout holds
                // these photos) can never succeed again: retire it, as the
                // website does on CONFLICT / CHECKOUT_IN_PROGRESS, so the next
                // try is a new checkout. Any other failure (PAYMENT_FAILED is a
                // 409 too) keeps it so a retry resumes the order it may have made.
                if (e.code() == 409 && refusesKey(e)) idempotency = null
                throw e
            }
        }
            // Order minted — the key has served its purpose; the next
            // checkout is a new order.
            .onSuccess { idempotency = null }
    }

    override suspend fun getOrders(token: String): Result<List<OrderListItemDto>> {
        return try {
            // Backend MAX_LIMIT is 200 — take it all; there's no paging loop
            // here, so a smaller limit silently truncates the receipt list.
            api.getOrders("Bearer $token", limit = 200).toResult("Failed to load orders")
                .map { it.items }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun getOrderDetail(token: String, orderId: String): Result<OrderDetailDto> {
        return try {
            api.getOrderDetail("Bearer $token", orderId).toResult("Failed to load order detail")
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun getGuestOrderDetail(orderId: String, shareToken: String): Result<OrderDetailDto> {
        return try {
            api.getGuestOrderDetail(orderId, shareToken).toResult("Failed to load order detail")
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun getOrderStatus(
        token: String,
        orderId: String,
        verify: Boolean,
    ): Result<OrderStatusDto> {
        return try {
            api.getOrderStatus("Bearer $token", orderId, if (verify) true else null)
                .toResult("Failed to load order status")
        } catch (e: Exception) {
            // Raw exception preserved so pollOrderReturn can detect a 404 (fail fast).
            Result.failure(e)
        }
    }

    override suspend fun getGuestOrderStatus(
        orderId: String,
        shareToken: String,
        verify: Boolean,
    ): Result<OrderStatusDto> {
        return try {
            api.getGuestOrderStatus(orderId, shareToken, if (verify) true else null)
                .toResult("Failed to load order status")
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    override suspend fun cancelOrder(token: String, orderId: String): Result<OrderStatusDto> =
        apiCall("Couldn't cancel the payment") { api.cancelOrder("Bearer $token", orderId) }

    override suspend fun cancelGuestOrder(orderId: String, shareToken: String): Result<OrderStatusDto> =
        apiCall("Couldn't cancel the payment") { api.cancelGuestOrder(orderId, shareToken) }


    override suspend fun submitRefund(
        token: String,
        orderId: String,
        photoIds: List<String>,
        reason: String,
        note: String
    ): Result<RefundResponse> {
        return apiCall("Failed to submit refund request") {
            api.submitRefund(
                "Bearer $token",
                orderId,
                RefundRequest(photoIds = photoIds, reason = reason, note = note)
            )
        }
    }

    override suspend fun withdrawDispute(token: String, disputeId: String): Result<RunnerDisputeDto> =
        apiCall("Failed to cancel refund request") { api.withdrawDispute("Bearer $token", disputeId) }
}

private val KEY_REFUSED = Regex(""""code"\s*:\s*"(CONFLICT|CHECKOUT_IN_PROGRESS)"""")

// Reads the error envelope's codes without consuming the body, so
// RetrofitClient.parseError can still show the backend's message.
private fun refusesKey(e: HttpException): Boolean {
    val source = e.response()?.errorBody()?.source() ?: return false
    source.request(Long.MAX_VALUE)
    return KEY_REFUSED.containsMatchIn(source.buffer.clone().readUtf8())
}
