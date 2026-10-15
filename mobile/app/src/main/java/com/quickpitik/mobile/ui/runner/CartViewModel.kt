package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.isPhotographerRole
import com.quickpitik.mobile.data.remote.CartItemDto
import com.quickpitik.mobile.data.remote.CouponPreviewDto
import com.quickpitik.mobile.data.remote.OrderDetailDto
import com.quickpitik.mobile.data.remote.OrderListItemDto
import com.quickpitik.mobile.data.remote.OrderResponse
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.repository.CartRepository
import com.quickpitik.mobile.data.repository.CartRepositoryImpl
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import retrofit2.HttpException

sealed class CheckoutState {
    object Idle : CheckoutState()
    object Loading : CheckoutState()
    data class Success(val order: OrderResponse) : CheckoutState()
    data class Error(val message: String) : CheckoutState()
}

sealed class OrdersState {
    object Loading : OrdersState()
    data class Success(val orders: List<OrderListItemDto>) : OrdersState()
    data class Error(val message: String) : OrdersState()
}

sealed class OrderDetailState {
    object Idle : OrderDetailState()
    object Loading : OrderDetailState()
    data class Success(val order: OrderDetailDto) : OrderDetailState()
    data class Error(val message: String) : OrderDetailState()
}

sealed class RefundActionState {
    object Idle : RefundActionState()
    object Submitting : RefundActionState()
    data class Success(val message: String) : RefundActionState()
    data class Error(val message: String) : RefundActionState()
}

// Drives /orders/return — the screen PayMongo bounces back to after checkout.
// Mirrors website OrdersReturnPage state machine (polling → paid/timeout/failed).
sealed class OrderReturnState {
    object Idle : OrderReturnState()
    data class Polling(val attempt: Int) : OrderReturnState()
    data class Paid(val order: OrderDetailDto) : OrderReturnState()
    object Timeout : OrderReturnState()
    data class Failed(val message: String) : OrderReturnState()
}

class CartViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val repository: CartRepository = CartRepositoryImpl()

    // The receipt route relaunches the poll on every entry; without this a
    // stale loop's Timeout could overwrite the fresh loop's Paid.
    private var pollJob: Job? = null

    val cartItems: StateFlow<List<CartItemDto>> = repository.cartItems

    val cartTotal: StateFlow<Double> = repository.cartItems
        .map { items -> items.sumOf { it.price } }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 0.0)

    // Coupon preview for the checkout sheet (V47/V50). Null until fetched; the net
    // the runner is actually charged is cartTotal − discountTotal. code == null →
    // auto-applied photographer coupons; a typed code overrides. couponPreview.code
    // is the typed override that produced the current preview — what checkout()
    // sends on CreateOrderRequest (auto coupons need no code).
    private val _couponPreview = MutableStateFlow<CouponPreviewDto?>(null)
    val couponPreview: StateFlow<CouponPreviewDto?> = _couponPreview

    private val _couponError = MutableStateFlow<String?>(null)
    val couponError: StateFlow<String?> = _couponError

    private val _couponLoading = MutableStateFlow(false)
    val couponLoading: StateFlow<Boolean> = _couponLoading

    private val _checkoutState = MutableStateFlow<CheckoutState>(CheckoutState.Idle)
    val checkoutState: StateFlow<CheckoutState> = _checkoutState

    // Floating-cart overlay state — mirrors the website's useUiStore cart/checkout
    // flags. Cart + checkout are sheets, not nav routes, so they're driven from
    // the VM and read by FloatingCart at the root of MainActivity.
    private val _cartSheetOpen = MutableStateFlow(false)
    val cartSheetOpen: StateFlow<Boolean> = _cartSheetOpen

    private val _checkoutSheetOpen = MutableStateFlow(false)
    val checkoutSheetOpen: StateFlow<Boolean> = _checkoutSheetOpen

    // Set by "Buy now" CTAs so the next add-to-cart skips the cart sheet and
    // jumps straight to checkout — mirrors website's expressCheckoutPending.
    private val _expressCheckoutPending = MutableStateFlow(false)
    val expressCheckoutPending: StateFlow<Boolean> = _expressCheckoutPending

    private val _ordersState = MutableStateFlow<OrdersState>(OrdersState.Loading)
    val ordersState: StateFlow<OrdersState> = _ordersState

    // Photo IDs the runner already owns — a PAID/FULFILLED order covers them.
    // Derived from the same /me/orders data OrdersScreen loads, so the cart and
    // checkout sheets can flag a duplicate before checkout instead of on the
    // 409. Kept fresh by a fetchOrders() call when those sheets open. Narrower
    // than the server gate (no PENDING/guest), so the backend 409 stays the net.
    val ownedPhotoIds: StateFlow<Set<String>> = _ordersState
        .map { settledPhotoIds(it) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptySet())

    private val _orderDetailState = MutableStateFlow<OrderDetailState>(OrderDetailState.Idle)
    val orderDetailState: StateFlow<OrderDetailState> = _orderDetailState

    private val _refundActionState = MutableStateFlow<RefundActionState>(RefundActionState.Idle)
    val refundActionState: StateFlow<RefundActionState> = _refundActionState

    private val _orderReturnState = MutableStateFlow<OrderReturnState>(OrderReturnState.Idle)
    val orderReturnState: StateFlow<OrderReturnState> = _orderReturnState

    // Flips true the moment the CheckoutSheet hands the PayMongo URL off to
    // the OS browser. Used to detect "user came back without finishing
    // checkout" (Activity ON_RESUME while still true ⇒ no deep link arrived
    // ⇒ they bailed) and route them to /orders so the pending order is
    // visible instead of stranding them on the redirecting overlay.
    private val _paymongoLaunched = MutableStateFlow(false)
    val paymongoLaunched: StateFlow<Boolean> = _paymongoLaunched

    fun markPaymongoLaunched() { _paymongoLaunched.value = true }
    fun clearPaymongoLaunched() { _paymongoLaunched.value = false }

    fun setCheckoutError(message: String) {
        _checkoutState.value = CheckoutState.Error(message)
    }

    init {
        // Orders belong to the signed-in account, but this ViewModel lives as
        // long as MainActivity and so outlives every sign-out. Any change of
        // account drops them (security audit 2026-10-14, MOB-3), as
        // PhotographerPurchasesViewModel does. Not in clearCart(): that is also
        // the runner's own "Clear cart" button. UNDISPATCHED so drop(1) skips
        // the value current now, never the first real change.
        viewModelScope.launch(start = CoroutineStart.UNDISPATCHED) {
            sessionManager.authChanges.drop(1).collect { dropAccountState() }
        }
        // Initial sync if logged in. Skipped for a PHOTOGRAPHER token — the
        // cart endpoints are RUNNER-role-gated, so the merge was one doomed
        // 403 per app start (and the cart UI is hidden for them anyway).
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            if (token != null && !isPhotographerRole(sessionManager.getUserRole())) {
                repository.mergeCart(token)
            }
        }
    }

    fun fetchCart() {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            repository.fetchCart(token)
        }
    }

    // One-shot user-facing outcome of a cart mutation. The repository reverts
    // an optimistic add/remove when the server refuses — without this the
    // revert was completely silent and the tile just snapped back.
    private val _cartMessage = MutableStateFlow<String?>(null)
    val cartMessage: StateFlow<String?> = _cartMessage
    fun clearCartMessage() { _cartMessage.value = null }

    fun addToCart(photoDto: PhotoDto, eventId: String, eventSlug: String, eventName: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            repository.addToCart(token, photoDto, eventId, eventSlug, eventName)
                .onFailure {
                    _cartMessage.value = it.localizedMessage
                        ?: "Couldn't add that photo to your cart. Try again."
                }
        }
    }

    fun removeFromCart(photoId: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            repository.removeFromCart(token, photoId)
                .onSuccess {
                    // The checkout sheet's discount was priced for the old item
                    // set; re-price it so the shown total never drifts from the
                    // charged one.
                    _couponPreview.value?.let { previewCoupon(it.code) }
                }
                .onFailure {
                    _cartMessage.value = it.localizedMessage
                        ?: "Couldn't remove that photo. Try again."
                }
        }
    }

    fun clearCart() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            repository.clearCart(token)
        }
    }

    // Fetch the coupon preview for the current cart. code null/blank → auto
    // photographer coupons; a typed code validates + overrides. On an invalid
    // typed code the backend 400s — surface the message and KEEP the previous
    // preview so the shown total never jumps back up to list price.
    fun previewCoupon(code: String?) {
        val photoIds = cartItems.value.map { it.photoId }
        if (photoIds.isEmpty()) {
            _couponPreview.value = null
            return
        }
        val token = sessionManager.getAccessToken()
        viewModelScope.launch {
            _couponLoading.value = true
            _couponError.value = null
            repository.previewCoupon(token, photoIds, code?.takeIf { it.isNotBlank() })
                .onSuccess { _couponPreview.value = it }
                .onFailure {
                    _couponError.value = it.localizedMessage ?: "That code can't be applied."
                }
            _couponLoading.value = false
        }
    }

    fun resetCoupon() {
        _couponPreview.value = null
        _couponError.value = null
    }

    fun checkout(recipientEmail: String?, paymentMethod: String) {
        val token = sessionManager.getAccessToken()
        val email = recipientEmail ?: sessionManager.getUserEmail() ?: ""
        // Send only a typed override; auto photographer coupons are applied
        // server-side without a code. couponPreview.code is the typed code that
        // produced the current preview (null for auto-only).
        val couponCode = _couponPreview.value?.code

        viewModelScope.launch {
            _checkoutState.value = CheckoutState.Loading
            val result = repository.checkout(token, email, paymentMethod, couponCode)
            result.onSuccess { order ->
                _checkoutState.value = CheckoutState.Success(order)
            }.onFailure { exception ->
                _checkoutState.value = CheckoutState.Error(exception.localizedMessage ?: "Payment checkout failed.")
            }
        }
    }

    fun resetCheckoutState() {
        _checkoutState.value = CheckoutState.Idle
    }

    fun openCartSheet() { _cartSheetOpen.value = true }
    fun closeCartSheet() { _cartSheetOpen.value = false }
    fun openCheckoutSheet() {
        _cartSheetOpen.value = false
        _checkoutSheetOpen.value = true
    }
    fun closeCheckoutSheet() {
        _checkoutSheetOpen.value = false
        resetCoupon()
    }
    fun triggerExpressCheckout() { _expressCheckoutPending.value = true }
    fun clearExpressCheckout() { _expressCheckoutPending.value = false }

    // Returns the Job so pull-to-refresh can join() it and settle its spinner
    // when the fetch actually finishes.
    fun fetchOrders(): Job = viewModelScope.launch {
        val generation = sessionManager.authChanges.value
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _ordersState.value = OrdersState.Error("Please log in to view order history.")
            return@launch
        }
        run {
            // Only show the spinner on first load — a refetch (closing an
            // order detail, pull-to-refresh) keeps the list on screen instead
            // of flashing back to a full-screen loading state.
            if (_ordersState.value !is OrdersState.Success) {
                _ordersState.value = OrdersState.Loading
            }
            val result = repository.getOrders(token)
            // The account changed while this was in flight: the answer is the
            // previous account's.
            if (generation != sessionManager.authChanges.value) return@launch
            result.onSuccess { orders ->
                _ordersState.value = OrdersState.Success(orders)
            }.onFailure { exception ->
                // A failed background refresh keeps the orders already on
                // screen; only a first load surfaces the error state.
                if (_ordersState.value !is OrdersState.Success) {
                    _ordersState.value = OrdersState.Error(
                        exception.localizedMessage ?: "Failed to retrieve orders."
                    )
                }
            }
        }
    }

    fun fetchOrderDetail(orderId: String) {
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _orderDetailState.value = OrderDetailState.Error("Please log in to view order details.")
            return
        }

        viewModelScope.launch {
            val generation = sessionManager.authChanges.value
            _orderDetailState.value = OrderDetailState.Loading
            val result = repository.getOrderDetail(token, orderId)
            // As in fetchOrders: an answer that lands after the account changed
            // is the previous account's.
            if (generation != sessionManager.authChanges.value) return@launch
            result.onSuccess { order ->
                _orderDetailState.value = OrderDetailState.Success(order)
            }.onFailure { exception ->
                _orderDetailState.value = OrderDetailState.Error(exception.localizedMessage ?: "Failed to retrieve order details.")
            }
        }
    }

    fun resetOrderDetailState() {
        _orderDetailState.value = OrderDetailState.Idle
    }

    // Files one Dispute per selected photo (reason + note shared). On success
    // we re-fetch the order so the embedded `disputes` payload — chips, photo
    // locks, timeline — comes straight from the server, never a local mirror.
    fun submitRefund(orderId: String, photoIds: List<String>, reason: String, note: String) {
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _refundActionState.value = RefundActionState.Error("Please log in to request a refund.")
            return
        }
        if (photoIds.isEmpty()) {
            _refundActionState.value = RefundActionState.Error("Pick at least one photo to refund.")
            return
        }
        viewModelScope.launch {
            _refundActionState.value = RefundActionState.Submitting
            repository.submitRefund(token, orderId, photoIds, reason, note)
                .onSuccess {
                    val count = photoIds.size
                    _refundActionState.value = RefundActionState.Success(
                        "Refund request sent for $count photo${if (count == 1) "" else "s"}. We'll review within 3 business days."
                    )
                    refreshOrderDetailSilently(orderId, token)
                }
                .onFailure { exception ->
                    _refundActionState.value = RefundActionState.Error(
                        exception.localizedMessage ?: "We couldn't send your request. Try again in a moment."
                    )
                }
        }
    }

    fun withdrawDispute(orderId: String, disputeId: String) {
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _refundActionState.value = RefundActionState.Error("Please log in to cancel a refund request.")
            return
        }
        viewModelScope.launch {
            _refundActionState.value = RefundActionState.Submitting
            repository.withdrawDispute(token, disputeId)
                .onSuccess {
                    _refundActionState.value = RefundActionState.Success("Refund request cancelled.")
                    refreshOrderDetailSilently(orderId, token)
                }
                .onFailure { exception ->
                    _refundActionState.value = RefundActionState.Error(
                        exception.localizedMessage ?: "We couldn't cancel that request. Try again in a moment."
                    )
                }
        }
    }

    // Re-pull the order detail without flipping the screen back to a spinner —
    // the refund banner stays visible while the timeline/chips refresh in place.
    private suspend fun refreshOrderDetailSilently(orderId: String, token: String) {
        val generation = sessionManager.authChanges.value
        val result = repository.getOrderDetail(token, orderId)
        if (generation != sessionManager.authChanges.value) return
        result.onSuccess { order ->
            _orderDetailState.value = OrderDetailState.Success(order)
        }
    }

    fun resetRefundActionState() {
        _refundActionState.value = RefundActionState.Idle
    }

    fun getLoggedInUserEmail(): String {
        return sessionManager.getUserEmail() ?: ""
    }

    // Poll /me/orders/{id} every 2s for up to 60s, mirroring the website's
    // OrdersReturnPage. Stops on PAID / FULFILLED (success) or REFUNDED
    // (terminal failure). On success the photos the order bought leave the cart on this phone.
    fun pollOrderReturn(orderId: String, shareToken: String? = null) {
        val token = sessionManager.getAccessToken()
        if (token == null && shareToken.isNullOrBlank()) {
            _orderReturnState.value = OrderReturnState.Failed("Please log in to view this order.")
            return
        }
        val isGuest = !shareToken.isNullOrBlank()
        pollJob?.cancel()
        pollJob = viewModelScope.launch {
            var attempts = 0
            while (attempts < POLL_MAX_ATTEMPTS) {
                attempts++
                _orderReturnState.value = OrderReturnState.Polling(attempts)
                // V39: poll status-only (lighter than full detail). On the last
                // attempt, ask PayMongo to actively settle/expire a stuck pending
                // intent (?verify=true) rather than just timing out.
                val verify = attempts == POLL_MAX_ATTEMPTS
                val statusResult = if (isGuest) {
                    repository.getGuestOrderStatus(orderId, shareToken!!, verify)
                } else {
                    repository.getOrderStatus(token!!, orderId, verify)
                }

                val terminal = statusResult.fold(
                    onSuccess = { status ->
                        val reported = status.status.uppercase()
                        when {
                            isSettledOrderStatus(status.status) -> {
                                hydrateReturnDetail(orderId, token, shareToken)
                                true
                            }
                            reported == "EXPIRED" -> {
                                _orderReturnState.value = OrderReturnState.Failed(
                                    "This payment session expired. Start checkout again if you still want the photos."
                                )
                                true
                            }
                            reported == "REFUNDED" -> {
                                _orderReturnState.value = OrderReturnState.Failed(
                                    "This order was refunded. Check your email or open it from Orders."
                                )
                                true
                            }
                            else -> false // PENDING — keep polling
                        }
                    },
                    onFailure = { e ->
                        // A fresh order can 404 briefly on read lag — tolerate the
                        // first couple of polls; past that a 404 is a definite
                        // not-found, so fail fast instead of spinning 60s (F7).
                        if (attempts >= 3 && (e as? HttpException)?.code() == 404) {
                            _orderReturnState.value = OrderReturnState.Failed(
                                "We couldn't find that order. Open it from Orders."
                            )
                            true
                        } else {
                            false // transient (visibility lag / BE hiccup) — keep polling
                        }
                    },
                )
                if (terminal) return@launch
                delay(POLL_INTERVAL_MS)
            }
            _orderReturnState.value = OrderReturnState.Timeout
        }
    }

    // Fetch the full order once after a terminal PAID/FULFILLED — a status poll
    // carries no photos/download URLs, and the receipt needs them.
    private suspend fun hydrateReturnDetail(orderId: String, token: String?, shareToken: String?) {
        val detail = when {
            !shareToken.isNullOrBlank() -> repository.getGuestOrderDetail(orderId, shareToken)
            token != null -> repository.getOrderDetail(token, orderId)
            else -> null
        }
        detail
            ?.onSuccess {
                // A receipt link can be anyone's order (security audit 2026-10-14,
                // MOB-2): drop only what it bought, on this phone. The backend
                // already took those photos out of a signed-in buyer's server
                // cart when the order was created; the website does the same.
                repository.dropLocal(it.photoIds)
                _orderReturnState.value = OrderReturnState.Paid(it)
            }
            ?.onFailure {
                _orderReturnState.value = OrderReturnState.Failed(
                    it.localizedMessage ?: "Payment received — open the receipt from Orders."
                )
            }
            ?: run {
                _orderReturnState.value = OrderReturnState.Failed(
                    "Payment received — open the receipt from Orders."
                )
            }
    }

    fun resetOrderReturnState() {
        _orderReturnState.value = OrderReturnState.Idle
    }

    private fun dropAccountState() {
        pollJob?.cancel()
        _ordersState.value = OrdersState.Loading
        _orderDetailState.value = OrderDetailState.Idle
        _orderReturnState.value = OrderReturnState.Idle
    }

    private companion object {
        const val POLL_INTERVAL_MS = 2_000L
        const val POLL_MAX_ATTEMPTS = 30 // 60s
    }
}

private val SETTLED_ORDER_STATUSES = setOf("PAID", "FULFILLED")

/**
 * The one place "this order is settled" is decided — the runner owns the photos,
 * the cart may clear and the receipt may render. Read by the cart's owned-photo
 * flags, the checkout sheet's no-redirect route to the receipt, and the receipt
 * poll's terminal branch. The backend writes the status UPPERCASE; compared
 * case-insensitively anyway. PENDING / EXPIRED / REFUNDED and an absent status
 * are all not settled.
 */
internal fun isSettledOrderStatus(status: String?): Boolean =
    status != null && status.uppercase() in SETTLED_ORDER_STATUSES

/**
 * Photo ids the runner already owns: every photo on a settled order. Anything
 * but a loaded list (loading, error) means "we don't know yet" — no duplicate
 * flags, and the backend's 409 stays the net.
 */
internal fun settledPhotoIds(state: OrdersState): Set<String> =
    if (state is OrdersState.Success) {
        state.orders.filter { isSettledOrderStatus(it.status) }.flatMap { it.photoIds }.toSet()
    } else {
        emptySet()
    }
