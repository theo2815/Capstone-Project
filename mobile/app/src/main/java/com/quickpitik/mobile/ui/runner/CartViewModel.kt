package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.*
import com.quickpitik.mobile.data.repository.CartRepository
import com.quickpitik.mobile.data.repository.CartRepositoryImpl
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

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

class CartViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val repository: CartRepository = CartRepositoryImpl()

    val cartItems: StateFlow<List<CartItemDto>> = repository.cartItems

    val cartTotal: StateFlow<Double> = repository.cartItems
        .map { items -> items.sumOf { it.price } }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 0.0)

    private val _checkoutState = MutableStateFlow<CheckoutState>(CheckoutState.Idle)
    val checkoutState: StateFlow<CheckoutState> = _checkoutState

    private val _ordersState = MutableStateFlow<OrdersState>(OrdersState.Loading)
    val ordersState: StateFlow<OrdersState> = _ordersState

    private val _orderDetailState = MutableStateFlow<OrderDetailState>(OrderDetailState.Idle)
    val orderDetailState: StateFlow<OrderDetailState> = _orderDetailState

    init {
        // Initial sync if logged in
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            if (token != null) {
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

    fun addToCart(photoDto: PhotoDto, eventId: String, eventSlug: String, eventName: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            repository.addToCart(token, photoDto, eventId, eventSlug, eventName)
        }
    }

    fun removeFromCart(photoId: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            repository.removeFromCart(token, photoId)
        }
    }

    fun clearCart() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken()
            repository.clearCart(token)
        }
    }

    fun checkout(recipientEmail: String?, paymentMethod: String) {
        val token = sessionManager.getAccessToken()
        val email = recipientEmail ?: sessionManager.getUserEmail() ?: ""
        
        viewModelScope.launch {
            _checkoutState.value = CheckoutState.Loading
            val result = repository.checkout(token, email, paymentMethod)
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

    fun fetchOrders() {
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _ordersState.value = OrdersState.Error("Please log in to view order history.")
            return
        }

        viewModelScope.launch {
            _ordersState.value = OrdersState.Loading
            val result = repository.getOrders(token)
            result.onSuccess { orders ->
                _ordersState.value = OrdersState.Success(orders)
            }.onFailure { exception ->
                _ordersState.value = OrdersState.Error(exception.localizedMessage ?: "Failed to retrieve orders.")
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
            _orderDetailState.value = OrderDetailState.Loading
            val result = repository.getOrderDetail(token, orderId)
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

    fun getLoggedInUserEmail(): String {
        return sessionManager.getUserEmail() ?: ""
    }
}
