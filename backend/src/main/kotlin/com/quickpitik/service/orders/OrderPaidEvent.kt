package com.quickpitik.service.orders

import java.util.UUID

// Spring application event published when an Order flips PENDING → PAID
// (or, ops/stub path, when an order is minted straight to PAID).
// Listeners run AFTER_COMMIT so a TX rollback can't trigger spurious
// emails. One event per order — multi-event carts publish N events.
data class OrderPaidEvent(
    val orderId: UUID,
    val checkoutSessionId: String?,
)
