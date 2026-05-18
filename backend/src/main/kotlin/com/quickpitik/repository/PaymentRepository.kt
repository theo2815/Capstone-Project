package com.quickpitik.repository

import com.quickpitik.entity.Payment
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface PaymentRepository : JpaRepository<Payment, UUID> {
    fun findByOrderId(orderId: UUID): List<Payment>

    fun findByProviderAndProviderRef(provider: String, providerRef: String): Payment?

    // Multi-row variant for PayMongo — one Checkout Session covers N orders
    // (multi-event cart split), so one cs_id maps to N Payment rows.
    fun findAllByProviderAndProviderRef(provider: String, providerRef: String): List<Payment>
}
