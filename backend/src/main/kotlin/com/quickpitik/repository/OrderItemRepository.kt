package com.quickpitik.repository

import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface OrderItemRepository : JpaRepository<OrderItem, OrderItemId> {
    fun findByIdOrderId(orderId: UUID): List<OrderItem>

    fun findByIdOrderIdIn(orderIds: Collection<UUID>): List<OrderItem>
}
