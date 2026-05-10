package com.quickpitik.repository

import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface OrderItemRepository : JpaRepository<OrderItem, OrderItemId> {
    fun findByIdOrderId(orderId: UUID): List<OrderItem>

    fun findByIdOrderIdIn(orderIds: Collection<UUID>): List<OrderItem>

    // Sales count per photo from paid order_items. Refund flow leaves the row
    // intact so this stays a sales-receipt count (PR 9 negative-amount tx rows
    // will reconcile revenue separately per Q-E3).
    @Query(
        """
        SELECT oi.id.photoId AS photoId, COUNT(oi) AS salesCount
        FROM OrderItem oi
        WHERE oi.id.photoId IN :photoIds
        GROUP BY oi.id.photoId
        """,
    )
    fun countSalesByPhotoIds(@Param("photoIds") photoIds: Collection<UUID>): List<PhotoSalesProjection>
}

interface PhotoSalesProjection {
    val photoId: UUID
    val salesCount: Long
}
