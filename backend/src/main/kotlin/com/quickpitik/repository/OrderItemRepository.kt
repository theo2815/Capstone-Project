package com.quickpitik.repository

import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface OrderItemRepository : JpaRepository<OrderItem, OrderItemId> {
    fun findByIdOrderId(orderId: UUID): List<OrderItem>

    fun findByIdOrderIdIn(orderIds: Collection<UUID>): List<OrderItem>

    fun existsByIdPhotoId(photoId: UUID): Boolean

    @Modifying
    @Query("DELETE FROM OrderItem oi WHERE oi.id.photoId = :photoId AND oi.id.orderId IN :orderIds")
    fun deleteUnpaidReferences(@Param("photoId") photoId: UUID, @Param("orderIds") orderIds: Collection<UUID>): Int

    // Sales count per photo from paid order_items. Refund flow leaves the row
    // intact so this stays a sales-receipt count (PR 9 negative-amount tx rows
    // will reconcile revenue separately per Q-E3).
    @Query(
        """
        SELECT oi.id.photoId AS photoId, COUNT(oi) AS salesCount
        FROM OrderItem oi
        WHERE oi.id.photoId IN :photoIds
          AND oi.id.orderId IN (
              SELECT o.id FROM Order o WHERE o.paidAt IS NOT NULL OR o.status IN (
                  com.quickpitik.entity.OrderStatus.PAID,
                  com.quickpitik.entity.OrderStatus.FULFILLED,
                  com.quickpitik.entity.OrderStatus.REFUNDED
              )
          )
        GROUP BY oi.id.photoId
        """,
    )
    fun countSalesByPhotoIds(@Param("photoIds") photoIds: Collection<UUID>): List<PhotoSalesProjection>

    @Query(
        """
        SELECT CASE WHEN COUNT(oi) > 0 THEN true ELSE false END
        FROM OrderItem oi, Photo p, Order o
        WHERE oi.id.photoId = p.id
          AND oi.id.orderId = o.id
          AND p.photographerId = :photographerId
          AND p.eventId = :eventId
          AND o.eventId = :eventId
          AND (o.paidAt IS NOT NULL OR o.status IN (
              com.quickpitik.entity.OrderStatus.PAID,
              com.quickpitik.entity.OrderStatus.FULFILLED,
              com.quickpitik.entity.OrderStatus.REFUNDED
          ))
        """,
    )
    fun existsSettledPurchaseForEvent(
        @Param("photographerId") photographerId: UUID,
        @Param("eventId") eventId: UUID,
    ): Boolean
}

interface PhotoSalesProjection {
    val photoId: UUID
    val salesCount: Long
}
