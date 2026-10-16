package com.quickpitik.repository

import com.quickpitik.entity.OrderItem
import com.quickpitik.entity.OrderItemId
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
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
        SELECT p.eventId AS eventId, COUNT(oi) AS purchaseCount
        FROM OrderItem oi, Photo p, Order o
        WHERE oi.id.photoId = p.id
          AND oi.id.orderId = o.id
          AND p.photographerId = :photographerId
          AND p.eventId IN :eventIds
          AND o.eventId = p.eventId
          AND (o.paidAt IS NOT NULL OR o.status IN (
              com.quickpitik.entity.OrderStatus.PAID,
              com.quickpitik.entity.OrderStatus.FULFILLED,
              com.quickpitik.entity.OrderStatus.REFUNDED
          ))
        GROUP BY p.eventId
        """,
    )
    fun countPurchasesByEventIds(
        @Param("photographerId") photographerId: UUID,
        @Param("eventIds") eventIds: Collection<UUID>,
    ): List<EventPurchaseCountProjection>

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

    @Query(
        value = """
        SELECT p.id AS photoId, COUNT(oi) AS purchaseCount, MAX(o.paidAt) AS lastPurchasedAt
        FROM OrderItem oi, Photo p, Order o
        WHERE oi.id.photoId = p.id AND oi.id.orderId = o.id
          AND p.photographerId = :photographerId AND p.eventId = :eventId AND o.eventId = :eventId
          AND (o.paidAt IS NOT NULL OR o.status IN (
              com.quickpitik.entity.OrderStatus.PAID,
              com.quickpitik.entity.OrderStatus.FULFILLED,
              com.quickpitik.entity.OrderStatus.REFUNDED
          ))
        GROUP BY p.id
        ORDER BY MAX(COALESCE(o.paidAt, o.createdAt)) DESC, p.id ASC
        """,
        countQuery = """
        SELECT COUNT(DISTINCT p.id) FROM OrderItem oi, Photo p, Order o
        WHERE oi.id.photoId = p.id AND oi.id.orderId = o.id
          AND p.photographerId = :photographerId AND p.eventId = :eventId AND o.eventId = :eventId
          AND (o.paidAt IS NOT NULL OR o.status IN (
              com.quickpitik.entity.OrderStatus.PAID,
              com.quickpitik.entity.OrderStatus.FULFILLED,
              com.quickpitik.entity.OrderStatus.REFUNDED
          ))
        """,
    )
    fun pagePurchasedPhotosForPhotographer(
        @Param("photographerId") photographerId: UUID,
        @Param("eventId") eventId: UUID,
        pageable: Pageable,
    ): Page<PurchasedPhotoProjection>

    @Query(
        value = """
        SELECT oi FROM OrderItem oi, Order o, Photo p
        WHERE oi.id.orderId = o.id AND oi.id.photoId = p.id
          AND p.photographerId = :photographerId AND p.id = :photoId AND o.eventId = p.eventId
          AND (o.paidAt IS NOT NULL OR o.status IN (
              com.quickpitik.entity.OrderStatus.PAID,
              com.quickpitik.entity.OrderStatus.FULFILLED,
              com.quickpitik.entity.OrderStatus.REFUNDED
          ))
        ORDER BY COALESCE(o.paidAt, o.createdAt) DESC, o.id ASC
        """,
        countQuery = """
        SELECT COUNT(oi) FROM OrderItem oi, Order o, Photo p
        WHERE oi.id.orderId = o.id AND oi.id.photoId = p.id
          AND p.photographerId = :photographerId AND p.id = :photoId AND o.eventId = p.eventId
          AND (o.paidAt IS NOT NULL OR o.status IN (
              com.quickpitik.entity.OrderStatus.PAID,
              com.quickpitik.entity.OrderStatus.FULFILLED,
              com.quickpitik.entity.OrderStatus.REFUNDED
          ))
        """,
    )
    fun pagePurchasesForPhotographerPhoto(
        @Param("photographerId") photographerId: UUID,
        @Param("photoId") photoId: UUID,
        pageable: Pageable,
    ): Page<OrderItem>

    @Query(
        """
        SELECT oi FROM OrderItem oi, Order o, Photo p
        WHERE oi.id.orderId = o.id AND oi.id.photoId = p.id
          AND p.photographerId = :photographerId AND p.id = :photoId
          AND o.id = :orderId AND o.eventId = p.eventId
          AND (o.paidAt IS NOT NULL OR o.status IN (
              com.quickpitik.entity.OrderStatus.PAID,
              com.quickpitik.entity.OrderStatus.FULFILLED,
              com.quickpitik.entity.OrderStatus.REFUNDED
          ))
        """,
    )
    fun findPurchaseForPhotographerPhoto(
        @Param("photographerId") photographerId: UUID,
        @Param("photoId") photoId: UUID,
        @Param("orderId") orderId: UUID,
    ): OrderItem?
}

interface PhotoSalesProjection {
    val photoId: UUID
    val salesCount: Long
}

interface EventPurchaseCountProjection {
    val eventId: UUID
    val purchaseCount: Long
}

interface PurchasedPhotoProjection {
    val photoId: UUID
    val purchaseCount: Long
    val lastPurchasedAt: OffsetDateTime?
}
