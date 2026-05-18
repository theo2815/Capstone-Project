package com.quickpitik.repository

import com.quickpitik.entity.Order
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface OrderRepository : JpaRepository<Order, UUID> {
    fun findByUserIdOrderByPaidAtDescCreatedAtDesc(userId: UUID, pageable: Pageable): Page<Order>

    fun findByIdempotencyKey(idempotencyKey: String): List<Order>

    // Guest order lookup. Used by `GET /orders/{id}/status?token=…` so the
    // /orders/return page can poll an unauthenticated session. UNIQUE column
    // guarantees at most one match.
    fun findByShareToken(shareToken: String): Order?
}
