package com.quickpitik.repository

import com.quickpitik.entity.Order
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

interface OrderRepository : JpaRepository<Order, UUID> {
    fun findByUserIdOrderByPaidAtDescCreatedAtDesc(userId: UUID, pageable: Pageable): Page<Order>

    fun findByIdempotencyKey(idempotencyKey: String): List<Order>

    // Guest order lookup. Used by `GET /orders/{id}/status?token=…` so the
    // /orders/return page can poll an unauthenticated session. UNIQUE column
    // guarantees at most one match.
    fun findByShareToken(shareToken: String): Order?

    // Atomically claim the right to send this order's receipt. Returns 1 if this
    // caller won the claim, 0 if someone else already holds it.
    //
    // The read-check-write it replaces (load order → check emailSentAt == null →
    // send → stamp) let two concurrent PayMongo webhook retries both observe null
    // and both send, so the buyer got the receipt twice. The WHERE clause makes
    // the check and the write one statement, so exactly one caller can win.
    @Modifying
    @Query("UPDATE Order o SET o.emailSentAt = :now WHERE o.id = :orderId AND o.emailSentAt IS NULL")
    fun claimReceiptSend(@Param("orderId") orderId: UUID, @Param("now") now: OffsetDateTime): Int

    // Give the claim back when the send itself failed, so a later webhook
    // re-delivery or a manual reprocess can retry. Without this the atomic claim
    // would convert every transient Resend failure into a permanently lost
    // receipt — the opposite of the old behaviour, which deliberately left
    // email_sent_at null on failure for exactly this reason.
    @Modifying
    @Query("UPDATE Order o SET o.emailSentAt = NULL WHERE o.id = :orderId")
    fun releaseReceiptSend(@Param("orderId") orderId: UUID): Int
}
