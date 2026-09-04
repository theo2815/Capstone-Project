package com.quickpitik.repository

import com.quickpitik.entity.Payment
import com.quickpitik.entity.PaymentStatus
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime
import java.util.UUID

interface PaymentRepository : JpaRepository<Payment, UUID> {
    fun findByOrderId(orderId: UUID): List<Payment>

    fun findByProviderAndProviderRef(provider: String, providerRef: String): Payment?

    // Multi-row variant for PayMongo — one Checkout Session covers N orders
    // (multi-event cart split), so one cs_id maps to N Payment rows.
    fun findAllByProviderAndProviderRef(provider: String, providerRef: String): List<Payment>

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT p FROM Payment p WHERE p.provider = :provider AND p.providerRef = :providerRef")
    fun findAllByProviderAndProviderRefForUpdate(
        @Param("provider") provider: String,
        @Param("providerRef") providerRef: String,
    ): List<Payment>

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT p FROM Payment p WHERE p.orderId IN :orderIds")
    fun findAllByOrderIdInForUpdate(@Param("orderIds") orderIds: Collection<UUID>): List<Payment>

    fun findByProviderAndStatusAndProviderRefIsNotNullAndCreatedAtBeforeOrderByCreatedAtAsc(
        provider: String,
        status: PaymentStatus,
        cutoff: OffsetDateTime,
        pageable: Pageable,
    ): List<Payment>

    fun findByProviderAndStatusAndProviderRefStartingWithOrderByCreatedAtAsc(
        provider: String,
        status: PaymentStatus,
        providerRef: String,
        pageable: Pageable,
    ): List<Payment>
}
