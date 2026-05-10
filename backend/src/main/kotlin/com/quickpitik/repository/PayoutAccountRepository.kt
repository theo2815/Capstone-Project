package com.quickpitik.repository

import com.quickpitik.entity.PayoutAccount
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface PayoutAccountRepository : JpaRepository<PayoutAccount, UUID> {
    fun findAllByUserIdOrderByCreatedAtAsc(userId: UUID): List<PayoutAccount>
    fun findByIdAndUserId(id: UUID, userId: UUID): PayoutAccount?
    fun countByUserId(userId: UUID): Long
    fun findByUserIdAndIsPrimaryTrue(userId: UUID): PayoutAccount?
}
