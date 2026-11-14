package com.quickpitik.repository

import com.quickpitik.entity.PayoutAccount
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface PayoutAccountRepository : JpaRepository<PayoutAccount, UUID> {
    fun findAllByUserIdOrderByCreatedAtAsc(userId: UUID): List<PayoutAccount>

    // Batch variant for AdminPayoutService.hydrateMany — one IN query for the
    // whole admin page instead of 1–2 account lookups per row.
    fun findAllByUserIdInOrderByCreatedAtAsc(userIds: Collection<UUID>): List<PayoutAccount>
    fun findByIdAndUserId(id: UUID, userId: UUID): PayoutAccount?
    fun countByUserId(userId: UUID): Long
    fun countByUserIdAndQrS3KeyIsNotNull(userId: UUID): Long
    fun findByUserIdAndIsPrimaryTrue(userId: UUID): PayoutAccount?

    // Repeat-account matching (AA-I5). Numbers are stored digits only (PayoutAccountService.sanitizeNumber).
    @Query("SELECT DISTINCT p.userId FROM PayoutAccount p WHERE p.accountNumber IN :numbers AND p.userId <> :excludeUserId")
    fun findUserIdsByAccountNumbers(
        @Param("numbers") numbers: Collection<String>,
        @Param("excludeUserId") excludeUserId: UUID,
    ): List<UUID>

    // Held until the transaction ends, so two accounts saving one number take turns at the repeat lookup.
    @Query(value = "SELECT CAST(pg_advisory_xact_lock($PAYOUT_NUMBER_LOCK_KEY, hashtext(:number)) AS text)", nativeQuery = true)
    fun lockAccountNumber(@Param("number") number: String): String?
}

// Two-key advisory lock namespace for payout numbers (ID numbers 78302, phone keys 78303).
const val PAYOUT_NUMBER_LOCK_KEY = 78304
