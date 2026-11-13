package com.quickpitik.repository

import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface SellerDetailsRepository : JpaRepository<SellerDetails, UUID> {
    fun existsByUserIdAndStatusIn(userId: UUID, statuses: Collection<SellerDetailsStatus>): Boolean

    // Admin queue: records waiting for review first, then oldest submission first.
    @Query(
        value = """
        SELECT s FROM SellerDetails s
        WHERE (:status IS NULL OR s.status = :status)
        ORDER BY CASE WHEN s.status = com.quickpitik.entity.SellerDetailsStatus.MANUAL_REVIEW THEN 0 ELSE 1 END,
                 s.submittedAt ASC, s.userId ASC
        """,
        countQuery = "SELECT COUNT(s) FROM SellerDetails s WHERE (:status IS NULL OR s.status = :status)",
    )
    fun findForAdmin(@Param("status") status: SellerDetailsStatus?, pageable: Pageable): Page<SellerDetails>

    // Repeat-account matching (AA-I5): other accounts sharing a detail; V94 indexes both columns.
    // [hmacs]: the number's hash under every ID type, so a different type on the other account still matches (R-1).
    @Query("SELECT s.userId FROM SellerDetails s WHERE s.idNumberHmac IN :hmacs AND s.userId <> :excludeUserId")
    fun findUserIdsByIdNumberHmacIn(@Param("hmacs") hmacs: Collection<String>, @Param("excludeUserId") excludeUserId: UUID): List<UUID>

    @Query("SELECT s.userId FROM SellerDetails s WHERE s.phoneKey = :phoneKey AND s.userId <> :excludeUserId")
    fun findUserIdsByPhoneKey(@Param("phoneKey") phoneKey: String, @Param("excludeUserId") excludeUserId: UUID): List<UUID>

    // R-9: held until the transaction ends, so two accounts submitting one ID number take turns at the repeat lookup.
    @Query(value = "SELECT CAST(pg_advisory_xact_lock($ID_NUMBER_LOCK_KEY, hashtext(:hmac)) AS text)", nativeQuery = true)
    fun lockIdNumber(@Param("hmac") hmac: String): String?
}

// Two-key advisory lock namespace for ID numbers (the race-creation lock is the one-key 78301).
const val ID_NUMBER_LOCK_KEY = 78302
