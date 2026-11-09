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
    fun existsByUserIdAndStatusNot(userId: UUID, status: SellerDetailsStatus): Boolean

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
    @Query("SELECT s.userId FROM SellerDetails s WHERE s.idNumberHmac = :hmac AND s.userId <> :excludeUserId")
    fun findUserIdsByIdNumberHmac(@Param("hmac") hmac: String, @Param("excludeUserId") excludeUserId: UUID): List<UUID>

    @Query("SELECT s.userId FROM SellerDetails s WHERE s.phoneKey = :phoneKey AND s.userId <> :excludeUserId")
    fun findUserIdsByPhoneKey(@Param("phoneKey") phoneKey: String, @Param("excludeUserId") excludeUserId: UUID): List<UUID>
}
