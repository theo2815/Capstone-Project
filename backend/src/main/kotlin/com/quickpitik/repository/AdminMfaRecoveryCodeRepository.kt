package com.quickpitik.repository

import com.quickpitik.entity.AdminMfaRecoveryCode
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface AdminMfaRecoveryCodeRepository : JpaRepository<AdminMfaRecoveryCode, UUID> {
    fun findByUserIdAndCodeHashAndUsedAtIsNull(userId: UUID, codeHash: String): AdminMfaRecoveryCode?

    fun deleteByUserId(userId: UUID)
}
