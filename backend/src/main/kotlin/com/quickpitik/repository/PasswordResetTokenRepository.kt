package com.quickpitik.repository

import com.quickpitik.entity.PasswordResetToken
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import java.util.UUID

@Repository
interface PasswordResetTokenRepository : JpaRepository<PasswordResetToken, UUID> {
    fun findByTokenHash(tokenHash: String): PasswordResetToken?
}
