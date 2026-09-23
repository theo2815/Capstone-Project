package com.quickpitik.repository

import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.VerificationStatus
import jakarta.persistence.LockModeType
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface PhotographerSettingsRepository : JpaRepository<PhotographerSettings, UUID> {
    fun findByHandleIgnoreCase(handle: String): PhotographerSettings?
    fun existsByHandleIgnoreCase(handle: String): Boolean

    fun countByVerificationStatus(status: VerificationStatus): Long

    fun findAllByUserIdIn(userIds: Collection<UUID>): List<PhotographerSettings>

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT s FROM PhotographerSettings s WHERE s.userId = :userId")
    fun findByUserIdForUpdate(@Param("userId") userId: UUID): PhotographerSettings?
}
