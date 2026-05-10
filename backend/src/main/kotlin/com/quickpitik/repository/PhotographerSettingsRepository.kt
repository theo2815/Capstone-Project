package com.quickpitik.repository

import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.VerificationStatus
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface PhotographerSettingsRepository : JpaRepository<PhotographerSettings, UUID> {
    fun findByHandleIgnoreCase(handle: String): PhotographerSettings?
    fun existsByHandleIgnoreCase(handle: String): Boolean

    fun countByVerificationStatus(status: VerificationStatus): Long

    fun findAllByUserIdIn(userIds: Collection<UUID>): List<PhotographerSettings>
}
