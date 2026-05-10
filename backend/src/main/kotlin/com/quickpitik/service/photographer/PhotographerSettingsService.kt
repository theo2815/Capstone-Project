package com.quickpitik.service.photographer

import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ConflictException
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

@Service
@Transactional
class PhotographerSettingsService(
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
) {
    // Lazy-create on first /me/photographer/* read. Photographers register via
    // /auth/register with role=PHOTOGRAPHER but the photographer_settings row
    // is created only when they first touch a photographer surface — keeps
    // Phase A unchanged and avoids stale rows for users who never log in.
    fun getOrCreate(userId: UUID): PhotographerSettings {
        val existing = photographerSettingsRepository.findById(userId).orElse(null)
        if (existing != null) return existing
        val user = userRepository.findById(userId).orElseThrow {
            ConflictException(code = "USER_NOT_FOUND", message = "User not found")
        }
        if (user.role != Role.PHOTOGRAPHER) {
            // Method-level @PreAuthorize already enforces this, but defend in
            // depth — if a non-photographer somehow reaches here, refuse to
            // pollute the photographer_settings table with a row.
            throw ConflictException(code = "FORBIDDEN", message = "User is not a photographer")
        }
        val fresh = PhotographerSettings(
            userId = userId,
            verificationStatus = VerificationStatus.INCOMPLETE,
        )
        return photographerSettingsRepository.save(fresh)
    }

    @Transactional(readOnly = true)
    fun findByHandle(handle: String): PhotographerSettings? =
        photographerSettingsRepository.findByHandleIgnoreCase(handle.trim().lowercase())
}
