package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import java.util.UUID

/**
 * The single gate for photographer dashboard actions that require an
 * admin-APPROVED, non-suspended account — creating coupons, creating events,
 * uploading photos, and any future dashboard feature. Routing every such action
 * through [requireApproved] keeps the rule in one place instead of re-deriving it
 * per surface.
 *
 * Preserves the two 403 codes the clients already branch on:
 * [ErrorCodes.ACCOUNT_SUSPENDED] and [ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED].
 * APPROVED already implies field-complete — [AdminUserService.approve] re-runs
 * `collectMissing` before flipping the row — so this gate reads the stored status
 * rather than recomputing the required-field set.
 */
@Service
class PhotographerAccessGuard(
    private val userRepository: UserRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
) {
    /**
     * Throws 403 if the photographer is suspended or not yet admin-APPROVED;
     * otherwise returns their (APPROVED) settings row so callers that need it
     * don't re-query.
     *
     * @param action short phrase describing the attempted action ("creating
     *   coupons", "creating an event", "uploading") — tails the message so copy
     *   stays contextual.
     */
    fun requireApproved(photographerId: UUID, action: String): PhotographerSettings {
        val user = userRepository.findById(photographerId).orElse(null)
            ?: throw NotFoundException(code = ErrorCodes.USER_NOT_FOUND, message = "User not found")
        if (user.suspendedAt != null) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.ACCOUNT_SUSPENDED,
                message = "Your account is suspended. Contact support to appeal before $action.",
            )
        }
        val settings = photographerSettingsRepository.findById(photographerId).orElse(null)
        if (settings == null || settings.verificationStatus != VerificationStatus.APPROVED) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED,
                message = "Submit your photographer verification before $action.",
            )
        }
        return settings
    }
}
