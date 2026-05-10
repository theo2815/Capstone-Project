package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminUserDetailDto
import com.quickpitik.dto.admin.AdminUserRowDto
import com.quickpitik.dto.admin.DecisionLogEntryDto
import com.quickpitik.dto.admin.ForceEditUserPatchRequest
import com.quickpitik.dto.admin.PhotographerSettingsSnapshotDto
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.ReservedHandleRepository
import com.quickpitik.repository.SocialLinkRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.reference.RegionsService
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class AdminUserService(
    private val userRepository: UserRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val socialLinkRepository: SocialLinkRepository,
    private val payoutAccountRepository: PayoutAccountRepository,
    private val reservedHandleRepository: ReservedHandleRepository,
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val regionsService: RegionsService,
    private val adminDecisionLogService: AdminDecisionLogService,
) {

    @Transactional(readOnly = true)
    fun list(
        roleFilter: String?,
        statusFilter: String?,
        search: String?,
        params: PaginationParams,
    ): PaginatedResponse<AdminUserRowDto> {
        val role = parseRole(roleFilter)
        val searchValue = search?.trim()?.lowercase().orEmpty()
        val pageable = OffsetLimitPageable(params)
        val page = userRepository.searchForAdmin(role, searchValue, pageable)

        // Filter on photographer-settings-derived status post-fetch — keeps
        // the JPQL on UserRepository flat. For paginated correctness we
        // would need a joined query, but the admin volume is small enough
        // (low-thousand users) that filter-after-page is fine for v1; we
        // can promote to a joined query when the dashboard scales.
        val items = page.content
        val settingsByUser = if (items.any { it.role == Role.PHOTOGRAPHER }) {
            photographerSettingsRepository.findAllByUserIdIn(
                items.filter { it.role == Role.PHOTOGRAPHER }.map { it.id },
            ).associateBy { it.userId }
        } else {
            emptyMap()
        }

        val rows = items
            .map { hydrateRow(it, settingsByUser[it.id]) }
            .let { rows -> if (statusFilter == null) rows else rows.filter { it.matchesStatus(statusFilter) } }

        return PaginatedResponse.of(rows, page.totalElements, params)
    }

    @Transactional(readOnly = true)
    fun detail(userId: UUID): AdminUserDetailDto {
        val (user, settings) = loadUserAndSettings(userId)
        val log = adminDecisionLogRepository.findForUser(userId)
            .map {
                DecisionLogEntryDto(
                    userId = it.targetUserId ?: userId,
                    decision = it.decision,
                    reason = it.reason,
                    decidedAt = it.decidedAt,
                    meta = it.meta,
                )
            }
        val row = hydrateRow(user, settings)
        return AdminUserDetailDto(
            userId = row.userId,
            role = row.role,
            email = row.email,
            name = row.name,
            brandName = row.brandName,
            handle = row.handle,
            region = row.region,
            city = row.city,
            createdAt = row.createdAt,
            verificationStatus = row.verificationStatus,
            suspendedAt = row.suspendedAt,
            suspensionReason = row.suspensionReason,
            settingsSnapshot = row.settingsSnapshot,
            decisionLog = log,
        )
    }

    fun approve(adminId: UUID, userId: UUID): AdminUserRowDto {
        val (user, settings) = loadUserAndSettings(userId)
        val s = requirePhotographerSettings(user, settings)
        s.verificationStatus = VerificationStatus.APPROVED
        photographerSettingsRepository.save(s)
        val decision = adminDecisionLogService.logUserDecision(
            adminId = adminId,
            targetUserId = userId,
            decision = "approved",
        )
        adminDecisionLogService.pushMessage(
            photographerId = userId,
            kind = PhotographerMessageKind.VERIFICATION_APPROVED,
            body = "Your photographer profile is approved. You can now upload to events.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateRow(user, s)
    }

    fun reject(adminId: UUID, userId: UUID, reason: String): AdminUserRowDto {
        val (user, settings) = loadUserAndSettings(userId)
        val s = requirePhotographerSettings(user, settings)
        // Mirrors the FE store: reject lands the row back at "incomplete"
        // with the rejection reason carried in the decision log.
        s.verificationStatus = VerificationStatus.INCOMPLETE
        photographerSettingsRepository.save(s)
        val decision = adminDecisionLogService.logUserDecision(
            adminId = adminId,
            targetUserId = userId,
            decision = "rejected",
            reason = reason,
        )
        adminDecisionLogService.pushMessage(
            photographerId = userId,
            kind = PhotographerMessageKind.VERIFICATION_REJECTED,
            body = "Your verification was not approved. Reason: $reason",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateRow(user, s)
    }

    fun resetVerification(adminId: UUID, userId: UUID): AdminUserRowDto {
        val (user, settings) = loadUserAndSettings(userId)
        val s = requirePhotographerSettings(user, settings)
        s.verificationStatus = VerificationStatus.INCOMPLETE
        photographerSettingsRepository.save(s)
        val decision = adminDecisionLogService.logUserDecision(
            adminId = adminId,
            targetUserId = userId,
            decision = "reset",
        )
        adminDecisionLogService.pushMessage(
            photographerId = userId,
            kind = PhotographerMessageKind.VERIFICATION_RESET,
            body = "Your verification status was reset by support. You can re-submit when ready.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return hydrateRow(user, s)
    }

    fun suspend(adminId: UUID, userId: UUID, reason: String): AdminUserRowDto {
        val (user, settings) = loadUserAndSettings(userId)
        if (user.role == Role.ADMIN) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "Admin accounts cannot be suspended",
                field = "userId",
            )
        }
        user.suspendedAt = OffsetDateTime.now()
        user.suspensionReason = reason
        userRepository.save(user)
        val decision = adminDecisionLogService.logUserDecision(
            adminId = adminId,
            targetUserId = userId,
            decision = "suspended",
            reason = reason,
        )
        if (user.role == Role.PHOTOGRAPHER) {
            adminDecisionLogService.pushMessage(
                photographerId = userId,
                kind = PhotographerMessageKind.SUSPENDED,
                body = "Your account has been suspended. Reason: $reason. Contact support to appeal.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
        return hydrateRow(user, settings)
    }

    fun unsuspend(adminId: UUID, userId: UUID): AdminUserRowDto {
        val (user, settings) = loadUserAndSettings(userId)
        if (user.role == Role.ADMIN) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "Admin accounts cannot be suspended",
                field = "userId",
            )
        }
        user.suspendedAt = null
        user.suspensionReason = null
        userRepository.save(user)
        val decision = adminDecisionLogService.logUserDecision(
            adminId = adminId,
            targetUserId = userId,
            decision = "unsuspended",
        )
        if (user.role == Role.PHOTOGRAPHER) {
            adminDecisionLogService.pushMessage(
                photographerId = userId,
                kind = PhotographerMessageKind.UNSUSPENDED,
                body = "Your account has been reinstated.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
        return hydrateRow(user, settings)
    }

    fun forceEdit(adminId: UUID, userId: UUID, patch: ForceEditUserPatchRequest): AdminUserRowDto {
        val (user, settings) = loadUserAndSettings(userId)
        val s = requirePhotographerSettings(user, settings)
        val changes = mutableListOf<Map<String, Any?>>()

        patch.handle?.let { handleRaw ->
            val normalized = handleRaw.trim().lowercase()
            if (normalized.isEmpty()) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "handle must not be empty",
                    field = "handle",
                )
            }
            if (!HANDLE_REGEX.matches(normalized)) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "handle must be 3-30 chars, lowercase letters/digits/hyphens, no leading/trailing hyphen",
                    field = "handle",
                )
            }
            if (reservedHandleRepository.existsByHandle(normalized)) {
                throw ApiException(
                    status = HttpStatus.CONFLICT,
                    code = ErrorCodes.RESERVED_HANDLE,
                    message = "That handle is reserved.",
                    field = "handle",
                )
            }
            val taken = photographerSettingsRepository.findByHandleIgnoreCase(normalized)
            if (taken != null && taken.userId != userId) {
                throw ApiException(
                    status = HttpStatus.CONFLICT,
                    code = ErrorCodes.HANDLE_TAKEN,
                    message = "That handle is already in use.",
                    field = "handle",
                )
            }
            if (s.handle != normalized) {
                changes += mapOf("field" to "handle", "from" to (s.handle ?: ""), "to" to normalized)
                s.handle = normalized
            }
        }

        patch.brandName?.let { brandRaw ->
            val trimmed = brandRaw.trim()
            if (trimmed.length > 100) {
                throw ValidationException(
                    code = ErrorCodes.VALIDATION_ERROR,
                    message = "brandName must be 0-100 chars",
                    field = "brandName",
                )
            }
            val nextValue = trimmed.ifBlank { null }
            if (s.brandName != nextValue) {
                changes += mapOf("field" to "brandName", "from" to (s.brandName ?: ""), "to" to (nextValue ?: ""))
                s.brandName = nextValue
            }
        }

        if (changes.isEmpty()) {
            return hydrateRow(user, s)
        }
        photographerSettingsRepository.save(s)

        // Log + inbox-message one row per changed field so the photographer
        // sees each edit individually (mirrors the FE store behaviour).
        for (change in changes) {
            val decision = adminDecisionLogService.logUserDecision(
                adminId = adminId,
                targetUserId = userId,
                decision = "force-edited",
                meta = change,
            )
            val field = change["field"]
            adminDecisionLogService.pushMessage(
                photographerId = userId,
                kind = PhotographerMessageKind.FORCE_EDIT,
                body = "Support updated your $field.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
        return hydrateRow(user, s)
    }

    // ─── Helpers ──────────────────────────────────────────────────────────
    private fun loadUserAndSettings(userId: UUID): Pair<User, PhotographerSettings?> {
        val user = userRepository.findById(userId).orElseThrow {
            NotFoundException(code = ErrorCodes.USER_NOT_FOUND, message = "User not found")
        }
        val settings = if (user.role == Role.PHOTOGRAPHER) {
            photographerSettingsRepository.findById(userId).orElse(null)
        } else {
            null
        }
        return user to settings
    }

    private fun requirePhotographerSettings(user: User, settings: PhotographerSettings?): PhotographerSettings {
        if (user.role != Role.PHOTOGRAPHER) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "Only photographers have a verification status",
                field = "userId",
            )
        }
        return settings ?: photographerSettingsRepository.save(
            PhotographerSettings(userId = user.id, verificationStatus = VerificationStatus.INCOMPLETE),
        )
    }

    internal fun hydrateRow(user: User, settings: PhotographerSettings?): AdminUserRowDto {
        val verificationStatus = when {
            user.role == Role.PHOTOGRAPHER -> (settings?.verificationStatus ?: VerificationStatus.INCOMPLETE).toWire()
            // Runners are always "approved" per the FE store contract.
            else -> "approved"
        }
        val regionLabel = settings?.let {
            val regions = regionsService.all()
            val region = regions.firstOrNull { r -> r.code == it.regionCode }
            val province = region?.provinces?.firstOrNull { p -> p.code == it.provinceCode }
            formatRegionLabel(province?.name, region?.name)
        }
        val snapshot = if (user.role == Role.PHOTOGRAPHER) {
            buildSnapshot(user.id, settings)
        } else {
            null
        }
        return AdminUserRowDto(
            userId = user.id,
            role = user.role.name,
            email = user.email,
            name = user.name,
            brandName = settings?.brandName,
            handle = settings?.handle,
            region = regionLabel,
            city = settings?.city ?: "",
            createdAt = user.createdAt.toString(),
            verificationStatus = verificationStatus,
            suspendedAt = user.suspendedAt?.toString(),
            suspensionReason = user.suspensionReason,
            settingsSnapshot = snapshot,
        )
    }

    private fun buildSnapshot(userId: UUID, settings: PhotographerSettings?): PhotographerSettingsSnapshotDto {
        val socialCount = socialLinkRepository.countByUserId(userId).toInt()
        val payoutCount = payoutAccountRepository.countByUserId(userId).toInt()
        return PhotographerSettingsSnapshotDto(
            hasCover = !settings?.coverS3Key.isNullOrBlank(),
            hasBrandName = !settings?.brandName.isNullOrBlank(),
            hasWatermark = !settings?.watermarkS3Key.isNullOrBlank(),
            hasHandle = !settings?.handle.isNullOrBlank(),
            hasRegion = !settings?.regionCode.isNullOrBlank() && !settings?.provinceCode.isNullOrBlank(),
            socialCount = socialCount,
            payoutCount = payoutCount,
        )
    }

    private fun parseRole(raw: String?): Role? {
        if (raw.isNullOrBlank()) return null
        return when (raw.trim().uppercase()) {
            "PHOTOGRAPHER" -> Role.PHOTOGRAPHER
            "RUNNER" -> Role.RUNNER
            "ADMIN" -> Role.ADMIN
            else -> null
        }
    }

    private fun AdminUserRowDto.matchesStatus(statusFilter: String): Boolean {
        return when (statusFilter.lowercase()) {
            "suspended" -> suspendedAt != null
            "approved" -> verificationStatus == "approved" && suspendedAt == null
            "pending" -> verificationStatus == "pending"
            "incomplete" -> verificationStatus == "incomplete"
            "rejected" -> verificationStatus == "rejected"
            else -> true
        }
    }

    private companion object {
        // Same shape as the photographer's own PUT.
        val HANDLE_REGEX = Regex("^[a-z0-9](?:[a-z0-9-]{1,28}[a-z0-9])?$")
    }
}
