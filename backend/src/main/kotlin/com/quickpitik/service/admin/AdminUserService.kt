package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.AdminPhotographerCoverDto
import com.quickpitik.dto.admin.AdminPhotographerRegionDto
import com.quickpitik.dto.admin.AdminPhotographerSettingsDto
import com.quickpitik.dto.admin.AdminPhotographerWatermarkDto
import com.quickpitik.dto.admin.AdminUserDetailDto
import com.quickpitik.dto.admin.AdminUserRowDto
import com.quickpitik.dto.admin.DecisionLogEntryDto
import com.quickpitik.dto.admin.PhotographerSettingsSnapshotDto
import com.quickpitik.dto.photographer.PayoutAccountDto
import com.quickpitik.dto.photographer.PayoutQrDto
import com.quickpitik.dto.photographer.PhotographerMessageDto
import com.quickpitik.dto.photographer.SocialLinkDto
import com.quickpitik.dto.photographer.toDto
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.RunnerMessage
import com.quickpitik.entity.RunnerMessageKind
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.SocialLinkRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.RefreshTokenService
import com.quickpitik.service.photographer.PhotographerSettingsService
import com.quickpitik.service.profile.UserDtoMapper
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.runner.RunnerMessagesService
import org.springframework.data.domain.PageRequest
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
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val regionsService: RegionsService,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val storageService: com.quickpitik.service.storage.StorageService,
    private val storageProperties: StorageProperties,
    private val userDtoMapper: UserDtoMapper,
    private val runnerMessagesService: RunnerMessagesService,
    private val photographerSettingsService: PhotographerSettingsService,
    private val refreshTokenService: RefreshTokenService,
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

        // L-5 — hoist regions once before the per-row loop. RegionsService.all()
        // is AtomicReference.get() so each call is O(1), but the per-row dispatch
        // is still wasted work proportional to page size.
        val regions = regionsService.all()
        val rows = items
            .map { hydrateRow(it, settingsByUser[it.id], regions) }
            .let { rows -> if (statusFilter == null) rows else rows.filter { it.matchesStatus(statusFilter) } }

        return PaginatedResponse.of(rows, page.totalElements, params)
    }

    @Transactional(readOnly = true)
    fun detail(userId: UUID): AdminUserDetailDto {
        val (user, settings) = loadUserAndSettings(userId)
        // H-6: cap to most-recent 50 — for a high-friction photographer
        // the unbounded log used to bloat the response into the
        // multi-kilobyte range. Deeper history can move to its own
        // GET /admin/users/{id}/decisions?offset&limit if the FE asks.
        val log = adminDecisionLogRepository.findForUserCapped(userId, PageRequest.of(0, DECISION_LOG_DETAIL_CAP))
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
            avatarUrl = row.avatarUrl,
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

    // F-NEW-1 — full photographer-settings read for the admin review surfaces
    // (verifications drawer + /admin/photographers/[handle]). Returns
    // presigned URLs for cover/watermark/payout-QR so the admin can preview
    // the actual media the photographer uploaded. 404s for non-photographer
    // users so the FE doesn't render an empty review pane.
    @Transactional(readOnly = true)
    fun photographerSettings(userId: UUID): AdminPhotographerSettingsDto {
        val (user, settings) = loadUserAndSettings(userId)
        val s = requirePhotographerSettings(user, settings)

        val region = if (!s.regionCode.isNullOrBlank() && !s.provinceCode.isNullOrBlank()) {
            AdminPhotographerRegionDto(
                regionCode = s.regionCode!!,
                provinceCode = s.provinceCode!!,
                city = s.city,
            )
        } else null

        val cover = if (s.coverS3Key != null || s.coverGradientFrom != null || s.coverGradientTo != null) {
            AdminPhotographerCoverDto(
                url = s.coverS3Key?.let {
                    storageService.presignedGetUrl(it, storageProperties.presignedTtl.cover)
                },
                gradientFrom = s.coverGradientFrom,
                gradientTo = s.coverGradientTo,
            )
        } else null

        val watermark = if (s.watermarkS3Key != null || s.watermarkLabel != null) {
            AdminPhotographerWatermarkDto(
                dataUrl = s.watermarkS3Key?.let {
                    storageService.presignedGetUrl(it, storageProperties.presignedTtl.watermark)
                },
                label = s.watermarkLabel,
            )
        } else null

        val socials: List<SocialLinkDto> = socialLinkRepository
            .findAllByUserIdOrderByCreatedAtAsc(userId)
            .map { it.toDto() }

        val payouts: List<PayoutAccountDto> = payoutAccountRepository
            .findAllByUserIdOrderByCreatedAtAsc(userId)
            .map { account ->
                val qrUrl = account.qrS3Key?.let {
                    storageService.presignedGetUrl(it, storageProperties.presignedTtl.cover)
                }
                account.toDto(qrUrl = qrUrl, qrUploadedAt = account.createdAt)
            }

        return AdminPhotographerSettingsDto(
            userId = userId.toString(),
            handle = s.handle,
            brandName = s.brandName,
            brandColor = s.brandColor,
            bio = s.bio,
            avatarUrl = userDtoMapper.resolveAvatarUrl(user),
            region = region,
            cover = cover,
            watermark = watermark,
            socials = socials,
            payouts = payouts,
        )
    }

    fun approve(adminId: UUID, userId: UUID): AdminUserRowDto {
        val (user, settings) = loadUserAndSettings(userId)
        val s = requirePhotographerSettings(user, settings)
        // Re-validate against the SAME required-field set the photographer's own
        // submit gate uses, so admin approval and self-submit can never diverge.
        // Without this an admin could approve a profile that never passed submit
        // — including one whose settings row requirePhotographerSettings just
        // lazy-created empty — and the public profile would render broken.
        val incomplete = photographerSettingsService.collectMissing(userId, s)
        if (!incomplete.isComplete) {
            throw ApiException(
                status = HttpStatus.UNPROCESSABLE_ENTITY,
                code = ErrorCodes.INCOMPLETE_PROFILE,
                message = "Cannot approve — profile is missing: ${incomplete.missing.joinToString(", ")}",
            )
        }
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

    fun resetVerification(adminId: UUID, userId: UUID, reason: String): AdminUserRowDto {
        val (user, settings) = loadUserAndSettings(userId)
        val s = requirePhotographerSettings(user, settings)
        s.verificationStatus = VerificationStatus.INCOMPLETE
        photographerSettingsRepository.save(s)
        val decision = adminDecisionLogService.logUserDecision(
            adminId = adminId,
            targetUserId = userId,
            decision = "reset",
            reason = reason,
        )
        adminDecisionLogService.pushMessage(
            photographerId = userId,
            kind = PhotographerMessageKind.VERIFICATION_RESET,
            body = "Your verification status was reset by support. Reason: $reason. You can re-submit when ready.",
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
        // Kill every live session. The access token's `suspended` claim expires
        // the current one within its TTL; without this the user could keep
        // rotating refresh tokens and mint fresh, unsuspended-looking access
        // tokens forever. unsuspend() needs no counterpart — the user logs in.
        refreshTokenService.revokeAllForUser(userId)
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
        } else if (user.role == Role.RUNNER) {
            runnerMessagesService.pushMessage(
                runnerId = userId,
                kind = RunnerMessageKind.ACCOUNT_SUSPENDED,
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
        } else if (user.role == Role.RUNNER) {
            runnerMessagesService.pushMessage(
                runnerId = userId,
                kind = RunnerMessageKind.ACCOUNT_UNSUSPENDED,
                body = "Your account has been reinstated.",
                sourceAdminId = adminId,
                sourceDecisionId = decision.id,
            )
        }
        return hydrateRow(user, settings)
    }

    fun sendMessage(
        adminId: UUID,
        userId: UUID,
        subject: String,
        body: String,
    ): PhotographerMessageDto {
        val (user, _) = loadUserAndSettings(userId)
        // Free-form admin DM lands in the role-appropriate inbox. ADMIN
        // accounts can't receive messages (they're the ones sending).
        if (user.role == Role.ADMIN) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "Messages can only be sent to photographers or runners",
                field = "userId",
            )
        }
        val trimmedSubject = subject.trim()
        val trimmedBody = body.trim()
        val decision = adminDecisionLogService.logUserDecision(
            adminId = adminId,
            targetUserId = userId,
            decision = "messaged",
            reason = trimmedSubject,
        )

        // PhotographerMessageDto is the wire shape for both inboxes — the
        // fields (id / kind / title / body / sourceDecisionId / createdAt /
        // readAt) map 1:1 between RunnerMessage and PhotographerMessage.
        // The admin UI only consumes id + createdAt for optimistic state.
        return when (user.role) {
            Role.PHOTOGRAPHER -> {
                val message: PhotographerMessage = adminDecisionLogService.pushMessage(
                    photographerId = userId,
                    kind = PhotographerMessageKind.ADMIN_MESSAGE,
                    title = trimmedSubject,
                    body = trimmedBody,
                    sourceAdminId = adminId,
                    sourceDecisionId = decision.id,
                )
                PhotographerMessageDto(
                    id = message.id,
                    kind = message.kindWire,
                    title = message.title,
                    body = message.body,
                    sourceDecisionId = message.sourceDecisionId,
                    createdAt = message.createdAt,
                    readAt = message.readAt,
                )
            }
            Role.RUNNER -> {
                val message: RunnerMessage = runnerMessagesService.pushMessage(
                    runnerId = userId,
                    kind = RunnerMessageKind.ADMIN_MESSAGE,
                    title = trimmedSubject,
                    body = trimmedBody,
                    sourceAdminId = adminId,
                    sourceDecisionId = decision.id,
                )
                PhotographerMessageDto(
                    id = message.id,
                    kind = message.kindWire,
                    title = message.title,
                    body = message.body,
                    sourceDecisionId = message.sourceDecisionId,
                    createdAt = message.createdAt,
                    readAt = message.readAt,
                )
            }
            else -> error("unreachable — Admin already blocked above")
        }
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

    internal fun hydrateRow(
        user: User,
        settings: PhotographerSettings?,
        regions: List<com.quickpitik.dto.reference.RegionDto> = regionsService.all(),
    ): AdminUserRowDto {
        val verificationStatus = when {
            user.role == Role.PHOTOGRAPHER -> (settings?.verificationStatus ?: VerificationStatus.INCOMPLETE).toWire()
            // Runners are always "approved" per the FE store contract.
            else -> "approved"
        }
        val regionLabel = settings?.let {
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
            avatarUrl = userDtoMapper.resolveAvatarUrl(user),
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
        // H-6: cap on rows returned to the admin user-detail page.
        const val DECISION_LOG_DETAIL_CAP = 50
    }
}
