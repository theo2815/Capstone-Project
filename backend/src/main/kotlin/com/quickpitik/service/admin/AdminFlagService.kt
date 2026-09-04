package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.AdminFlagDto
import com.quickpitik.dto.admin.FlagPhotoSnapshotDto
import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class AdminFlagService(
    private val flagRepository: FlagRepository,
    private val photoRepository: PhotoRepository,
    private val eventRepository: EventRepository,
    private val userRepository: UserRepository,
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val adminProperties: AdminProperties,
) {

    fun ensureEnabled() {
        if (!adminProperties.flagsEnabled) {
            throw ApiException(
                status = HttpStatus.FORBIDDEN,
                code = ErrorCodes.FLAGS_DISABLED,
                message = "Flagging surface is disabled. Set ADMIN_FLAGS_ENABLED=true to enable.",
            )
        }
    }

    @Transactional(readOnly = true)
    fun list(
        statusFilter: String?,
        query: String?,
        params: PaginationParams,
    ): PaginatedResponse<AdminFlagDto> {
        ensureEnabled()
        val statusWire = statusFilter?.takeIf { it.isNotBlank() }?.let { parseStatus(it).wire }
        // ponytail: LIKE wildcards in q are left as-is (admin-only search); only the length is capped.
        val q = query?.takeIf { it.isNotBlank() }?.take(100)
        val page = flagRepository.pageForAdmin(statusWire, q, OffsetLimitPageable(params))
        if (page.isEmpty) return PaginatedResponse.empty(params)
        val items = hydrateMany(page.content)
        return PaginatedResponse.of(items, page.totalElements, params)
    }

    // Transitions: OPEN → hide / dismiss / resolve / escalate; ESCALATED → hide /
    // dismiss / resolve; HIDDEN → dismiss / resolve (restores the photo when no
    // other hidden flag still targets it). Everything else is 409.

    fun resolve(adminId: UUID, flagId: UUID, resolutionNote: String?): AdminFlagDto =
        close(adminId, flagId, resolutionNote, FlagStatus.RESOLVED)

    fun dismiss(adminId: UUID, flagId: UUID, resolutionNote: String?): AdminFlagDto =
        close(adminId, flagId, resolutionNote, FlagStatus.DISMISSED)

    fun hide(adminId: UUID, flagId: UUID, resolutionNote: String?): AdminFlagDto {
        ensureEnabled()
        val flag = loadFlag(flagId)
        requireStatus(flag, FlagStatus.OPEN, FlagStatus.ESCALATED)
        flag.status = FlagStatus.HIDDEN
        flag.resolutionNote = resolutionNote
        flag.resolvedBy = adminId
        flag.resolvedAt = OffsetDateTime.now()
        flagRepository.save(flag)

        if (flag.targetKind == FlagTargetKind.PHOTO) {
            photoRepository.findById(flag.targetId).ifPresent { photo ->
                photo.status = PhotoStatus.HIDDEN
                photoRepository.save(photo)
            }
        }

        return hydrateOne(flag)
    }

    fun escalate(adminId: UUID, flagId: UUID, resolutionNote: String?): AdminFlagDto {
        ensureEnabled()
        val flag = loadFlag(flagId)
        requireStatus(flag, FlagStatus.OPEN)
        flag.status = FlagStatus.ESCALATED
        // Escalation is not a resolution — the reviewer stamp stays empty until
        // the higher tier hides or dismisses.
        flag.resolutionNote = resolutionNote
        flagRepository.save(flag)
        return hydrateOne(flag)
    }

    private fun close(adminId: UUID, flagId: UUID, resolutionNote: String?, target: FlagStatus): AdminFlagDto {
        ensureEnabled()
        val flag = loadFlag(flagId)
        requireStatus(flag, FlagStatus.OPEN, FlagStatus.ESCALATED, FlagStatus.HIDDEN)
        val wasHidden = flag.status == FlagStatus.HIDDEN
        flag.status = target
        flag.resolutionNote = resolutionNote
        flag.resolvedBy = adminId
        flag.resolvedAt = OffsetDateTime.now()
        flagRepository.save(flag)
        if (wasHidden) restorePhotoIfUnflagged(flag)
        return hydrateOne(flag)
    }

    private fun requireStatus(flag: Flag, vararg allowed: FlagStatus) {
        if (flag.status !in allowed) {
            throw ApiException(
                status = HttpStatus.CONFLICT,
                code = ErrorCodes.INVALID_STATE_TRANSITION,
                message = "Flag is already ${flag.statusWire}",
            )
        }
    }

    private fun restorePhotoIfUnflagged(flag: Flag) {
        if (flag.targetKind != FlagTargetKind.PHOTO) return
        val stillHiddenByAnother = flagRepository.existsByTargetKindWireAndTargetIdAndStatusWireAndIdNot(
            flag.targetKindWire, flag.targetId, FlagStatus.HIDDEN.wire, flag.id,
        )
        if (stillHiddenByAnother) return
        photoRepository.findById(flag.targetId).ifPresent { photo ->
            if (photo.status == PhotoStatus.HIDDEN) {
                photo.status = PhotoStatus.LIVE
                photoRepository.save(photo)
            }
        }
    }

    private fun loadFlag(flagId: UUID): Flag =
        flagRepository.findById(flagId).orElseThrow {
            NotFoundException(code = ErrorCodes.FLAG_NOT_FOUND, message = "Flag not found")
        }

    private fun parseStatus(raw: String): FlagStatus =
        runCatching { FlagStatus.fromWire(raw) }.getOrElse {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "status must be one of open / resolved / hidden / dismissed / escalated",
                field = "status",
            )
        }

    private fun hydrateMany(flags: List<Flag>): List<AdminFlagDto> {
        val photoIds = flags.filter { it.targetKind == FlagTargetKind.PHOTO }.map { it.targetId }.distinct()
        val photosById = if (photoIds.isNotEmpty()) photoRepository.findAllById(photoIds).associateBy { it.id } else emptyMap()

        val eventIds = photosById.values.map { it.eventId }.distinct()
        val eventsById = if (eventIds.isNotEmpty()) eventRepository.findAllById(eventIds).associateBy { it.id } else emptyMap()

        val photographerIds = photosById.values.mapNotNull { it.photographerId }.distinct()
        val photographerSettingsById = if (photographerIds.isNotEmpty()) {
            photographerSettingsRepository.findAllById(photographerIds).associateBy { it.userId }
        } else emptyMap()

        val userIds = (flags.mapNotNull { it.reporterId } + flags.mapNotNull { it.resolvedBy } + photographerIds).distinct()
        val usersById = if (userIds.isNotEmpty()) userRepository.findAllById(userIds).associateBy { it.id } else emptyMap()

        return flags.map { flag ->
            val photo = if (flag.targetKind == FlagTargetKind.PHOTO) photosById[flag.targetId] else null
            val event = photo?.let { eventsById[it.eventId] }
            val photographerUser = photo?.photographerId?.let { usersById[it] }
            val photographerSettings = photo?.photographerId?.let { photographerSettingsById[it] }
            val reporterUser = flag.reporterId?.let { usersById[it] }
            val reviewerUser = flag.resolvedBy?.let { usersById[it] }

            val photographerHandle = photographerSettings?.handle?.ifBlank { null }
                ?: photographerUser?.name?.ifBlank { null }
                ?: "photographer"

            val reportedBy = reporterUser?.name?.ifBlank { null }
                ?: reporterUser?.email?.ifBlank { null }
                ?: "system"

            val reviewedBy = reviewerUser?.name?.ifBlank { null }
                ?: reviewerUser?.email?.ifBlank { null }
                ?: if (flag.resolvedBy != null) "admin" else null

            // Never the original s3Key: a still-PROCESSING photo has no derived
            // asset yet and the card falls back to its gradient placeholder.
            val thumbnailUrl = (photo?.watermarkS3Key ?: photo?.thumbnailS3Key)?.let { key ->
                storageService.presignedGetUrl(key, storageProperties.presignedTtl.thumbnail)
            }

            AdminFlagDto(
                id = flag.id,
                photoId = if (flag.targetKind == FlagTargetKind.PHOTO) flag.targetId else null,
                eventId = event?.id ?: photo?.eventId,
                eventName = event?.name,
                photographerHandle = photographerHandle,
                photographerName = photographerUser?.name,
                reportedBy = reportedBy,
                reason = flag.reason,
                note = flag.note,
                status = flag.statusWire,
                reportedAt = flag.createdAt.toString(),
                reviewedAt = flag.resolvedAt?.toString(),
                reviewedBy = reviewedBy,
                reviewerNote = flag.resolutionNote,
                photoSnapshot = FlagPhotoSnapshotDto(
                    alt = photo?.altText ?: (event?.name?.let { "$it photo" } ?: "Flagged photo"),
                    kmMark = photo?.km,
                    bib = photo?.bibs?.minByOrNull { it.bibNumber }?.bibNumber,
                    thumbnailUrl = thumbnailUrl,
                ),
                targetKind = flag.targetKindWire,
                targetId = flag.targetId,
                reporterId = flag.reporterId,
            )
        }
    }

    private fun hydrateOne(flag: Flag): AdminFlagDto = hydrateMany(listOf(flag)).first()
}
