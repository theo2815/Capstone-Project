package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AdminProperties
import com.quickpitik.dto.admin.AdminFlagDto
import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.FlagRepository
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.util.UUID

@Service
@Transactional
class AdminFlagService(
    private val flagRepository: FlagRepository,
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
        params: PaginationParams,
    ): PaginatedResponse<AdminFlagDto> {
        ensureEnabled()
        val statusWire = statusFilter?.takeIf { it.isNotBlank() }?.let { parseStatus(it).wire }
        val page = flagRepository.pageByStatus(statusWire, OffsetLimitPageable(params))
        if (page.isEmpty) return PaginatedResponse.empty(params)
        return PaginatedResponse.of(page.content.map { it.toDto() }, page.totalElements, params)
    }

    fun resolve(adminId: UUID, flagId: UUID, resolutionNote: String?): AdminFlagDto {
        ensureEnabled()
        val flag = loadFlag(flagId)
        flag.status = FlagStatus.RESOLVED
        flag.resolutionNote = resolutionNote
        flag.resolvedBy = adminId
        flag.resolvedAt = OffsetDateTime.now()
        flagRepository.save(flag)
        return flag.toDto()
    }

    fun hide(adminId: UUID, flagId: UUID, resolutionNote: String?): AdminFlagDto {
        ensureEnabled()
        val flag = loadFlag(flagId)
        flag.status = FlagStatus.HIDDEN
        flag.resolutionNote = resolutionNote
        flag.resolvedBy = adminId
        flag.resolvedAt = OffsetDateTime.now()
        flagRepository.save(flag)
        return flag.toDto()
    }

    fun dismiss(adminId: UUID, flagId: UUID, resolutionNote: String?): AdminFlagDto {
        ensureEnabled()
        val flag = loadFlag(flagId)
        flag.status = FlagStatus.DISMISSED
        flag.resolutionNote = resolutionNote
        flag.resolvedBy = adminId
        flag.resolvedAt = OffsetDateTime.now()
        flagRepository.save(flag)
        return flag.toDto()
    }

    private fun loadFlag(flagId: UUID): Flag =
        flagRepository.findById(flagId).orElseThrow {
            NotFoundException(code = ErrorCodes.FLAG_NOT_FOUND, message = "Flag not found")
        }

    private fun parseStatus(raw: String): FlagStatus =
        runCatching { FlagStatus.fromWire(raw) }.getOrElse {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "status must be one of open / resolved / hidden / dismissed",
                field = "status",
            )
        }

    private fun Flag.toDto(): AdminFlagDto = AdminFlagDto(
        id = id,
        targetKind = targetKindWire,
        targetId = targetId,
        reporterId = reporterId,
        reason = reason,
        note = note,
        status = statusWire,
        resolutionNote = resolutionNote,
        resolvedBy = resolvedBy,
        resolvedAt = resolvedAt?.toString(),
        createdAt = createdAt.toString(),
    )
}
