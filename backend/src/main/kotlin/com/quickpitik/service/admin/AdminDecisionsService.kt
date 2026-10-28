package com.quickpitik.service.admin

import com.quickpitik.dto.admin.AdminDecisionDto
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.UserRepository
import org.springframework.data.domain.PageRequest
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional

@Service
class AdminDecisionsService(
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val userRepository: UserRepository,
) {

    @Transactional(readOnly = true)
    fun recent(limit: Int?): List<AdminDecisionDto> {
        val rows = adminDecisionLogRepository.findAllByOrderByDecidedAtDescIdAsc(
            PageRequest.of(0, (limit ?: DEFAULT_LIMIT).coerceIn(1, MAX_LIMIT)),
        )
        if (rows.isEmpty()) return emptyList()
        // One lookup covers the admins and the user targets.
        val names = userRepository.findAllById(rows.flatMap { listOfNotNull(it.adminId, it.targetUserId) }.toSet())
            .associate { it.id to it.name.ifBlank { it.email } }
        return rows.map {
            val (type, id) = target(it)
            AdminDecisionDto(it.id, it.decision, it.reason, it.decidedAt, type, id, it.targetUserId?.let(names::get), it.adminId, names[it.adminId])
        }
    }

    // Photo-removal rows have no target column; the writer keeps the case id in meta.requestId.
    private fun target(row: AdminDecisionLog): Pair<String?, String?> {
        val requestId = row.meta?.get("requestId")
        return when {
            row.targetUserId != null -> "user" to row.targetUserId.toString()
            row.targetPayoutId != null -> "payout" to row.targetPayoutId
            row.targetDisputeId != null -> "dispute" to row.targetDisputeId.toString()
            row.targetEventId != null -> "event" to row.targetEventId.toString()
            requestId != null -> "removal" to requestId.toString()
            else -> null to null
        }
    }

    companion object {
        const val DEFAULT_LIMIT = 20
        const val MAX_LIMIT = 100
    }
}
