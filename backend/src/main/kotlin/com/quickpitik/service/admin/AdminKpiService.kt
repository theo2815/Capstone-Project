package com.quickpitik.service.admin

import com.quickpitik.dto.admin.AdminKpisDto
import com.quickpitik.dto.admin.AdminTrendPointDto
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.sql.Date
import java.sql.Timestamp
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.temporal.WeekFields
import java.util.Locale

@Service
@Transactional(readOnly = true)
class AdminKpiService(
    private val photographerSettingsRepository: PhotographerSettingsRepository,
    private val userRepository: UserRepository,
    private val payoutCycleRepository: PayoutCycleRepository,
    private val disputeRepository: DisputeRepository,
    private val flagRepository: FlagRepository,
    private val eventRepository: EventRepository,
    private val adminDecisionLogRepository: AdminDecisionLogRepository,
    private val jdbcTemplate: JdbcTemplate,
) {

    fun kpis(): AdminKpisDto {
        val now = LocalDate.now(PH_ZONE)
        val weekFields = WeekFields.of(Locale.UK)
        val monday = now.with(weekFields.dayOfWeek(), 1)
        val from = monday.atStartOfDay(PH_ZONE).toOffsetDateTime()
        val to = monday.plusWeeks(1).atStartOfDay(PH_ZONE).toOffsetDateTime()

        val pendingVerifications = photographerSettingsRepository.countByVerificationStatus(VerificationStatus.PENDING)
        val approvedPhotographers = photographerSettingsRepository.countByVerificationStatus(VerificationStatus.APPROVED)
        val suspended = countSuspendedUsers()
        val liveEvents = countLiveEventsToday(now)
        val decisionsThisWeek = adminDecisionLogRepository.countLogicalDecisionsBetween(from, to)
        val openDisputes = disputeRepository.countByStatusWire("open")
        val openFlags = flagRepository.countByStatusWire("open")
        val pendingPayouts = payoutCycleRepository.countByStatusWire(PayoutCycleStatus.PENDING.wire) +
            payoutCycleRepository.countByStatusWire(PayoutCycleStatus.SCHEDULED.wire)
        val pendingEventRequests = eventRepository.countByReviewStatusInAndDeletedAtIsNull(
            listOf(EventReviewStatus.PENDING, EventReviewStatus.CHANGE_PENDING),
        )

        return AdminKpisDto(
            pendingVerifications = pendingVerifications,
            approvedPhotographers = approvedPhotographers,
            suspended = suspended,
            liveEvents = liveEvents,
            decisionsThisWeek = decisionsThisWeek,
            openDisputes = openDisputes,
            openFlags = openFlags,
            pendingPayouts = pendingPayouts,
            pendingEventRequests = pendingEventRequests,
        )
    }

    fun trend(daysRaw: Int?): List<AdminTrendPointDto> {
        val days = (daysRaw ?: 30).coerceIn(1, 365)
        val today = LocalDate.now(PH_ZONE)
        val from = today.minusDays((days - 1).toLong()).atStartOfDay(PH_ZONE).toOffsetDateTime()
        val to = today.plusDays(1).atStartOfDay(PH_ZONE).toOffsetDateTime()

        val decisionByDay = adminDecisionLogRepository.decisionCountsByDayBetween(from, to)
            .associate { extractDate(it[0]) to (it[1] as Number).toLong() }
        val disputeByDay = disputesOpenedByDay(from, to)
        val payoutByDay = payoutsApprovedByDay(from, to)

        return (0 until days).map { i ->
            val day = today.minusDays((days - 1 - i).toLong())
            AdminTrendPointDto(
                date = day.toString(),
                decisions = decisionByDay[day] ?: 0L,
                disputes = disputeByDay[day] ?: 0L,
                payouts = payoutByDay[day] ?: 0L,
            )
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────
    private fun countSuspendedUsers(): Long =
        jdbcTemplate.queryForObject(
            "SELECT COUNT(*) FROM users WHERE suspended_at IS NOT NULL",
            Long::class.java,
        ) ?: 0L

    private fun countLiveEventsToday(today: LocalDate): Long =
        jdbcTemplate.queryForObject(
            """
            SELECT COUNT(*) FROM events
            WHERE deleted_at IS NULL AND status = 'ACTIVE' AND date = ?
            """.trimIndent(),
            Long::class.java,
            Date.valueOf(today),
        ) ?: 0L

    private fun disputesOpenedByDay(from: OffsetDateTime, to: OffsetDateTime): Map<LocalDate, Long> {
        val rows = jdbcTemplate.queryForList(
            """
            SELECT opened_at::date AS day, COUNT(*) AS cnt
            FROM disputes
            WHERE opened_at >= ? AND opened_at < ?
            GROUP BY day
            ORDER BY day
            """.trimIndent(),
            Timestamp.from(from.toInstant()),
            Timestamp.from(to.toInstant()),
        )
        return rows.associate {
            val day = (it["day"] as Date).toLocalDate()
            day to (it["cnt"] as Number).toLong()
        }
    }

    private fun payoutsApprovedByDay(from: OffsetDateTime, to: OffsetDateTime): Map<LocalDate, Long> {
        val rows = jdbcTemplate.queryForList(
            """
            SELECT decided_at::date AS day, COUNT(*) AS cnt
            FROM admin_decision_log
            WHERE decided_at >= ? AND decided_at < ?
              AND target_payout_id IS NOT NULL
              AND decision IN ('approved','marked-paid','held')
            GROUP BY day
            ORDER BY day
            """.trimIndent(),
            Timestamp.from(from.toInstant()),
            Timestamp.from(to.toInstant()),
        )
        return rows.associate {
            val day = (it["day"] as Date).toLocalDate()
            day to (it["cnt"] as Number).toLong()
        }
    }

    private fun extractDate(raw: Any?): LocalDate = when (raw) {
        is Date -> raw.toLocalDate()
        is LocalDate -> raw
        else -> LocalDate.parse(raw.toString())
    }
}
