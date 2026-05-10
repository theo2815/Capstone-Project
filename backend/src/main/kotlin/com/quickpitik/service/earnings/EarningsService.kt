package com.quickpitik.service.earnings

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.earnings.EarningsOverviewDto
import com.quickpitik.dto.earnings.PerEventEarningDto
import com.quickpitik.dto.earnings.WeeklyRevenuePointDto
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.TransactionRepository
import org.springframework.data.domain.PageRequest
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.DayOfWeek
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.util.UUID
import kotlin.collections.LinkedHashMap

/**
 * Server-computed earnings rollups + per-event aggregations. Reads from
 * `transactions` (sales + refund-as-negative-amount per Q-E3) for every money
 * total so refunds attribute correctly without a separate ledger join.
 *
 * The 12-week series is bucketed in-memory off raw rows because the row count
 * is small (one photographer's 12 weeks of sales — typically <500 rows even at
 * peak Cebu Marathon volume) and writing portable date-trunc SQL across
 * Postgres + the test H2 buffer is more friction than the pull is worth.
 *
 * Per-event rollups read from PR 7's `event_photographer` table — the same
 * table the photographer-events list uses — so PR 7's photoCount and the new
 * salesCount/revenueKept counters stay coherent under one query path. Bumping
 * those counters is OrderService's responsibility (PR 5 + PR 9 wiring).
 */
@Service
@Transactional(readOnly = true)
class EarningsService(
    private val transactionRepository: TransactionRepository,
    private val payoutCycleRepository: PayoutCycleRepository,
    private val eventRepository: EventRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
) {
    fun getOverview(photographerId: UUID): EarningsOverviewDto {
        val zone = DISPLAY_ZONE
        val nowZoned = OffsetDateTime.now().atZoneSameInstant(zone)
        val today = nowZoned.toLocalDate()
        val currentWeekMonday = today.with(DayOfWeek.MONDAY)
        val firstOfMonth = today.withDayOfMonth(1)

        val seriesStart = currentWeekMonday.minusWeeks(WEEKLY_SERIES_LENGTH - 1L)
        val seriesEndExclusive = currentWeekMonday.plusWeeks(1)

        val weekStartInstant = currentWeekMonday.atStartOfDay(zone).toOffsetDateTime()
        val weekEndInstant = currentWeekMonday.plusWeeks(1).atStartOfDay(zone).toOffsetDateTime()
        val monthStartInstant = firstOfMonth.atStartOfDay(zone).toOffsetDateTime()
        val monthEndInstant = firstOfMonth.plusMonths(1).atStartOfDay(zone).toOffsetDateTime()
        val seriesStartInstant = seriesStart.atStartOfDay(zone).toOffsetDateTime()
        val seriesEndInstant = seriesEndExclusive.atStartOfDay(zone).toOffsetDateTime()

        val lifetime = transactionRepository.sumLifetime(photographerId)
        val thisWeek = transactionRepository.sumWindow(photographerId, weekStartInstant, weekEndInstant)
        val thisMonth = transactionRepository.sumWindow(photographerId, monthStartInstant, monthEndInstant)
        val thisWeekSold = transactionRepository.countSalesWindow(photographerId, weekStartInstant, weekEndInstant)
        val thisMonthSold = transactionRepository.countSalesWindow(photographerId, monthStartInstant, monthEndInstant)

        val weeklySeries = buildWeeklySeries(
            photographerId = photographerId,
            seriesStart = seriesStart,
            seriesEnd = seriesEndExclusive,
            from = seriesStartInstant,
            to = seriesEndInstant,
        )

        val payoutPending = payoutCycleRepository.sumOpenAmount(photographerId)
        val nextCycle = payoutCycleRepository.findNextUpcoming(
            photographerId,
            PageRequest.of(0, 1),
        ).firstOrNull()
        val payoutScheduledFor = nextCycle?.settledAt?.atZoneSameInstant(zone)?.toLocalDate()
            ?: nextCycle?.weekOf?.plusDays(SCHEDULED_FOR_OFFSET_DAYS)

        return EarningsOverviewDto(
            lifetimeKept = lifetime,
            thisWeek = thisWeek,
            thisMonth = thisMonth,
            payoutPending = payoutPending,
            payoutScheduledFor = payoutScheduledFor,
            weeklySeries = weeklySeries,
            thisWeekSold = thisWeekSold,
            thisMonthSold = thisMonthSold,
        )
    }

    fun listPerEvent(
        photographerId: UUID,
        params: PaginationParams,
    ): PaginatedResponse<PerEventEarningDto> {
        val rows = eventPhotographerRepository.pageEarningsForPhotographer(
            photographerId = photographerId,
            pageable = OffsetLimitPageable(params),
        )
        if (rows.isEmpty) return PaginatedResponse.empty(params)

        val eventIds = rows.content.map { it.id.eventId }
        val events = eventRepository.findAllById(eventIds).associateBy { it.id }
        val items = rows.content.mapNotNull { row ->
            val event = events[row.id.eventId] ?: return@mapNotNull null
            PerEventEarningDto(
                eventId = row.id.eventId.toString(),
                eventName = event.name,
                eventDate = event.date,
                photoCount = row.photoCount,
                salesCount = row.salesCount,
                revenueKept = row.revenueKeptPhp,
            )
        }
        return PaginatedResponse.of(items, rows.totalElements, params)
    }

    private fun buildWeeklySeries(
        photographerId: UUID,
        seriesStart: LocalDate,
        seriesEnd: LocalDate,
        from: OffsetDateTime,
        to: OffsetDateTime,
    ): List<WeeklyRevenuePointDto> {
        // Pre-fill every Monday with zero so the FE always gets exactly
        // WEEKLY_SERIES_LENGTH points oldest-first, even when the photographer
        // has no transactions in some weeks.
        val buckets = LinkedHashMap<LocalDate, BigDecimal>(WEEKLY_SERIES_LENGTH)
        var monday = seriesStart
        while (monday.isBefore(seriesEnd)) {
            buckets[monday] = BigDecimal.ZERO
            monday = monday.plusWeeks(1)
        }

        val rows = transactionRepository.pickAmountsForBucketing(photographerId, from, to)
        for (row in rows) {
            val paidAt = row[0] as OffsetDateTime
            val amount = row[1] as BigDecimal
            val weekMonday = paidAt
                .atZoneSameInstant(DISPLAY_ZONE)
                .toLocalDate()
                .with(DayOfWeek.MONDAY)
            buckets.compute(weekMonday) { _, existing ->
                (existing ?: BigDecimal.ZERO).add(amount)
            }
        }

        return buckets.map { (week, amount) -> WeeklyRevenuePointDto(week, amount) }
    }

    private companion object {
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        const val WEEKLY_SERIES_LENGTH = 12
        // Cycles minted without an explicit settled_at default to "Monday + 4
        // days" (Friday of the cycle week) per the FE mock cadence — first
        // approximation until the cycle generator lands.
        const val SCHEDULED_FOR_OFFSET_DAYS = 4L
    }
}
