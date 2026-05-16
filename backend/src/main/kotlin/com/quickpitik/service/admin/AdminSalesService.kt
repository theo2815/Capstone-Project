package com.quickpitik.service.admin

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.admin.AdminSalesEventRowDto
import com.quickpitik.dto.admin.AdminSalesKpisDto
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.TransactionRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.temporal.WeekFields
import java.util.Locale
import java.util.UUID

@Service
@Transactional(readOnly = true)
class AdminSalesService(
    private val transactionRepository: TransactionRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val eventRepository: EventRepository,
    private val platformProperties: PlatformProperties,
) {

    fun kpis(rangeRaw: String?): AdminSalesKpisDto {
        val (from, to) = windowFor(rangeRaw)
        val photographerKeep = transactionRepository.sumPhotographerKeepInWindow(from, to)
        val refunds = transactionRepository.sumRefundsInWindow(from, to)
        val totalSales = transactionRepository.countSalesInWindow(from, to)
        val keepRate = platformProperties.photographerKeepRate
        // GMV is reconstructed from photographer keep / keepRate so we don't
        // need a parallel source-of-truth on order_items totals (the
        // transactions table already nets refunds via the negative-amount
        // row pattern from PR 9).
        val gmv = if (keepRate.signum() == 0) BigDecimal.ZERO
        else photographerKeep.divide(keepRate, 2, RoundingMode.HALF_UP)
        val platformRevenue = gmv.subtract(photographerKeep).max(BigDecimal.ZERO)
        return AdminSalesKpisDto(
            gmv = gmv,
            platformRevenue = platformRevenue,
            refundsIssued = refunds,
            netPlatformRevenue = platformRevenue.subtract(refunds.multiply(BigDecimal.ONE.subtract(keepRate))),
            photographerKeep = photographerKeep,
            totalSalesCount = totalSales,
        )
    }

    fun byEvent(orderRaw: String?, params: PaginationParams): PaginatedResponse<AdminSalesEventRowDto> {
        // Aggregate transactions grouped by event_id, then join against the
        // events table for slug / name / date / status. Sort + paginate
        // in-memory because we expect the dashboard to show the top-N
        // events (low double-digit count); sorting on the DB side would
        // require a more complex aggregation query than the saved volume
        // justifies for v1.
        val rows = eventPhotographerRepository.salesAggregatesByEvent()
            .map { row ->
                val eventId = row[0] as UUID
                val impliedCut = (row[1] as? Number)?.let { BigDecimal(it.toString()) } ?: BigDecimal.ZERO
                val refunds = (row[2] as? Number)?.let { BigDecimal(it.toString()) } ?: BigDecimal.ZERO
                eventId to Pair(impliedCut, refunds)
            }
            .toMap()

        val events = eventRepository.findAllById(rows.keys).filter { it.deletedAt == null }
        val keepRate = platformProperties.photographerKeepRate
        val items = events.mapNotNull { event ->
            val (impliedCut, refunds) = rows[event.id] ?: return@mapNotNull null
            val impliedGmv = if (keepRate.signum() == 0) BigDecimal.ZERO
            else impliedCut.divide(keepRate, 2, RoundingMode.HALF_UP)
            event to AdminSalesEventRowDto(
                id = event.id.toString(),
                slug = event.slug,
                name = event.name,
                date = event.date.toString(),
                city = com.quickpitik.service.events.EventDtoMapper.cityFromLocation(event.location),
                status = event.status.name,
                state = com.quickpitik.service.events.EventDtoMapper.deriveAdminEventState(event),
                photoCount = event.photoCount,
                impliedGmv = impliedGmv,
                impliedCut = impliedCut,
                refundsIssued = refunds,
            )
        }

        val sorted = when (orderRaw?.trim()?.lowercase()) {
            "refunds" -> items.sortedByDescending { it.second.refundsIssued }
            else -> items.sortedByDescending { it.second.impliedGmv }
        }
        val total = sorted.size.toLong()
        val sliced = sorted.drop(params.offset).take(params.limit).map { it.second }
        return PaginatedResponse.of(sliced, total, params)
    }

    private fun windowFor(rangeRaw: String?): Pair<OffsetDateTime, OffsetDateTime> {
        val today = LocalDate.now(PH_ZONE)
        return when (rangeRaw?.trim()?.lowercase()) {
            "week" -> {
                val weekFields = WeekFields.of(Locale.UK) // Monday-start
                val monday = today.with(weekFields.dayOfWeek(), 1)
                val from = monday.atStartOfDay(PH_ZONE).toOffsetDateTime()
                val to = monday.plusWeeks(1).atStartOfDay(PH_ZONE).toOffsetDateTime()
                from to to
            }
            "month" -> {
                val first = today.withDayOfMonth(1)
                val from = first.atStartOfDay(PH_ZONE).toOffsetDateTime()
                val to = first.plusMonths(1).atStartOfDay(PH_ZONE).toOffsetDateTime()
                from to to
            }
            else -> {
                // YTD default — Jan 1 of the current year (PHT) → "now+1d".
                val first = LocalDate.of(today.year, 1, 1)
                val from = first.atStartOfDay(PH_ZONE).toOffsetDateTime()
                val to = today.plusDays(1).atStartOfDay(PH_ZONE).toOffsetDateTime()
                from to to
            }
        }
    }
}
