package com.quickpitik.service.admin

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.admin.AdminSalesEventRowDto
import com.quickpitik.dto.admin.AdminSalesKpisDto
import com.quickpitik.repository.DisputeRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.TransactionRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.temporal.WeekFields
import java.util.Locale
import java.util.UUID

@Service
@Transactional(readOnly = true)
class AdminSalesService(
    private val transactionRepository: TransactionRepository,
    private val disputeRepository: DisputeRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val eventRepository: EventRepository,
) {

    fun kpis(rangeRaw: String?): AdminSalesKpisDto {
        val (from, to) = windowFor(rangeRaw)
        val photographerKeep = transactionRepository.sumPhotographerKeepInWindow(from, to)
        val discounts = transactionRepository.sumDiscountsInWindow(from, to)
        val refunds = disputeRepository.sumSucceededRefundsInWindow(from, to)
        val totalSales = transactionRepository.countSalesInWindow(from, to)
        // GMV is the list price the sold items were charged from (order_items),
        // net of refunded shares — never kept / keepRate, which would restate
        // history whenever the configured cut changes (M-16). A coupon (V45)
        // lowers kept but not the share the platform fee comes out of, so the
        // discount is added back before taking the fee.
        val gmv = transactionRepository.sumListGrossInWindow(from, to)
        val grossBase = photographerKeep.add(discounts)
        val platformRevenue = gmv.subtract(grossBase).max(BigDecimal.ZERO)
        return AdminSalesKpisDto(
            gmv = gmv,
            platformRevenue = platformRevenue,
            refundsIssued = refunds,
            // The keep and discount sums already include negative refund rows.
            netPlatformRevenue = platformRevenue,
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
                val keep = (row[1] as? Number)?.let { BigDecimal(it.toString()) } ?: BigDecimal.ZERO
                val discounts = (row.getOrNull(3) as? Number)?.let { BigDecimal(it.toString()) } ?: BigDecimal.ZERO
                val listGross = (row.getOrNull(4) as? Number)?.let { BigDecimal(it.toString()) } ?: BigDecimal.ZERO
                eventId to Triple(keep, discounts, listGross)
            }
            .toMap()
        val refundsByEvent = disputeRepository.sumSucceededRefundsByEvent().associate { row ->
            val eventId = row[0] as UUID
            eventId to ((row[1] as? Number)?.let { BigDecimal(it.toString()) } ?: BigDecimal.ZERO)
        }

        val events = eventRepository.findAllById(rows.keys).filter { it.deletedAt == null }
        val items = events.mapNotNull { event ->
            val (keep, discounts, impliedGmv) = rows[event.id] ?: return@mapNotNull null
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
                // Same add-back as kpis(): kept + coupon discounts is the share
                // the platform fee was taken from.
                impliedCut = impliedGmv.subtract(keep.add(discounts)),
                refundsIssued = refundsByEvent[event.id] ?: BigDecimal.ZERO,
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
