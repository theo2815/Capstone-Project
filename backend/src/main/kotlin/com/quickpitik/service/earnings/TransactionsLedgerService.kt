package com.quickpitik.service.earnings

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginationParams
import com.quickpitik.dto.earnings.PhotographerTransactionDto
import com.quickpitik.dto.earnings.TransactionsLedgerResponse
import com.quickpitik.dto.earnings.privacyBuyerDisplay
import com.quickpitik.dto.earnings.toDto
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotographerPhotoDtoMapper
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.UUID

/**
 * Builds the photographer's transactions ledger. monthTotals is computed over
 * ALL of the photographer's rows (not just the current page) so the FE's
 * sticky-header math stays correct as the user scrolls — a paged total alone
 * would understate the month once rows roll past the limit.
 *
 * Buyer display is the snapshot stored on the row at sale time; if the row
 * has no snapshot (rare — only legacy rows from before snapshotting was wired)
 * the service falls back to the live User.name. ON DELETE SET NULL on
 * buyer_id keeps the ledger readable even after a runner deletes their account.
 */
@Service
@Transactional(readOnly = true)
class TransactionsLedgerService(
    private val transactionRepository: TransactionRepository,
    private val userRepository: UserRepository,
    private val eventRepository: EventRepository,
    private val photoRepository: PhotoRepository,
    private val photoDtoMapper: PhotographerPhotoDtoMapper,
) {
    fun list(photographerId: UUID, params: PaginationParams): TransactionsLedgerResponse {
        val page = transactionRepository.pageForPhotographer(
            photographerId = photographerId,
            pageable = OffsetLimitPageable(params),
        )

        // Resolve buyer displays: prefer the snapshot, fall back to live name.
        val needsLiveLookup = page.content
            .filter { it.buyerDisplayName.isBlank() && it.buyerId != null }
            .mapNotNull { it.buyerId }
            .toSet()
        val liveBuyers = if (needsLiveLookup.isNotEmpty()) {
            userRepository.findAllById(needsLiveLookup)
                .associate { it.id to privacyBuyerDisplay(it.name) }
        } else {
            emptyMap()
        }

        // Resolve current retained event name/slug per row so the FE billing
        // ledger can show the title without a parallel events lookup.
        val eventIds = page.content.map { it.eventId }.toSet()
        val eventsById = if (eventIds.isNotEmpty()) {
            eventRepository.findAllById(eventIds).associateBy { it.id }
        } else {
            emptyMap()
        }

        val photoIds = page.content.map { it.photoId }.toSet()
        val thumbnailsByPhotoId = if (photoIds.isNotEmpty()) {
            photoRepository.findAllById(photoIds)
                .filter { it.photographerId == photographerId }
                .associate { it.id to photoDtoMapper.thumbnailUrl(it) }
        } else {
            emptyMap()
        }

        val items = page.content.map { tx ->
            val display = tx.buyerDisplayName.takeIf { it.isNotBlank() }
                ?: liveBuyers[tx.buyerId]
                ?: ""
            val event = eventsById[tx.eventId]
            tx.toDto(
                buyerDisplay = display,
                eventName = event?.name,
                eventSlug = event?.slug,
                thumbnailUrl = thumbnailsByPhotoId[tx.photoId],
            )
        }

        val monthTotals = computeMonthTotals(photographerId)

        return TransactionsLedgerResponse(
            items = items,
            total = page.totalElements,
            offset = params.offset,
            limit = params.limit,
            monthTotals = monthTotals,
        )
    }

    private fun computeMonthTotals(photographerId: UUID): Map<String, BigDecimal> {
        // Pull every row for the photographer to bucket by YYYY-MM in PHT. At
        // realistic per-photographer scale (a few thousand sales total over a
        // year) this is cheap enough to not need a SQL date_trunc; if a single
        // photographer ever crosses a 5-figure row count we'll add a
        // GROUP BY query. PR 9 ships the simple path.
        val zone = DISPLAY_ZONE
        val rows = transactionRepository.pickAmountsForBucketing(
            photographerId = photographerId,
            from = MIN_INSTANT,
            to = MAX_INSTANT,
        )
        if (rows.isEmpty()) return emptyMap()
        val totals = LinkedHashMap<String, BigDecimal>()
        for (row in rows) {
            val paidAt = row[0] as java.time.OffsetDateTime
            val amount = row[1] as BigDecimal
            val key = paidAt.atZoneSameInstant(zone).format(MONTH_KEY)
            totals.compute(key) { _, existing -> (existing ?: BigDecimal.ZERO).add(amount) }
        }
        return totals
    }

    private companion object {
        val DISPLAY_ZONE: ZoneId = ZoneId.of("Asia/Manila")
        val MONTH_KEY: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM")
        // Window the bucketing query against — using MIN/MAX bounds rather than
        // a separate "all rows" query keeps the repository surface lean.
        val MIN_INSTANT: java.time.OffsetDateTime =
            java.time.OffsetDateTime.parse("1970-01-01T00:00:00Z")
        val MAX_INSTANT: java.time.OffsetDateTime =
            java.time.OffsetDateTime.parse("9999-12-31T23:59:59Z")
    }
}
