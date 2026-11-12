package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.dto.photographer.RacePriceDto
import com.quickpitik.dto.admin.AdminRacePricesDto
import com.quickpitik.dto.admin.AdminRacePriceHistoryDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.RacePriceChange
import com.quickpitik.entity.RacePriceReason
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RacePriceChangeRepository
import com.quickpitik.service.photographer.PhotographerAccessGuard
import com.quickpitik.service.admin.AdminDecisionLogService
import org.springframework.data.domain.PageRequest
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID

/**
 * A covering photographer's own price in a shared race (AA-R11, auto-approval Plan 2): between
 * the race price (the floor, `events.price_per_photo`) and the ceiling (floor × multiplier),
 * changed at most once per cooldown. Lives in `service/events` because `service/photographer`
 * is past the ~25-file split trigger.
 *
 * Lock order for every price writer: this photographer's photos, then their coverage row, then
 * (AA-R12's admin path) the event. The floor is read after the locks, so a race price change that
 * commits first bounds the new price.
 */
@Service
class RacePriceService(
    private val eventRepository: EventRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val photoRepository: PhotoRepository,
    private val racePriceChangeRepository: RacePriceChangeRepository,
    private val accessGuard: PhotographerAccessGuard,
    private val raceProperties: RaceProperties,
    private val platformProperties: PlatformProperties,
    private val clock: Clock,
    private val decisions: AdminDecisionLogService,
) {
    // Caller holds photos -> coverage -> event and changes the event's floor after this call.
    @Transactional
    fun applyFloor(event: Event, newFloor: BigDecimal, adminId: UUID) {
        if (event.eventType == null) {
            photoRepository.updatePriceByEventId(event.id, newFloor)
            return
        }
        val ceiling = ceilingFor(newFloor)
        val now = OffsetDateTime.now(clock)
        for (row in eventPhotographerRepository.lockByEventIdForUpdate(event.id)) {
            val old = row.pricePhp ?: continue
            val clamped = old.coerceIn(newFloor, ceiling)
            row.pricePhp = clamped.takeIf { it.compareTo(newFloor) != 0 }
            if (old.compareTo(clamped) != 0) {
                racePriceChangeRepository.save(RacePriceChange(eventId = event.id, photographerId = row.id.photographerId,
                    changedBy = adminId, fromPhp = old, toPhp = clamped, reason = RacePriceReason.CLAMP, changedAt = now))
            }
        }
        // Flush the clamped coverage values before the native query reads them.
        eventPhotographerRepository.flush()
        photoRepository.updateRacePricesByEventId(event.id, newFloor)
    }

    @Transactional
    fun resetToFloor(adminId: UUID, eventId: UUID, photographerId: UUID): AdminRacePricesDto {
        adminRace(eventId)
        photoRepository.lockIdsByEventIdAndPhotographerIdForUpdate(eventId, photographerId)
        val row = eventPhotographerRepository.findByIdForUpdate(EventPhotographerId(eventId, photographerId))
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val floor = eventRepository.findPricePerPhoto(eventId)!!
        val from = row.pricePhp ?: floor
        row.pricePhp = null
        eventPhotographerRepository.flush()
        photoRepository.updatePriceByEventIdAndPhotographerId(eventId, photographerId, floor)
        racePriceChangeRepository.save(RacePriceChange(eventId = eventId, photographerId = photographerId,
            changedBy = adminId, fromPhp = from, toPhp = floor, reason = RacePriceReason.RESET, changedAt = OffsetDateTime.now(clock)))
        decisions.logEventDecision(adminId, eventId, "race_price_reset", meta = mapOf("photographerId" to photographerId.toString()))
        return listForAdmin(eventId)
    }

    @Transactional(readOnly = true)
    fun listForAdmin(eventId: UUID): AdminRacePricesDto {
        adminRace(eventId)
        val floor = eventRepository.findPricePerPhoto(eventId)!!
        return AdminRacePricesDto(floor, ceilingFor(floor), eventPhotographerRepository.findAdminPrices(eventId),
            eventPhotographerRepository.findPriceHistory(eventId, PageRequest.of(0, 50)).map {
                AdminRacePriceHistoryDto(it.photographerId, it.changedBy, it.fromPhp, it.toPhp, it.reason, it.changedAt)
            })
    }

    private fun adminRace(eventId: UUID): Event {
        val event = eventRepository.findById(eventId).orElse(null)?.takeIf { it.deletedAt == null }
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        if (event.eventType == null) throw ConflictException(code = ErrorCodes.RACE_PRICE_NOT_ALLOWED, message = "Prices are only listed for race events.")
        return event
    }

    @Transactional
    fun setMyPrice(photographerId: UUID, eventId: UUID, pricePhp: BigDecimal?): RacePriceDto {
        val event = eventRepository.findById(eventId).orElse(null)
        if (!raceProperties.selfServeEnabled || (event != null && !priceable(event))) {
            throw ConflictException(code = ErrorCodes.RACE_PRICE_NOT_ALLOWED,
                message = "Your own price can only be set on a live paid race event.")
        }
        val id = EventPhotographerId(eventId, photographerId)
        if (event == null || !eventPhotographerRepository.existsById(id)) {
            throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        }
        accessGuard.requireApproved(photographerId, "setting a race price")
        val requested = pricePhp?.setScale(2, RoundingMode.HALF_UP)

        photoRepository.lockIdsByEventIdAndPhotographerIdForUpdate(eventId, photographerId)
        val row = eventPhotographerRepository.findByIdForUpdate(id)
            ?: throw NotFoundException(code = ErrorCodes.EVENT_NOT_FOUND, message = "Event not found")
        val floor = eventRepository.findPricePerPhoto(eventId)!!.setScale(2, RoundingMode.HALF_UP)
        val ceiling = ceilingFor(floor)
        if (requested != null && (requested < floor || requested > ceiling)) {
            throw ValidationException(field = "pricePhp", message = "Choose a price from ₱${floor.toPlainString()} to ₱${ceiling.toPlainString()}.")
        }
        // The race price itself is stored as NULL, so a later race price change moves it too.
        val stored = requested?.takeIf { it.compareTo(floor) != 0 }
        val now = OffsetDateTime.now(clock)
        if (sameMoney(stored, row.pricePhp)) return dto(floor, row, now) // a repeat is not a second change

        nextChangeAt(row, now)?.let {
            throw ConflictException(code = ErrorCodes.RACE_PRICE_COOLDOWN,
                message = "You can change your price again after $it.")
        }
        val from = row.pricePhp ?: floor
        val effective = stored ?: floor
        row.pricePhp = stored
        row.priceChangedAt = now
        eventPhotographerRepository.save(row)
        photoRepository.updatePriceByEventIdAndPhotographerId(eventId, photographerId, effective)
        racePriceChangeRepository.save(RacePriceChange(eventId = eventId, photographerId = photographerId, changedBy = photographerId,
            fromPhp = from, toPhp = effective, reason = RacePriceReason.SET, changedAt = now))
        return dto(floor, row, now)
    }

    fun ceilingFor(floor: BigDecimal): BigDecimal =
        floor.multiply(platformProperties.racePriceCeilingMultiplier).setScale(2, RoundingMode.HALF_UP)

    /** The detail page's price card: null unless a race, a coverage row and the switch on. */
    fun toDto(event: Event, row: EventPhotographer?): RacePriceDto? {
        if (row == null || event.eventType == null || !raceProperties.selfServeEnabled) return null
        return dto(event.pricePerPhoto.setScale(2, RoundingMode.HALF_UP), row, OffsetDateTime.now(clock))
    }

    private fun dto(floor: BigDecimal, row: EventPhotographer, now: OffsetDateTime) =
        RacePriceDto(floor, ceilingFor(floor), row.pricePhp, row.pricePhp ?: floor, nextChangeAt(row, now))

    private fun nextChangeAt(row: EventPhotographer, now: OffsetDateTime): OffsetDateTime? =
        row.priceChangedAt?.plus(platformProperties.racePriceCooldown)?.takeIf { it.isAfter(now) }

    private fun priceable(event: Event) = event.eventType != null && !event.isFree && event.deletedAt == null &&
        event.status != EventStatus.ARCHIVED

    private fun sameMoney(a: BigDecimal?, b: BigDecimal?) = if (a == null || b == null) a == b else a.compareTo(b) == 0
}
