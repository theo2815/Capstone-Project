package com.quickpitik.service.events

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.RaceProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographer
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.RacePriceChange
import com.quickpitik.entity.RacePriceReason
import com.quickpitik.entity.RaceType
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RacePriceChangeRepository
import com.quickpitik.service.photographer.PhotographerAccessGuard
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import java.math.BigDecimal
import java.time.Clock
import java.time.Instant
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RacePriceServiceTest {
    private val me = UUID.randomUUID()
    private val now = Instant.parse("2026-11-09T04:00:00Z")
    private val clock = Clock.fixed(now, ZoneOffset.UTC)

    private lateinit var events: EventRepository
    private lateinit var coverage: EventPhotographerRepository
    private lateinit var photos: PhotoRepository
    private lateinit var history: RacePriceChangeRepository
    private lateinit var guard: PhotographerAccessGuard
    private lateinit var race: Event
    private lateinit var row: EventPhotographer

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T { Mockito.eq(value); return value }

    @BeforeEach
    fun setUp() {
        events = Mockito.mock(EventRepository::class.java)
        coverage = Mockito.mock(EventPhotographerRepository::class.java)
        photos = Mockito.mock(PhotoRepository::class.java)
        history = Mockito.mock(RacePriceChangeRepository::class.java)
        guard = Mockito.mock(PhotographerAccessGuard::class.java)
        race = Event(slug = "cebu-city-marathon", name = "Cebu City Marathon", date = LocalDate.of(2026, 12, 6),
            location = "Cebu City", status = EventStatus.ACTIVE, pricePerPhoto = BigDecimal("125.00"))
            .apply { eventType = RaceType.ROAD_RACE }
        row = EventPhotographer(id = EventPhotographerId(race.id, me))
        Mockito.`when`(events.findById(race.id)).thenReturn(Optional.of(race))
        Mockito.`when`(events.findPricePerPhoto(race.id)).thenReturn(BigDecimal("125.00"))
        Mockito.`when`(coverage.existsById(EventPhotographerId(race.id, me))).thenReturn(true)
        Mockito.`when`(coverage.findByIdForUpdate(EventPhotographerId(race.id, me))).thenReturn(row)
        Mockito.`when`(guard.requireApproved(eqArg(me), Mockito.anyString())).thenReturn(PhotographerSettings(userId = me))
    }

    private fun service(selfServe: Boolean = true) =
        RacePriceService(events, coverage, photos, history, guard, RaceProperties(selfServeEnabled = selfServe), PlatformProperties(), clock)

    @Test
    fun `refuses a price above twice the race price`() {
        val error = assertFailsWith<ValidationException> { service().setMyPrice(me, race.id, BigDecimal("250.01")) }

        assertEquals("pricePhp", error.field)
        assertTrue("125.00" in error.message!! && "250.00" in error.message!!, error.message)
        Mockito.verify(photos, Mockito.never()).updatePriceByEventIdAndPhotographerId(anyArg(), anyArg(), anyArg())
        assertEquals(BigDecimal("250.00"), service().setMyPrice(me, race.id, BigDecimal("250")).effectivePhp)
    }

    @Test
    fun `refuses a price below the race price`() {
        assertEquals("pricePhp", assertFailsWith<ValidationException> { service().setMyPrice(me, race.id, BigDecimal("124.99")) }.field)
    }

    @Test
    fun `stores the race price as null`() {
        row.pricePhp = BigDecimal("200.00")
        row.priceChangedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC).minusDays(2)

        val dto = service().setMyPrice(me, race.id, BigDecimal("125"))

        assertNull(row.pricePhp)
        assertNull(dto.myPricePhp)
        assertEquals(BigDecimal("125.00"), dto.effectivePhp)
        Mockito.verify(photos).updatePriceByEventIdAndPhotographerId(race.id, me, BigDecimal("125.00"))
    }

    @Test
    fun `refuses a second change inside the cooldown`() {
        row.pricePhp = BigDecimal("150.00")
        row.priceChangedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC).minusHours(1)

        val error = assertFailsWith<ConflictException> { service().setMyPrice(me, race.id, BigDecimal("200")) }

        assertEquals(ErrorCodes.RACE_PRICE_COOLDOWN, error.code)
        assertTrue("2026-11-10T03:00" in error.message!!, error.message) // when it opens again (24 h after the change)
        Mockito.verify(photos, Mockito.never()).updatePriceByEventIdAndPhotographerId(anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `the same price again is a no-op, not a cooldown`() {
        row.pricePhp = BigDecimal("150.00")
        row.priceChangedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC).minusHours(1)

        assertEquals(BigDecimal("150.00"), service().setMyPrice(me, race.id, BigDecimal("150")).myPricePhp)
        Mockito.verifyNoInteractions(history)
    }

    @Test
    fun `refuses a non-race event`() {
        for (event in listOf(
            race.apply { eventType = null },
        )) {
            val error = assertFailsWith<ConflictException> { service().setMyPrice(me, event.id, BigDecimal("150")) }
            assertEquals(ErrorCodes.RACE_PRICE_NOT_ALLOWED, error.code)
        }
    }

    @Test
    fun `refuses a free, archived or deleted race`() {
        for (change in listOf<(Event) -> Unit>(
            { it.pricingMode = EventPricingMode.FREE },
            { it.status = EventStatus.ARCHIVED },
            { it.deletedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC) },
        )) {
            setUp()
            change(race)
            val error = assertFailsWith<ConflictException> { service().setMyPrice(me, race.id, BigDecimal("150")) }
            assertEquals(ErrorCodes.RACE_PRICE_NOT_ALLOWED, error.code)
        }
    }

    @Test
    fun `answers not found without a coverage row`() {
        Mockito.`when`(coverage.existsById(EventPhotographerId(race.id, me))).thenReturn(false)

        val error = assertFailsWith<NotFoundException> { service().setMyPrice(me, race.id, BigDecimal("150")) }

        assertEquals(ErrorCodes.EVENT_NOT_FOUND, error.code)
        Mockito.verifyNoInteractions(guard)
    }

    @Test
    fun `re-prices only this photographer's photos in this race, photos locked before the row`() {
        service().setMyPrice(me, race.id, BigDecimal("200"))

        val order = Mockito.inOrder(photos, coverage, events)
        order.verify(photos).lockIdsByEventIdAndPhotographerIdForUpdate(race.id, me)
        order.verify(coverage).findByIdForUpdate(EventPhotographerId(race.id, me))
        order.verify(events).findPricePerPhoto(race.id)
        Mockito.verify(photos).updatePriceByEventIdAndPhotographerId(race.id, me, BigDecimal("200.00"))
        Mockito.verify(photos, Mockito.never()).updatePriceByEventId(anyArg(), anyArg())
        assertEquals(BigDecimal("200.00"), row.pricePhp)
        assertEquals(OffsetDateTime.ofInstant(now, ZoneOffset.UTC), row.priceChangedAt)
    }

    @Test
    fun `writes a SET history row`() {
        service().setMyPrice(me, race.id, BigDecimal("200"))

        val saved = ArgumentCaptor.forClass(RacePriceChange::class.java)
        Mockito.verify(history).save(saved.capture())
        val change = saved.value
        assertEquals(race.id, change.eventId)
        assertEquals(me, change.photographerId)
        assertEquals(me, change.changedBy)
        assertEquals(BigDecimal("125.00"), change.fromPhp)
        assertEquals(BigDecimal("200.00"), change.toPhp)
        assertEquals(RacePriceReason.SET, change.reason)
    }

    @Test
    fun `refuses when self-serve races are off`() {
        val error = assertFailsWith<ConflictException> { service(selfServe = false).setMyPrice(me, race.id, BigDecimal("150")) }

        assertEquals(ErrorCodes.RACE_PRICE_NOT_ALLOWED, error.code)
        Mockito.verifyNoInteractions(photos, history)
    }

    @Test
    fun `racePrice is null unless a race, a coverage row and the switch`() {
        assertNotNull(service().toDto(race, row))
        assertNull(service().toDto(race, null))
        assertNull(service(selfServe = false).toDto(race, row))
        assertNull(service().toDto(race.apply { eventType = null }, row))
    }

    @Test
    fun `the price card shows the ceiling and when a change opens again`() {
        row.pricePhp = BigDecimal("180.00")
        row.priceChangedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC).minusHours(1)

        val dto = service().toDto(race, row)!!

        assertEquals(BigDecimal("125.00"), dto.floorPhp)
        assertEquals(BigDecimal("250.00"), dto.ceilingPhp)
        assertEquals(BigDecimal("180.00"), dto.effectivePhp)
        assertEquals(OffsetDateTime.ofInstant(now, ZoneOffset.UTC).plusHours(23), dto.nextChangeAt)
        row.priceChangedAt = OffsetDateTime.ofInstant(now, ZoneOffset.UTC).minusDays(2)
        assertNull(service().toDto(race, row)!!.nextChangeAt, "null = can change now")
    }
}
