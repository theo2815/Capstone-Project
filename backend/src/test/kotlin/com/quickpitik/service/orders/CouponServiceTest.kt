package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.orders.CouponPreviewRequest
import com.quickpitik.dto.orders.UpsertCouponRequest
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerCoupon
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.OrderStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerCouponRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.math.RoundingMode
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// The one rule this feature must never break: a coupon comes out of the
// photographer's share, so QuickPitik's cut on a photo is identical with or
// without a code. Everything here pins that arithmetic and the guard rails
// around who may redeem what.
class CouponServiceTest {
    private val keep = BigDecimal("0.75")
    private val photographerId = UUID.randomUUID()
    private val otherPhotographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()

    private lateinit var couponRepository: PhotographerCouponRepository
    private lateinit var photoRepository: PhotoRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var eventPhotographerRepository: EventPhotographerRepository
    private lateinit var orderRepository: OrderRepository
    private lateinit var settingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var service: CouponService

    @BeforeEach
    fun setUp() {
        couponRepository = Mockito.mock(PhotographerCouponRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        eventPhotographerRepository = Mockito.mock(EventPhotographerRepository::class.java)
        orderRepository = Mockito.mock(OrderRepository::class.java)
        settingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        Mockito.`when`(couponRepository.saveAndFlush(anyArg())).thenAnswer { it.arguments[0] }
        Mockito.`when`(settingsRepository.findById(photographerId))
            .thenReturn(
                Optional.of(
                    PhotographerSettings(
                        userId = photographerId,
                        handle = "aira",
                        verificationStatus = VerificationStatus.APPROVED,
                    ),
                ),
            )
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(owner()))
        // Default fixture: an event this photographer created (no coverage row
        // needed) — the "covered" variants below override findById.
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(event()))
        service = CouponService(
            couponRepository,
            photoRepository,
            settingsRepository,
            userRepository,
            PlatformProperties(couponMaxPercent = 50),
            eventRepository,
            eventPhotographerRepository,
            orderRepository,
        )
    }

    @Test
    fun `discount is a percentage of the photographer share, rounded half up`() {
        assertEquals(BigDecimal("22.50"), couponDiscount(BigDecimal("150.00"), keep, 20))
        assertEquals(BigDecimal("9.38"), couponDiscount(BigDecimal("125.00"), keep, 10))
        assertEquals(BigDecimal("37.50"), couponDiscount(BigDecimal("99.99"), keep, 50))
    }

    @Test
    fun `platform fee is unchanged by any coupon percentage`() {
        for (price in listOf("20.00", "99.99", "125.00", "150.00", "333.33").map(::BigDecimal)) {
            val feeWithoutCoupon = price.subtract(price.multiply(keep).setScale(2, RoundingMode.HALF_UP))
            for (pct in 1..50) {
                val discount = couponDiscount(price, keep, pct)
                val charged = price.subtract(discount)
                val kept = price.multiply(keep).setScale(2, RoundingMode.HALF_UP).subtract(discount)
                assertEquals(feeWithoutCoupon, charged.subtract(kept), "price=$price pct=$pct")
                assertTrue(kept.signum() > 0, "photographer still earns something at $pct%")
            }
        }
    }

    @Test
    fun `upsert stores the code trimmed and uppercased`() {
        Mockito.`when`(couponRepository.existsByCodeAndEventIdIsNotNullAndIdNot(anyArg(), anyArg())).thenReturn(false)

        val dto = service.upsert(photographerId, eventId, UpsertCouponRequest(code = "  photo20 ", percentOff = 20))

        assertEquals("PHOTO20", dto.code)
        assertEquals(eventId, dto.eventId)
        assertEquals(20, dto.percentOff)
        assertTrue(dto.active)
    }

    @Test
    fun `upsert rejects codes outside 4 to 16 alphanumerics`() {
        val ex = assertFailsWith<ValidationException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest(code = "no spaces!", percentOff = 20))
        }
        assertEquals("code", ex.field)
    }

    @Test
    fun `upsert rejects a percentage above the configured cap`() {
        val ex = assertFailsWith<ValidationException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest(code = "PHOTO51", percentOff = 51))
        }
        assertEquals("percentOff", ex.field)
    }

    @Test
    fun `upsert refuses a code another photographer already owns`() {
        Mockito.`when`(couponRepository.existsByCodeAndEventIdIsNotNullAndIdNot(anyArg(), anyArg())).thenReturn(true)

        val ex = assertFailsWith<ConflictException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest(code = "PHOTO20", percentOff = 20))
        }
        assertEquals(ErrorCodes.COUPON_CODE_TAKEN, ex.code)
    }

    @Test
    fun `checkout resolution distinguishes unknown, inactive and expired codes`() {
        Mockito.`when`(couponRepository.findByCodeAndEventIdIsNotNull("NOPE")).thenReturn(null)
        Mockito.`when`(couponRepository.findByCodeAndEventIdIsNotNull("OFF"))
            .thenReturn(coupon(code = "OFF", active = false))
        Mockito.`when`(couponRepository.findByCodeAndEventIdIsNotNull("OLD"))
            .thenReturn(coupon(code = "OLD", expiresAt = OffsetDateTime.now().minusDays(1)))

        assertEquals(ErrorCodes.COUPON_INVALID, assertFailsWith<ValidationException> { service.resolveForCheckout("nope") }.code)
        assertEquals(ErrorCodes.COUPON_INVALID, assertFailsWith<ValidationException> { service.resolveForCheckout("off") }.code)
        assertEquals(ErrorCodes.COUPON_EXPIRED, assertFailsWith<ValidationException> { service.resolveForCheckout("old") }.code)
    }

    @Test
    fun `preview discounts only the owner's paid photos`() {
        val mine = photo(photographerId, "150.00")
        val theirs = photo(otherPhotographerId, "150.00")
        val free = photo(photographerId, "0.00")
        Mockito.`when`(couponRepository.findByCodeAndEventIdIsNotNull("PHOTO20")).thenReturn(coupon(code = "PHOTO20"))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(mine, theirs, free))

        val preview = service.preview(
            CouponPreviewRequest(code = "photo20", photoIds = listOf(mine.id, theirs.id, free.id)),
        )

        assertEquals(listOf(mine.id), preview.items.map { it.photoId })
        assertEquals(BigDecimal("22.50"), preview.items.single().discount)
        assertEquals(BigDecimal("22.50"), preview.discountTotal)
        assertEquals(1, preview.eligibleCount)
        assertEquals("aira", preview.photographerHandle)
        assertEquals("Aira Santos", preview.photographerName)
    }

    @Test
    fun `preview refuses a code that matches none of the photos`() {
        val theirs = photo(otherPhotographerId, "150.00")
        Mockito.`when`(couponRepository.findByCodeAndEventIdIsNotNull("PHOTO20")).thenReturn(coupon(code = "PHOTO20"))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(theirs))

        val ex = assertFailsWith<ValidationException> {
            service.preview(CouponPreviewRequest(code = "PHOTO20", photoIds = listOf(theirs.id)))
        }
        assertEquals(ErrorCodes.COUPON_NOT_APPLICABLE, ex.code)
    }

    @Test
    fun `same photographer photo from another event is not eligible`() {
        val otherEventPhoto = photo(photographerId, "150.00", UUID.randomUUID())
        Mockito.`when`(couponRepository.findByCodeAndEventIdIsNotNull("PHOTO20")).thenReturn(coupon(code = "PHOTO20"))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(otherEventPhoto))

        val ex = assertFailsWith<ValidationException> {
            service.preview(CouponPreviewRequest("PHOTO20", listOf(otherEventPhoto.id)))
        }

        assertEquals(ErrorCodes.COUPON_NOT_APPLICABLE, ex.code)
    }

    @Test
    fun `an admin event the photographer covers can receive a coupon`() {
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(event(createdBy = null)))
        Mockito.`when`(eventPhotographerRepository.existsById(EventPhotographerId(eventId, photographerId)))
            .thenReturn(true)
        Mockito.`when`(couponRepository.existsByCodeAndEventIdIsNotNullAndIdNot(anyArg(), anyArg())).thenReturn(false)

        val dto = service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20))

        assertEquals(eventId, dto.eventId)
        assertEquals("PHOTO20", dto.code)
    }

    @Test
    fun `another photographer's owned event is coverable once uploaded to`() {
        Mockito.`when`(eventRepository.findById(eventId))
            .thenReturn(Optional.of(event(createdBy = otherPhotographerId)))
        Mockito.`when`(eventPhotographerRepository.existsById(EventPhotographerId(eventId, photographerId)))
            .thenReturn(true)
        Mockito.`when`(couponRepository.existsByCodeAndEventIdIsNotNullAndIdNot(anyArg(), anyArg())).thenReturn(false)

        assertEquals(eventId, service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20)).eventId)
    }

    @Test
    fun `events the photographer never covered answer not found`() {
        val unknown = UUID.randomUUID()
        Mockito.`when`(eventRepository.findById(unknown)).thenReturn(Optional.empty())
        assertFailsWith<NotFoundException> {
            service.upsert(photographerId, unknown, UpsertCouponRequest("PHOTO20", 20))
        }

        // Admin event, no event_photographer row for this caller.
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(event(createdBy = null)))
        Mockito.`when`(eventPhotographerRepository.existsById(EventPhotographerId(eventId, photographerId)))
            .thenReturn(false)
        assertFailsWith<NotFoundException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20))
        }
        assertFailsWith<NotFoundException> { service.get(photographerId, eventId) }
        assertFailsWith<NotFoundException> { service.delete(photographerId, eventId) }

        // Someone else's owned event the caller never uploaded to.
        Mockito.`when`(eventRepository.findById(eventId))
            .thenReturn(Optional.of(event(createdBy = otherPhotographerId)))
        assertFailsWith<NotFoundException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20))
        }

        // Deleted events are gone even for their creator.
        Mockito.`when`(eventRepository.findById(eventId))
            .thenReturn(Optional.of(event().apply { deletedAt = OffsetDateTime.now() }))
        assertFailsWith<NotFoundException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20))
        }
    }

    @Test
    fun `free events cannot receive a coupon, covered or owned`() {
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(event(EventPricingMode.FREE)))
        val owned = assertFailsWith<ValidationException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20))
        }
        assertEquals(ErrorCodes.COUPON_NOT_APPLICABLE, owned.code)

        Mockito.`when`(eventRepository.findById(eventId))
            .thenReturn(Optional.of(event(EventPricingMode.FREE, createdBy = null)))
        Mockito.`when`(eventPhotographerRepository.existsById(EventPhotographerId(eventId, photographerId)))
            .thenReturn(true)
        val covered = assertFailsWith<ValidationException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20))
        }
        assertEquals(ErrorCodes.COUPON_NOT_APPLICABLE, covered.code)
    }

    @Test
    fun `usage limit rejects a new checkout after its reserved use`() {
        val limited = coupon(code = "ONCE", usageLimit = 1)
        Mockito.`when`(couponRepository.findScopedByCodeForUpdate("ONCE")).thenReturn(limited)
        Mockito.`when`(orderRepository.countUsesExcludingStatus(limited.id, OrderStatus.EXPIRED)).thenReturn(1)

        val ex = assertFailsWith<ValidationException> { service.reserveForCheckout("ONCE") }

        assertEquals(ErrorCodes.COUPON_USAGE_LIMIT_REACHED, ex.code)
    }

    // Auto-apply (2026-09-05): checkout asks for every live coupon of the
    // (event, photographer) pairs in the cart. Nothing here throws — a coupon
    // that can't be redeemed simply isn't applied.
    @Test
    fun `reserveAutoFor returns live coupons only for photographer-event pairs in the cart`() {
        val thirdPhotographerId = UUID.randomUUID()
        val absentPhotographerId = UUID.randomUUID()
        val a = coupon(code = "AAAA")
        val c = coupon(code = "CCCC", owner = thirdPhotographerId)
        val absent = coupon(code = "DDDD", owner = absentPhotographerId)
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(eventId))).thenReturn(listOf(a, c, absent))
        val cart = listOf(
            photo(photographerId, "125.00"),
            photo(photographerId, "125.00"),
            photo(otherPhotographerId, "150.00"),
            photo(thirdPhotographerId, "150.00"),
        )

        val resolved = service.reserveAutoFor(cart)

        assertEquals(
            mapOf((eventId to photographerId) to a, (eventId to thirdPhotographerId) to c),
            resolved,
        )
    }

    @Test
    fun `reserveAutoFor skips expired and exhausted coupons silently`() {
        val exhaustedOwner = UUID.randomUUID()
        val expired = coupon(code = "OLD", expiresAt = OffsetDateTime.now().minusDays(1))
        val exhausted = coupon(code = "ONCE", owner = exhaustedOwner, usageLimit = 1)
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(eventId)))
            .thenReturn(listOf(expired, exhausted))
        Mockito.`when`(orderRepository.countUsesExcludingStatus(exhausted.id, OrderStatus.EXPIRED)).thenReturn(1)

        val resolved = service.reserveAutoFor(
            listOf(photo(photographerId, "125.00"), photo(exhaustedOwner, "125.00")),
        )

        assertEquals(emptyMap(), resolved)
    }

    @Test
    fun `reserveAutoFor never carries a coupon across events`() {
        val otherEventId = UUID.randomUUID()
        Mockito.`when`(couponRepository.findActiveByEventIdInForUpdate(setOf(otherEventId))).thenReturn(emptyList())

        val resolved = service.reserveAutoFor(listOf(photo(photographerId, "125.00", otherEventId)))

        assertEquals(emptyMap(), resolved)
        Mockito.verify(couponRepository).findActiveByEventIdInForUpdate(setOf(otherEventId))
    }

    @Test
    fun `preview without a code prices every photographer's own live coupon`() {
        val thirdPhotographerId = UUID.randomUUID()
        val a1 = photo(photographerId, "125.00")
        val a2 = photo(photographerId, "125.00")
        val b1 = photo(otherPhotographerId, "150.00")
        val c1 = photo(thirdPhotographerId, "150.00")
        val a = coupon(code = "AAAA")
        val c = coupon(code = "CCCC", owner = thirdPhotographerId)
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(a1, a2, b1, c1))
        Mockito.`when`(couponRepository.findLiveForEvent(eqArg(eventId), anyArg(), anyArg())).thenReturn(listOf(a, c))

        val preview = service.preview(CouponPreviewRequest(code = null, photoIds = listOf(a1.id, a2.id, b1.id, c1.id)))

        assertEquals(null, preview.code)
        assertEquals(listOf(a1.id, a2.id, c1.id), preview.items.map { it.photoId })
        assertEquals(listOf("AAAA", "AAAA", "CCCC"), preview.items.map { it.couponCode })
        assertEquals(BigDecimal("18.75"), preview.items[0].discount)
        assertEquals(BigDecimal("22.50"), preview.items[2].discount)
        assertEquals(BigDecimal("60.00"), preview.discountTotal)
        assertEquals(3, preview.eligibleCount)
    }

    @Test
    fun `preview with a code keeps the other photographers' automatic coupons`() {
        val thirdPhotographerId = UUID.randomUUID()
        val a1 = photo(photographerId, "125.00")
        val c1 = photo(thirdPhotographerId, "150.00")
        val a = coupon(code = "AAAA")
        val c = coupon(code = "CCCC", owner = thirdPhotographerId)
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(a1, c1))
        Mockito.`when`(couponRepository.findByCodeAndEventIdIsNotNull("AAAA")).thenReturn(a)
        Mockito.`when`(couponRepository.findLiveForEvent(eqArg(eventId), anyArg(), anyArg())).thenReturn(listOf(a, c))

        val preview = service.preview(CouponPreviewRequest(code = "aaaa", photoIds = listOf(a1.id, c1.id)))

        assertEquals("AAAA", preview.code)
        assertEquals("Aira Santos", preview.photographerName)
        assertEquals(listOf("AAAA", "CCCC"), preview.items.map { it.couponCode })
        assertEquals(BigDecimal("41.25"), preview.discountTotal)
    }

    @Test
    fun `suspended owner cannot create or update a coupon`() {
        Mockito.`when`(userRepository.findById(photographerId))
            .thenReturn(Optional.of(owner().apply { suspendedAt = OffsetDateTime.now() }))

        val ex = assertFailsWith<ApiException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20))
        }

        assertEquals(ErrorCodes.ACCOUNT_SUSPENDED, ex.code)
    }

    @Test
    fun `unverified owner cannot create or update a coupon`() {
        Mockito.`when`(settingsRepository.findById(photographerId)).thenReturn(
            Optional.of(
                PhotographerSettings(
                    userId = photographerId,
                    handle = null,
                    verificationStatus = VerificationStatus.PENDING,
                ),
            ),
        )

        val ex = assertFailsWith<ApiException> {
            service.upsert(photographerId, eventId, UpsertCouponRequest("PHOTO20", 20))
        }

        assertEquals(ErrorCodes.PHOTOGRAPHER_NOT_VERIFIED, ex.code)
    }

    // Free giveaway (2026-09-05): exactly 100% is allowed only on a paid
    // event the photographer created. Covered admin events and other
    // photographers' events keep the cap, and 51–99 stay rejected everywhere.
    @Test
    fun `a 100 percent giveaway is allowed only on a paid event the photographer created`() {
        assertEquals(100, service.upsert(photographerId, eventId, UpsertCouponRequest("FREE100", 100)).percentOff)
        for (pct in listOf(51, 99, 101)) {
            val ex = assertFailsWith<ValidationException> {
                service.upsert(photographerId, eventId, UpsertCouponRequest("FREE100", pct))
            }
            assertEquals("percentOff", ex.field, "pct=$pct")
        }

        Mockito.`when`(eventPhotographerRepository.existsById(EventPhotographerId(eventId, photographerId)))
            .thenReturn(true)
        for (creator in listOf(null, otherPhotographerId)) {
            Mockito.`when`(eventRepository.findById(eventId)).thenReturn(Optional.of(event(createdBy = creator)))
            val ex = assertFailsWith<ValidationException> {
                service.upsert(photographerId, eventId, UpsertCouponRequest("FREE100", 100))
            }
            assertEquals("percentOff", ex.field, "creator=$creator")
        }
    }

    @Test
    fun `a 100 percent coupon zeroes the list price while lower percentages keep the share rule`() {
        val giveaway = PhotographerCoupon(eventId = eventId, photographerId = photographerId, code = "FREE100", percentOff = 100)
        val photo = photo(photographerId, "125.00")

        assertEquals(BigDecimal("125.00"), service.discountFor(photo, giveaway))
        assertEquals(0, service.quoteFor(photo, giveaway)!!.price.signum())
        assertEquals(BigDecimal("46.88"), service.discountFor(photo, coupon("HALF").apply { percentOff = 50 }))
    }

    private fun coupon(
        code: String,
        active: Boolean = true,
        expiresAt: OffsetDateTime? = null,
        usageLimit: Int? = null,
        owner: UUID = photographerId,
    ) = PhotographerCoupon(
        eventId = eventId,
        photographerId = owner,
        code = code,
        percentOff = 20,
        active = active,
        expiresAt = expiresAt,
        usageLimit = usageLimit,
    )

    private fun photo(owner: UUID, price: String, photoEventId: UUID = eventId) = Photo(
        eventId = photoEventId,
        s3Key = "photos/${UUID.randomUUID()}.jpg",
        pricePhp = BigDecimal(price),
    ).also {
        it.photographerId = owner
        it.status = PhotoStatus.LIVE
    }

    private fun event(mode: EventPricingMode = EventPricingMode.PAID, createdBy: UUID? = photographerId) = Event(
        id = eventId,
        slug = "event",
        name = "Event",
        date = LocalDate.of(2026, 9, 4),
        location = "Cebu",
        status = EventStatus.ACTIVE,
        createdBy = createdBy,
        pricingMode = mode,
    )

    private fun owner() = User(
        id = photographerId,
        name = "Aira Santos",
        email = "aira@example.com",
        passwordHash = "x",
        role = Role.PHOTOGRAPHER,
    )

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
