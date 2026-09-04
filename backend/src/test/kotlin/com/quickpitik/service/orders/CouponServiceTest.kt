package com.quickpitik.service.orders

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.PlatformProperties
import com.quickpitik.dto.orders.CouponPreviewRequest
import com.quickpitik.dto.orders.UpsertCouponRequest
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.PhotographerCoupon
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerCouponRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.math.RoundingMode
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
    private lateinit var service: CouponService

    @BeforeEach
    fun setUp() {
        couponRepository = Mockito.mock(PhotographerCouponRepository::class.java)
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        val settings = Mockito.mock(PhotographerSettingsRepository::class.java)
        val users = Mockito.mock(UserRepository::class.java)
        Mockito.`when`(couponRepository.save(anyArg())).thenAnswer { it.arguments[0] }
        Mockito.`when`(settings.findById(photographerId))
            .thenReturn(Optional.of(PhotographerSettings(userId = photographerId, handle = "aira")))
        Mockito.`when`(users.findById(photographerId)).thenReturn(
            Optional.of(
                User(
                    id = photographerId,
                    name = "Aira Santos",
                    email = "aira@example.com",
                    passwordHash = "x",
                    role = Role.PHOTOGRAPHER,
                ),
            ),
        )
        service = CouponService(
            couponRepository,
            photoRepository,
            settings,
            users,
            PlatformProperties(couponMaxPercent = 50),
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
        Mockito.`when`(couponRepository.existsByCodeAndPhotographerIdNot("PHOTO20", photographerId)).thenReturn(false)

        val dto = service.upsert(photographerId, UpsertCouponRequest(code = "  photo20 ", percentOff = 20))

        assertEquals("PHOTO20", dto.code)
        assertEquals(20, dto.percentOff)
        assertTrue(dto.active)
    }

    @Test
    fun `upsert rejects codes outside 4 to 16 alphanumerics`() {
        val ex = assertFailsWith<ValidationException> {
            service.upsert(photographerId, UpsertCouponRequest(code = "no spaces!", percentOff = 20))
        }
        assertEquals("code", ex.field)
    }

    @Test
    fun `upsert rejects a percentage above the configured cap`() {
        val ex = assertFailsWith<ValidationException> {
            service.upsert(photographerId, UpsertCouponRequest(code = "PHOTO51", percentOff = 51))
        }
        assertEquals("percentOff", ex.field)
    }

    @Test
    fun `upsert refuses a code another photographer already owns`() {
        Mockito.`when`(couponRepository.existsByCodeAndPhotographerIdNot("PHOTO20", photographerId)).thenReturn(true)

        val ex = assertFailsWith<ConflictException> {
            service.upsert(photographerId, UpsertCouponRequest(code = "PHOTO20", percentOff = 20))
        }
        assertEquals(ErrorCodes.COUPON_CODE_TAKEN, ex.code)
    }

    @Test
    fun `checkout resolution distinguishes unknown, inactive and expired codes`() {
        Mockito.`when`(couponRepository.findByCode("NOPE")).thenReturn(null)
        Mockito.`when`(couponRepository.findByCode("OFF")).thenReturn(coupon(code = "OFF", active = false))
        Mockito.`when`(couponRepository.findByCode("OLD"))
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
        Mockito.`when`(couponRepository.findByCode("PHOTO20")).thenReturn(coupon(code = "PHOTO20"))
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
        Mockito.`when`(couponRepository.findByCode("PHOTO20")).thenReturn(coupon(code = "PHOTO20"))
        Mockito.`when`(photoRepository.findAllById(anyArg<Iterable<UUID>>())).thenReturn(listOf(theirs))

        val ex = assertFailsWith<ValidationException> {
            service.preview(CouponPreviewRequest(code = "PHOTO20", photoIds = listOf(theirs.id)))
        }
        assertEquals(ErrorCodes.COUPON_NOT_APPLICABLE, ex.code)
    }

    private fun coupon(
        code: String,
        active: Boolean = true,
        expiresAt: OffsetDateTime? = null,
    ) = PhotographerCoupon(
        photographerId = photographerId,
        code = code,
        percentOff = 20,
        active = active,
        expiresAt = expiresAt,
    )

    private fun photo(owner: UUID, price: String) = Photo(
        eventId = eventId,
        s3Key = "photos/${UUID.randomUUID()}.jpg",
        pricePhp = BigDecimal(price),
    ).also {
        it.photographerId = owner
        it.status = PhotoStatus.LIVE
    }

    private fun <T> anyArg(): T = Mockito.any()
}
