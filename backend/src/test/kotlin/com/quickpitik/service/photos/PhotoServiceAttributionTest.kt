package com.quickpitik.service.photos

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotographerCoupon
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerCouponRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.orders.CouponService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// Cross-module reconciliation (2026-08-15). PhotoDto carried no photographer
// attribution, which left mobile's runner-side photographer discovery blocked
// outright — a runner could see a photo but had no way to reach the
// photographer's public profile at /{handle}.
//
// The two things worth pinning here are the null-handle contract (an
// unverified photographer has no handle, so clients must not render a link)
// and the batch shape: attribution must not reintroduce an N+1 on a surface
// that renders 240 photos a page.
class PhotoServiceAttributionTest {

    private lateinit var photoRepository: PhotoRepository
    private lateinit var downloadGrantRepository: DownloadGrantRepository
    private lateinit var photographerSettingsRepository: PhotographerSettingsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var storageService: StorageService
    private lateinit var couponRepository: PhotographerCouponRepository
    private lateinit var eventRepository: EventRepository

    private val eventId: UUID = UUID.randomUUID()
    private val pagination = PaginationParams.of(0, 240)

    @BeforeEach
    fun setUp() {
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        downloadGrantRepository = Mockito.mock(DownloadGrantRepository::class.java)
        photographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        couponRepository = Mockito.mock(PhotographerCouponRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")
    }

    // Free events (V46): the preview is unmarked and the original is anyone's
    // to download, so the clean + download URLs are minted for every visitor —
    // grant or no grant, signed in or not.
    @Test
    fun `a free event hands every visitor the clean and download URLs`() {
        stubEvent(EventPricingMode.FREE)
        stubPage(listOf(photo(UUID.randomUUID(), price = "0.00")))
        Mockito.`when`(storageService.presignedDownloadUrl(anyArg(), anyArg(), anyArg())).thenReturn("https://dl")

        val dto = service().listForEvent(eventId, bib = null, pagination = pagination).items.single()

        assertTrue(dto.free)
        assertEquals(0, dto.price.signum())
        assertEquals("https://thumb", dto.cleanUrl)
        assertEquals("https://dl", dto.downloadUrl)
        assertNull(dto.couponCode)
        Mockito.verify(downloadGrantRepository, Mockito.never()).findOwnedPhotoIdsByUserAndPhotoIds(anyArg(), anyArg())
    }

    @Test
    fun `a paid event keeps the original behind a grant`() {
        stubEvent(EventPricingMode.PAID)
        stubPage(listOf(photo(UUID.randomUUID())))

        val dto = service().listForEvent(eventId, bib = null, pagination = pagination).items.single()

        assertFalse(dto.free)
        assertNull(dto.cleanUrl)
        assertNull(dto.downloadUrl)
        Mockito.verify(storageService, Mockito.never()).presignedDownloadUrl(anyArg(), anyArg(), anyArg())
    }

    private fun stubEvent(mode: EventPricingMode) {
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(
            Optional.of(
                Event(
                    id = eventId,
                    slug = "e-$eventId",
                    name = "Fun Run",
                    date = LocalDate.of(2026, 9, 1),
                    location = "Cebu City",
                    status = EventStatus.ACTIVE,
                    pricingMode = mode,
                ),
            ),
        )
    }

    // Photographer coupons (V45): the card shows the code and the price with
    // it, computed server-side so no client ever does money math.
    @Test
    fun `a photo carries its photographer's live coupon and the discounted price`() {
        val id = UUID.randomUUID()
        stubPage(listOf(photo(id)))
        stubPhotographers(listOf(settings(id, handle = "cebu-shots")), listOf(user(id, name = "Cebu Shots")))
        Mockito.`when`(couponRepository.findLiveForEvent(eqArg(eventId), anyArg(), anyArg())).thenReturn(
            listOf(PhotographerCoupon(eventId = eventId, photographerId = id, code = "SHOTS10", percentOff = 10)),
        )

        val dto = service().listForEvent(eventId, bib = null, pagination = pagination).items.single()

        assertEquals("SHOTS10", dto.couponCode)
        assertEquals(10, dto.couponPercentOff)
        // ₱199 × 0.75 × 10% = ₱14.93 off the photographer's share.
        assertEquals(BigDecimal("184.07"), dto.couponPrice)
    }

    @Test
    fun `a switched-off coupon and a free photo carry no coupon fields`() {
        val paused = UUID.randomUUID()
        val generous = UUID.randomUUID()
        stubPage(listOf(photo(paused), photo(generous, price = "0.00")))
        stubPhotographers(
            listOf(settings(paused, handle = "paused"), settings(generous, handle = "generous")),
            listOf(user(paused, name = "Paused"), user(generous, name = "Generous")),
        )
        Mockito.`when`(couponRepository.findLiveForEvent(eqArg(eventId), anyArg(), anyArg())).thenReturn(
            listOf(
                PhotographerCoupon(eventId = eventId, photographerId = generous, code = "FREEBIE", percentOff = 10),
            ),
        )

        val items = service().listForEvent(eventId, bib = null, pagination = pagination).items

        assertEquals(2, items.size)
        items.forEach {
            assertNull(it.couponCode, "photo ${it.id}")
            assertNull(it.couponPrice, "photo ${it.id}")
        }
    }

    @Test
    fun `a photo carries its photographer's handle and name`() {
        val id = UUID.randomUUID()
        stubPage(listOf(photo(id)))
        stubPhotographers(listOf(settings(id, handle = "cebu-shots")), listOf(user(id, name = "Cebu Shots")))

        val dto = service().listForEvent(eventId, bib = null, pagination = pagination).items.single()

        assertEquals("cebu-shots", dto.photographerHandle)
        assertEquals("Cebu Shots", dto.photographerName)
    }

    @Test
    fun `an unverified photographer has no handle but still has a name`() {
        // PhotographerSettings.handle is only assigned at verification. A null
        // handle is the "not linkable" signal — the name still renders, as
        // plain text rather than a link to /{null}.
        val id = UUID.randomUUID()
        stubPage(listOf(photo(id)))
        stubPhotographers(listOf(settings(id, handle = null)), listOf(user(id, name = "Unverified Pro")))

        val dto = service().listForEvent(eventId, bib = null, pagination = pagination).items.single()

        assertNull(dto.photographerHandle)
        assertEquals("Unverified Pro", dto.photographerName)
    }

    @Test
    fun `a full page of photos costs one lookup per repository, not one per photo`() {
        // The N+1 guard. Three photographers covering one event, 30 photos —
        // resolution must collapse to a single IN query each, the same shape
        // resolveOwnedIds already uses for ownership.
        val ids = List(3) { UUID.randomUUID() }
        val photos = (0 until 30).map { photo(ids[it % 3]) }
        stubPage(photos)
        stubPhotographers(
            ids.mapIndexed { i, id -> settings(id, handle = "shooter-$i") },
            ids.mapIndexed { i, id -> user(id, name = "Shooter $i") },
        )

        val items = service().listForEvent(eventId, bib = null, pagination = pagination).items

        assertEquals(30, items.size)
        assertEquals(setOf("shooter-0", "shooter-1", "shooter-2"), items.mapNotNull { it.photographerHandle }.toSet())
        Mockito.verify(photographerSettingsRepository, Mockito.times(1)).findAllById(anyArg())
        Mockito.verify(userRepository, Mockito.times(1)).findAllById(anyArg())
    }

    // 2026-08-27 perf pass: plain browsing must take the join-free fast path
    // (no bibs LEFT JOIN, no DISTINCT); only an actual bib filter pays for
    // searchForEvent.
    @Test
    fun `no bib routes to the fast path, a bib routes to the search query`() {
        stubPage(emptyList())

        service().listForEvent(eventId, bib = null, pagination = pagination)
        Mockito.verify(photoRepository).findForEventNoBib(anyArg(), anyArg(), Mockito.anyLong(), anyArg())
        Mockito.verify(photoRepository, Mockito.never())
            .searchForEvent(anyArg(), anyArg(), anyArg(), anyArg())

        service().listForEvent(eventId, bib = "183", pagination = pagination)
        Mockito.verify(photoRepository).searchForEvent(anyArg(), anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `a photo with no photographer resolves to no attribution instead of failing`() {
        // photographerId is nullable on Photo. Such a row simply has no
        // attribution; it must not blow up the whole page.
        stubPage(listOf(photo(photographerId = null)))

        val dto = service().listForEvent(eventId, bib = null, pagination = pagination).items.single()

        assertNull(dto.photographerHandle)
        assertNull(dto.photographerName)
        Mockito.verify(photographerSettingsRepository, Mockito.never()).findAllById(anyArg())
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun service() = PhotoService(
        photoRepository,
        downloadGrantRepository,
        photographerSettingsRepository,
        userRepository,
        storageService,
        StorageProperties(),
        CouponService(
            couponRepository,
            photoRepository,
            photographerSettingsRepository,
            userRepository,
            PlatformProperties(),
            eventRepository,
            Mockito.mock(EventPhotographerRepository::class.java),
            Mockito.mock(OrderRepository::class.java),
        ),
        eventRepository,
    )

    private fun stubPage(photos: List<Photo>) {
        // listForEvent with no bib takes the join-free fast path; the bib
        // variant keeps searchForEvent. Stub both so either route serves.
        Mockito.`when`(photoRepository.findForEventNoBib(anyArg(), anyArg(), Mockito.anyLong(), anyArg()))
            .thenReturn(PageImpl(photos))
        Mockito.`when`(photoRepository.searchForEvent(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenReturn(PageImpl(photos))
    }

    private fun stubPhotographers(settings: List<PhotographerSettings>, users: List<User>) {
        Mockito.`when`(photographerSettingsRepository.findAllById(anyArg())).thenReturn(settings)
        Mockito.`when`(userRepository.findAllById(anyArg())).thenReturn(users)
    }

    private fun photo(photographerId: UUID?, price: String = "199.00"): Photo = Photo(
        eventId = eventId,
        photographerId = photographerId,
        s3Key = "events/$eventId/photos/x/original.jpg",
        thumbnailS3Key = "events/$eventId/photos/x/watermark.jpg",
        pricePhp = BigDecimal(price),
    )

    private fun settings(id: UUID, handle: String?) =
        PhotographerSettings(userId = id, handle = handle)

    private fun user(id: UUID, name: String) = User(
        id = id,
        email = "$id@test.local",
        passwordHash = "x",
        name = name,
        role = Role.PHOTOGRAPHER,
    )

    private fun <T> anyArg(): T = Mockito.any()
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value
}
