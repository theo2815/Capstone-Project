package com.quickpitik.service.photos

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.PublicProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPricingMode
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotographerCoupon
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.DownloadGrantRepository
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.OrderRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.PhotographerCouponRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotographerAccessGuard
import com.quickpitik.service.photographer.PhotoMetadataService
import com.quickpitik.service.orders.CouponService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.math.BigDecimal
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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
    private lateinit var photoMetadataService: PhotoMetadataService

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
        photoMetadataService = Mockito.mock(PhotoMetadataService::class.java)
        Mockito.`when`(photoMetadataService.metadataAccess(anyArg(), anyArg(), anyArg())).thenAnswer { invocation ->
            invocation.getArgument<Collection<Photo>>(1).associate { it.id to "none" }
        }
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://thumb")
    }

    // Free events (V46): the preview is unmarked and the original is anyone's
    // to download, so the clean + download URLs are minted for every visitor —
    // grant or no grant, signed in or not.
    //
    // The download URL is a backend route, not a presigned R2 link (2026-09-05):
    // Meta's in-app browsers append `fbclid=` to the navigation, which breaks
    // a SigV4 signature but is ignored by our own controller.
    @Test
    fun `a free event hands every visitor the clean and download URLs`() {
        stubEvent(EventPricingMode.FREE)
        val photo = photo(UUID.randomUUID(), price = "0.00")
        stubPage(listOf(photo))

        val dto = service().listForEvent(eventId, bib = null, pagination = pagination).items.single()

        assertTrue(dto.free)
        assertEquals(0, dto.price.signum())
        assertEquals("https://thumb", dto.cleanUrl)
        assertEquals("https://api.test/api/v1/events/$eventId/photos/${photo.id}/download", dto.downloadUrl)
        assertNull(dto.couponCode)
        Mockito.verify(storageService, Mockito.never()).presignedDownloadUrl(anyArg(), anyArg(), anyArg())
        Mockito.verify(downloadGrantRepository, Mockito.never()).findOwnedPhotoIdsByUserAndPhotoIds(anyArg(), anyArg())
    }

    @ParameterizedTest
    @EnumSource(EventStatus::class, names = ["ACTIVE", "COMPLETED"])
    fun `freeDownload resolves the original key and filename for a live photo of a free event`(status: EventStatus) {
        stubEvent(EventPricingMode.FREE, status)
        val photo = photo(UUID.randomUUID(), price = "0.00")
        Mockito.`when`(photoRepository.findById(photo.id)).thenReturn(Optional.of(photo))

        val download = service().freeDownload(eventId, photo.id)

        assertEquals(photo.s3Key, download.s3Key)
        assertEquals(PhotoFilenames.downloadFilenameOf(photo), download.filename)
    }

    @Test
    fun `freeDownload refuses a photo of a paid event`() {
        stubEvent(EventPricingMode.PAID)
        val photo = photo(UUID.randomUUID())
        Mockito.`when`(photoRepository.findById(photo.id)).thenReturn(Optional.of(photo))

        assertFailsWith<NotFoundException> { service().freeDownload(eventId, photo.id) }
    }

    @Test
    fun `freeDownload refuses a photo that belongs to another event`() {
        stubEvent(EventPricingMode.FREE)
        val stray = Photo(eventId = UUID.randomUUID(), s3Key = "elsewhere.jpg", pricePhp = BigDecimal.ZERO)
        Mockito.`when`(photoRepository.findById(stray.id)).thenReturn(Optional.of(stray))

        assertFailsWith<NotFoundException> { service().freeDownload(eventId, stray.id) }
    }

    @Test
    fun `free stream checks current public eligibility before opening the original`() {
        val photoId = UUID.randomUUID()
        val download = PhotoService.FreeDownload("original.jpg", "photo.jpg")
        val bytes = byteArrayOf(1, 2, 3)
        Mockito.`when`(photoRepository.isFreeDeliveryAvailable(eventId, photoId)).thenReturn(true)
        Mockito.`when`(storageService.open(download.s3Key)).thenReturn(ByteArrayInputStream(bytes))
        val out = ByteArrayOutputStream()

        service().writeFreeDownload(eventId, photoId, download, out)

        assertTrue(bytes.contentEquals(out.toByteArray()))
        val sequence = Mockito.inOrder(photoRepository, storageService)
        sequence.verify(photoRepository).isFreeDeliveryAvailable(eventId, photoId)
        sequence.verify(storageService).open(download.s3Key)
    }

    @Test
    fun `free stream refuses current ineligibility without opening storage`() {
        val photoId = UUID.randomUUID()
        Mockito.`when`(photoRepository.isFreeDeliveryAvailable(eventId, photoId)).thenReturn(false)
        val out = ByteArrayOutputStream()

        val failure = assertFailsWith<NotFoundException> {
            service().writeFreeDownload(eventId, photoId, PhotoService.FreeDownload("original.jpg", "photo.jpg"), out)
        }

        assertEquals(ErrorCodes.PHOTO_NOT_FOUND, failure.code)
        assertEquals(0, out.size())
        Mockito.verifyNoInteractions(storageService)
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

    private fun stubEvent(mode: EventPricingMode, status: EventStatus = EventStatus.ACTIVE) {
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(
            Optional.of(
                Event(
                    id = eventId,
                    slug = "e-$eventId",
                    name = "Fun Run",
                    date = LocalDate.of(2026, 9, 1),
                    location = "Cebu City",
                    status = status,
                    pricingMode = mode,
                ),
            ),
        )
    }

    @ParameterizedTest
    @EnumSource(EventStatus::class, names = ["DRAFT", "ARCHIVED"])
    fun `freeDownload refuses a draft or archived free event`(status: EventStatus) {
        stubEvent(EventPricingMode.FREE, status)
        val photo = photo(UUID.randomUUID(), price = "0.00")
        Mockito.`when`(photoRepository.findById(photo.id)).thenReturn(Optional.of(photo))

        val ex = assertFailsWith<NotFoundException> { service().freeDownload(eventId, photo.id) }

        assertEquals(ErrorCodes.PHOTO_NOT_FOUND, ex.code)
        Mockito.verifyNoInteractions(storageService)
    }

    @Test
    fun `bib search strips wildcards and punctuation before the substring query`() {
        stubPage(emptyList())

        service().listForEvent(eventId, bib = "b- 18%_3\\a", pagination = pagination)

        Mockito.verify(photoRepository).searchForEvent(anyArg(), anyArg(), eqArg("183A"), anyArg())
        Mockito.verify(photoRepository, Mockito.never())
            .findForEventNoBib(anyArg(), anyArg(), Mockito.anyLong(), anyArg())
    }

    @Test
    fun `public thumbnails use derivatives or null and never the original`() {
        val photos = listOf(
            photo(null).also { it.thumbnailS3Key = null },
            photo(null).also { it.thumbnailS3Key = null; it.watermarkS3Key = "watermark.jpg" },
            photo(null).also { it.thumbnailS3Key = "thumbnail.jpg"; it.watermarkS3Key = "watermark.jpg" },
            photo(null).also { it.thumbnailS3Key = "watermark.jpg"; it.watermarkS3Key = "watermark.jpg" },
        )
        stubPage(photos)
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg()))
            .thenAnswer { "https://storage/${it.getArgument<String>(0)}" }

        val items = service().listForEvent(eventId, bib = null, pagination = pagination).items

        // imageUrl stays the 1280 preview (watermark key) even when a grid tile exists (FS-9).
        assertEquals(listOf(null, "https://storage/watermark.jpg", "https://storage/watermark.jpg", "https://storage/watermark.jpg"),
            items.map { it.imageUrl })
        // gridUrl only when a separate tile exists; older photos (equal keys) leave it null.
        assertEquals(listOf(null, null, "https://storage/thumbnail.jpg", null), items.map { it.gridUrl })
        assertTrue(items.all { it.cleanUrl == null && it.downloadUrl == null })
        Mockito.verify(storageService, Mockito.never()).presignedGetUrl(eqArg(photos.first().s3Key), anyArg())
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
    fun `a photo credits the photographer's public brand name`() {
        val id = UUID.randomUUID()
        stubPage(listOf(photo(id)))
        stubPhotographers(
            listOf(settings(id, handle = "cebu-shots").also { it.brandName = "Finish Line Studio" }),
            listOf(user(id, name = "Cebu Shots")),
        )

        val dto = service().listForEvent(eventId, bib = null, pagination = pagination).items.single()

        assertEquals("Finish Line Studio", dto.photographerName)
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
        // The N+1 guard. Three photographers covering one event, 100 photos —
        // resolution must collapse to a single IN query each, the same shape
        // resolveOwnedIds already uses for ownership.
        val ids = List(3) { UUID.randomUUID() }
        val photos = (0 until 100).map { photo(ids[it % 3]) }
        stubPage(photos)
        stubPhotographers(
            ids.mapIndexed { i, id -> settings(id, handle = "shooter-$i") },
            ids.mapIndexed { i, id -> user(id, name = "Shooter $i") },
        )

        val items = service().listForEvent(eventId, bib = null, pagination = pagination).items

        assertEquals(100, items.size)
        assertEquals(setOf("shooter-0", "shooter-1", "shooter-2"), items.mapNotNull { it.photographerHandle }.toSet())
        Mockito.verify(photographerSettingsRepository, Mockito.times(1)).findAllById(anyArg())
        Mockito.verify(userRepository, Mockito.times(1)).findAllById(anyArg())
        Mockito.verify(photoMetadataService, Mockito.times(1)).metadataAccess(anyArg(), anyArg(), anyArg())
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
            PhotographerAccessGuard(userRepository, photographerSettingsRepository),
        ),
        eventRepository,
        // Trailing slash on purpose — the builder must not emit `//events`.
        PublicProperties(apiBaseUrl = "https://api.test/api/v1/"),
        photoMetadataService,
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
