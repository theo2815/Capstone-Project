package com.quickpitik.service.photographer

import com.quickpitik.config.PlatformProperties
import com.quickpitik.config.PublicProperties
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoBundleRow
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.time.LocalDate
import java.util.Optional
import java.util.UUID
import java.util.zip.ZipInputStream
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

// The "Download All" ZIP: prepare() resolves the spec, writeTo() streams it,
// and a signed token (real, not mocked) gates the public stream route.
class PhotographerBundleServiceTest {

    private lateinit var photoRepository: PhotoRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var storageService: StorageService
    private lateinit var tokenService: PhotographerBundleTokenService
    private lateinit var service: PhotographerBundleService

    private val photographerId = UUID.randomUUID()
    private val eventId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        photoRepository = Mockito.mock(PhotoRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        tokenService = PhotographerBundleTokenService(PlatformProperties())
        service = PhotographerBundleService(
            photoRepository, eventRepository, storageService, tokenService, PublicProperties(),
        )
    }

    private fun stubEvent(slug: String = "fun-run") {
        Mockito.`when`(eventRepository.findById(eventId)).thenReturn(
            Optional.of(
                Event(
                    id = eventId,
                    slug = slug,
                    name = "Fun Run",
                    date = LocalDate.of(2026, 9, 1),
                    location = "Cebu City",
                    status = EventStatus.ACTIVE,
                ),
            ),
        )
    }

    private fun row(s3Key: String, bib: String?, id: UUID = UUID.randomUUID()): PhotoBundleRow =
        object : PhotoBundleRow {
            override val id = id
            override val s3Key = s3Key
            override val bib = bib
        }

    private fun stubRows(vararg rows: PhotoBundleRow) {
        Mockito.`when`(photoRepository.findActiveBundleRows(eventId, photographerId)).thenReturn(rows.toList())
    }

    @Test
    fun `prepare de-duplicates colliding entry names and zips multiple photos`() {
        stubEvent()
        stubRows(row("k1", "101"), row("k2", "101")) // same bib → colliding names

        val spec = service.prepare(photographerId, eventId)

        assertEquals(2, spec.entries.size)
        assertEquals("application/zip", spec.contentType)
        assertTrue(spec.filename.endsWith(".zip"))
        assertNotEquals(spec.entries[0].entryName, spec.entries[1].entryName)
    }

    @Test
    fun `prepare collapses a single photo to a raw image`() {
        stubEvent()
        stubRows(row("k1", "101"))

        val spec = service.prepare(photographerId, eventId)

        assertEquals(1, spec.entries.size)
        assertEquals("image/jpeg", spec.contentType)
        assertEquals(spec.entries[0].entryName, spec.filename)
        assertTrue(spec.filename.endsWith(".jpg"))
    }

    @Test
    fun `prepare 404s when there is nothing active to download`() {
        stubEvent()
        stubRows()

        assertFailsWith<NotFoundException> { service.prepare(photographerId, eventId) }
    }

    @Test
    fun `prepareDownloadUrl mints a token that verifies for this event only`() {
        stubEvent()
        Mockito.`when`(photoRepository.countByEventIdAndPhotographerIdAndDeletedAtIsNull(eventId, photographerId))
            .thenReturn(3L)

        val dto = service.prepareDownloadUrl(photographerId, eventId)

        assertTrue(dto.url.startsWith("http://localhost:8080/api/v1/photographer-bundles/$eventId?token="))
        val token = dto.url.substringAfter("token=")
        assertEquals(photographerId, tokenService.verify(token, eventId))
        assertNull(tokenService.verify(token, UUID.randomUUID())) // wrong event → no grant
    }

    @Test
    fun `prepareDownloadUrl 404s when the event has no active photos`() {
        stubEvent()
        Mockito.`when`(photoRepository.countByEventIdAndPhotographerIdAndDeletedAtIsNull(eventId, photographerId))
            .thenReturn(0L)

        assertFailsWith<NotFoundException> { service.prepareDownloadUrl(photographerId, eventId) }
    }

    @Test
    fun `writeTo streams a zip whose entries match the spec`() {
        Mockito.`when`(storageService.open(anyArg())).thenAnswer { ByteArrayInputStream(byteArrayOf(1, 2, 3)) }
        val spec = PhotographerBundleService.BundleSpec(
            filename = "quickpitik-fun-run-abcd1234.zip",
            contentType = "application/zip",
            entries = listOf(
                PhotographerBundleService.BundleEntryRef("k1", "a.jpg"),
                PhotographerBundleService.BundleEntryRef("k2", "b.jpg"),
            ),
        )

        val out = ByteArrayOutputStream()
        service.writeTo(spec, out)

        val names = mutableListOf<String>()
        ZipInputStream(ByteArrayInputStream(out.toByteArray())).use { zip ->
            var e = zip.nextEntry
            while (e != null) {
                names += e.name
                e = zip.nextEntry
            }
        }
        assertEquals(listOf("a.jpg", "b.jpg"), names)
    }

    @Test
    fun `writeTo passes a single photo through as raw bytes`() {
        Mockito.`when`(storageService.open("k1")).thenReturn(ByteArrayInputStream(byteArrayOf(9, 8, 7)))
        val spec = PhotographerBundleService.BundleSpec(
            filename = "quickpitik-fun-run-bib-101.jpg",
            contentType = "image/jpeg",
            entries = listOf(PhotographerBundleService.BundleEntryRef("k1", "quickpitik-fun-run-bib-101.jpg")),
        )

        val out = ByteArrayOutputStream()
        service.writeTo(spec, out)

        assertTrue(out.toByteArray().contentEquals(byteArrayOf(9, 8, 7)))
    }

    @Test
    fun `token verify rejects a tampered token`() {
        val grant = tokenService.issue(photographerId, eventId)
        val tampered = grant.token.dropLast(1) + if (grant.token.last() == 'A') 'B' else 'A'
        assertNull(tokenService.verify(tampered, eventId))
    }

    @Test
    fun `failed second original aborts instead of finalizing a partial archive`() {
        Mockito.`when`(storageService.open("k1")).thenReturn(ByteArrayInputStream(byteArrayOf(1, 2, 3)))
        Mockito.`when`(storageService.open("k2")).thenAnswer { throw java.io.IOException("storage unavailable") }
        val spec = PhotographerBundleService.BundleSpec("photos.zip", "application/zip", listOf(
            PhotographerBundleService.BundleEntryRef("k1", "a.jpg"),
            PhotographerBundleService.BundleEntryRef("k2", "b.jpg"),
        ))
        val out = ByteArrayOutputStream()
        assertFailsWith<java.io.IOException> { service.writeTo(spec, out) }
        // No ZIP end-of-central-directory record means consumers cannot mistake
        // the successfully copied first photo for the complete event archive.
        assertTrue(out.toByteArray().toList().windowed(4).none { it == listOf<Byte>(0x50, 0x4b, 0x05, 0x06) })
    }
}
