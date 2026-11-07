package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.SellerDetailsDto
import com.quickpitik.dto.photographer.SellerDetailsForm
import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerIdType
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import org.springframework.mock.web.MockMultipartFile
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.time.Clock
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.io.path.extension
import kotlin.io.path.readText
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SellerDetailsServiceTest {
    private lateinit var repository: SellerDetailsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var storage: StorageService
    private lateinit var txManager: PlatformTransactionManager
    private val clock = Clock.fixed(Instant.parse("2026-11-07T02:00:00Z"), ZoneOffset.UTC)
    private val userId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()
    // Platform-typed like PhotoUploadServiceTest: capture() returns null at runtime.
    private fun <T> capture(c: ArgumentCaptor<T>): T = c.capture()

    @BeforeEach
    fun setUp() {
        repository = Mockito.mock(SellerDetailsRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        storage = Mockito.mock(StorageService::class.java)
        txManager = Mockito.mock(PlatformTransactionManager::class.java).also {
            Mockito.`when`(it.getTransaction(anyArg())).thenReturn(SimpleTransactionStatus())
        }
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.empty())
        Mockito.`when`(repository.save(anyArg<SellerDetails>())).thenAnswer { it.getArgument(0) }
    }

    private fun service() = SellerDetailsService(repository, userRepository, storage, TransactionTemplate(txManager), clock)

    private fun jpeg(width: Int = 64, height: Int = 48): ByteArray = ByteArrayOutputStream().use {
        check(ImageIO.write(BufferedImage(width, height, BufferedImage.TYPE_INT_RGB), "jpg", it))
        it.toByteArray()
    }

    private fun part(name: String, bytes: ByteArray = jpeg(), type: String = "image/jpeg") =
        MockMultipartFile(name, "$name.jpg", type, bytes)

    private fun form(
        idType: String = "drivers_license",
        phone: String = "+63 917 123 4567",
        idFront: MockMultipartFile? = part("idFront"),
        idBack: MockMultipartFile? = null,
        selfie: MockMultipartFile? = part("selfie"),
        birForm: MockMultipartFile? = null,
    ) = SellerDetailsForm(
        legalName = "  Juan Dela Cruz  ",
        idType = idType,
        address = "12 Osmeña Blvd, Cebu City",
        phone = phone,
        idFront = idFront,
        idBack = idBack,
        selfie = selfie,
        birForm = birForm,
    )

    // StorageService is a Kotlin interface (non-null params), so read its recorded
    // calls instead of an ArgumentCaptor, whose null placeholder Kotlin rejects.
    private fun puts() = Mockito.mockingDetails(storage).invocations.filter { it.method.name == "put" }

    private fun storedKeys(): List<String> = puts().map { it.arguments[0] as String }

    private fun storedBytes(): List<ByteArray> = puts().map { it.arguments[1] as ByteArray }

    private fun saved(): SellerDetails {
        val row = ArgumentCaptor.forClass(SellerDetails::class.java)
        Mockito.verify(repository).save(capture(row))
        return row.value
    }

    private fun existing(status: SellerDetailsStatus = SellerDetailsStatus.REJECTED) = SellerDetails(
        userId = userId,
        legalName = "Old Name",
        idType = SellerIdType.PHILSYS,
        idFrontKey = "seller-details/$userId/old-front.jpg",
        idBackKey = "seller-details/$userId/old-back.jpg",
        selfieKey = "seller-details/$userId/old-selfie.jpg",
        birFormKey = "seller-details/$userId/old-bir.jpg",
        address = "Old address",
        phone = "09170000000",
        status = status,
        rejectionReason = "blurry",
        revision = 4,
        submittedAt = OffsetDateTime.parse("2026-11-01T00:00:00Z"),
        reviewedAt = OffsetDateTime.parse("2026-11-02T00:00:00Z"),
        reviewedBy = UUID.randomUUID(),
    )

    // 1
    @Test
    fun `submit stores upright JPEGs under seller-details with random names and saves SUBMITTED`() {
        val dto = service().submit(userId, form(idFront = part("idFront", jpeg(4000, 3000))))

        val keys = storedKeys()
        assertEquals(2, keys.size)
        keys.forEach {
            assertTrue(Regex("seller-details/$userId/[0-9a-f-]{36}\\.jpg").matches(it), it)
            assertFalse("Juan" in it || "Cruz" in it, it)
        }
        Mockito.verify(storage, Mockito.times(2)).put(anyArg<String>(), anyArg<ByteArray>(), Mockito.eq("image/jpeg") ?: "")
        val front = ImageIO.read(storedBytes().first().inputStream())
        assertTrue(maxOf(front.width, front.height) <= 2400, "${front.width}x${front.height}")
        storedBytes().forEach { assertFalse(hasExifSegment(it), "re-encoded JPEG must carry no EXIF") }
        val row = saved()
        assertEquals(SellerDetailsStatus.SUBMITTED, row.status)
        assertEquals("Juan Dela Cruz", row.legalName)
        assertEquals(1L, row.revision)
        assertEquals(OffsetDateTime.now(clock), row.submittedAt)
        Mockito.verify(userRepository).lockById(userId)
        assertEquals("submitted", dto.status)
    }

    /** A JPEG APP1 segment starting "Exif" is where cameras put GPS and other hidden data. */
    private fun hasExifSegment(jpeg: ByteArray): Boolean = (0 until jpeg.size - 6).any { i ->
        jpeg[i] == 0xFF.toByte() && jpeg[i + 1] == 0xE1.toByte() && String(jpeg, i + 4, 4, Charsets.US_ASCII) == "Exif"
    }

    @Test
    fun `an EXIF-tagged input comes out with no EXIF segment`() {
        // Input built by hand: a JPEG with an APP1 Exif segment spliced in after SOI.
        val plain = jpeg()
        val app1 = byteArrayOf(0xFF.toByte(), 0xE1.toByte(), 0x00, 0x08) + "Exif".toByteArray() + byteArrayOf(0, 0)
        val tagged = plain.copyOfRange(0, 2) + app1 + plain.copyOfRange(2, plain.size)
        assertTrue(hasExifSegment(tagged))

        service().submit(userId, form(idFront = part("idFront", tagged)))

        storedBytes().forEach { assertFalse(hasExifSegment(it)) }
    }

    // 2
    @Test
    fun `submit refuses a missing ID front`() {
        val error = assertFailsWith<ValidationException> { service().submit(userId, form(idFront = null)) }
        assertEquals(ErrorCodes.VALIDATION_ERROR, error.code)
        assertEquals("idFront", error.field)
        assertNothingStored()
    }

    @Test
    fun `submit refuses a missing selfie`() {
        val error = assertFailsWith<ValidationException> { service().submit(userId, form(selfie = null)) }
        assertEquals("selfie", error.field)
        assertNothingStored()
    }

    @Test
    fun `submit names each blank text field`() {
        for ((field, broken) in listOf(
            "legalName" to form().apply { legalName = "   " },
            "legalName" to form().apply { legalName = "x".repeat(201) },
            "idType" to form(idType = "other"),
            "address" to form().apply { address = null },
            "address" to form().apply { address = "x".repeat(501) },
            "phone" to form().apply { phone = null },
        )) {
            val error = assertFailsWith<ValidationException>(field) { service().submit(userId, broken) }
            assertEquals(field, error.field)
        }
        assertNothingStored()
    }

    // 3
    @Test
    fun `a PhilSys ID without a back is refused, other ID types pass without one`() {
        val error = assertFailsWith<ValidationException> { service().submit(userId, form(idType = "philsys")) }
        assertEquals("idBack", error.field)
        assertNothingStored()

        for (type in SellerIdType.entries.filter { it != SellerIdType.PHILSYS }) {
            service().submit(userId, form(idType = type.toWire()))
        }
        service().submit(userId, form(idType = "philsys", idBack = part("idBack")))
    }

    // 4
    @Test
    fun `phone rule`() {
        for (good in listOf("+63 917 123 4567", "0917-123-4567", "(02) 8123 4567", "0917.123.4567")) {
            assertEquals(good, SellerDetailsService.validPhone(good), good)
        }
        for (bad in listOf("917 123 4567", "+1 555 123 4567", "0917", "09171234567890", "0917abc4567", "", "+63", "0917" + " ".repeat(20) + "1234567")) {
            assertNull(SellerDetailsService.validPhone(bad), bad)
        }
        val error = assertFailsWith<ValidationException> { service().submit(userId, form(phone = "917 123 4567")) }
        assertEquals("phone", error.field)
    }

    // 5
    @Test
    fun `a wrong type, a file over 5 MB, an undecodable image and an over-80-megapixel image are refused, and nothing is stored`() {
        val pdf = assertFailsWith<ValidationException> {
            service().submit(userId, form(selfie = part("selfie", jpeg(), "application/pdf")))
        }
        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, pdf.code)
        assertEquals("selfie", pdf.field)

        val big = assertFailsWith<ApiException> {
            service().submit(userId, form(idFront = part("idFront", ByteArray(5 * 1024 * 1024 + 1))))
        }
        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, big.status)
        assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, big.code)
        assertEquals("idFront", big.field)

        val garbage = assertFailsWith<ValidationException> {
            service().submit(userId, form(idFront = part("idFront", "not an image".toByteArray())))
        }
        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, garbage.code)
        assertEquals("idFront", garbage.field)

        // A tiny PNG whose header claims 10000 x 10000 pixels (100 MP).
        val huge = ByteArrayOutputStream().use { ImageIO.write(BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB), "png", it); it.toByteArray() }
        java.nio.ByteBuffer.wrap(huge, 16, 8).putInt(10_000).putInt(10_000)
        val pixels = assertFailsWith<ValidationException> {
            service().submit(userId, form(idFront = part("idFront", huge, "image/png")))
        }
        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, pixels.code)
        assertEquals("idFront", pixels.field)

        assertNothingStored()
    }

    private fun assertNothingStored() {
        Mockito.verify(storage, Mockito.never()).put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>())
        Mockito.verify(repository, Mockito.never()).save(anyArg<SellerDetails>())
    }

    // 6
    @Test
    fun `a resubmission replaces fields and files, bumps revision, clears a rejection, and deletes old keys only after commit`() {
        val old = existing()
        val oldKeys = old.fileKeys()
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))

        service().submit(userId, form())

        val row = saved()
        assertEquals(5L, row.revision)
        assertEquals(SellerDetailsStatus.SUBMITTED, row.status)
        assertEquals(SellerIdType.DRIVERS_LICENSE, row.idType)
        assertNull(row.rejectionReason)
        assertNull(row.reviewedAt)
        assertNull(row.reviewedBy)
        assertNull(row.idBackKey, "a resubmission without an optional file means none")
        assertNull(row.birFormKey)
        assertEquals(storedKeys().toSet(), row.fileKeys().toSet())
        val order = Mockito.inOrder(repository, txManager, storage)
        order.verify(repository).save(anyArg<SellerDetails>())
        order.verify(txManager).commit(anyArg())
        oldKeys.forEach { order.verify(storage).delete(it) }
        storedKeys().forEach { Mockito.verify(storage, Mockito.never()).delete(it) }
    }

    @Test
    fun `a rollback deletes only the new keys`() {
        val old = existing()
        val oldKeys = old.fileKeys()
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))
        Mockito.`when`(repository.save(anyArg<SellerDetails>())).thenThrow(IllegalStateException("db down"))

        assertFailsWith<IllegalStateException> { service().submit(userId, form()) }

        val created = storedKeys()
        assertEquals(2, created.size)
        created.forEach { Mockito.verify(storage).delete(it) }
        oldKeys.forEach { Mockito.verify(storage, Mockito.never()).delete(it) }
    }

    @Test
    fun `a failed old-file delete does not fail the submission`() {
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(existing()))
        Mockito.`when`(storage.delete(anyArg())).thenThrow(IllegalStateException("r2 down"))

        assertEquals("submitted", service().submit(userId, form()).status)
    }

    // 7
    @Test
    fun `get answers missing with nulls, and the DTO has no link or key fields`() {
        assertEquals(SellerDetailsDto.MISSING, service().get(userId))
        assertEquals("missing", SellerDetailsDto.MISSING.status)
        assertNull(SellerDetailsDto.MISSING.legalName)

        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(existing(SellerDetailsStatus.APPROVED)))
        val dto = service().get(userId)
        assertEquals("approved", dto.status)
        assertEquals("philsys", dto.idType)
        assertEquals(true, dto.hasIdBack)

        val names = SellerDetailsDto::class.java.declaredFields.map { it.name.lowercase() }
        assertTrue(names.none { "key" in it || "url" in it || "link" in it }, names.toString())
        assertFalse("seller-details/" in dto.toString())
    }

    // 8
    @Test
    fun `the seller-details prefix appears only in SellerDetailsService and SecurityConfig, and the service imports nothing from service ai`() {
        val main = Path.of("src/main/kotlin")
        val mentions = Files.walk(main).use { paths ->
            paths.filter { it.extension == "kt" }.toList().filter { file ->
                file.readText().lines().any { "seller-details" in it && !it.trim().startsWith("@") }
            }.map { it.fileName.toString() }
        }
        assertEquals(setOf("SellerDetailsService.kt", "SecurityConfig.kt"), mentions.toSet())
        val service = main.resolve("com/quickpitik/service/photographer/SellerDetailsService.kt").readText()
        assertFalse("com.quickpitik.service.ai" in service)
    }
}
