package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.SellerDetailsDto
import com.quickpitik.dto.photographer.SellerDetailsForm
import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerIdType
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.SellerDetailsRejectRequest
import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import jakarta.validation.Validation
import org.springframework.context.ApplicationEventPublisher
import org.springframework.data.domain.PageImpl
import java.time.Duration
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
    private lateinit var decisionRepository: AdminDecisionLogRepository
    private lateinit var messageRepository: PhotographerMessageRepository
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
        decisionRepository = Mockito.mock(AdminDecisionLogRepository::class.java)
        Mockito.`when`(decisionRepository.save(anyArg<AdminDecisionLog>())).thenAnswer { it.getArgument(0) }
        messageRepository = Mockito.mock(PhotographerMessageRepository::class.java)
        Mockito.`when`(messageRepository.save(anyArg<PhotographerMessage>())).thenAnswer { it.getArgument(0) }
    }

    // The real decision-log service over mocked repositories, so tests read the rows it would write.
    private fun service() = SellerDetailsService(
        repository, userRepository, storage, TransactionTemplate(txManager), clock,
        AdminDecisionLogService(decisionRepository, messageRepository, Mockito.mock(ApplicationEventPublisher::class.java)),
        StorageProperties(),
    )

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

    // ─── SC-1b: admin review ─────────────────────────────────────────────

    private val adminId = UUID.randomUUID()

    private fun withRecord(status: SellerDetailsStatus = SellerDetailsStatus.SUBMITTED, birForm: Boolean = true): SellerDetails {
        val row = existing(status).apply {
            if (status != SellerDetailsStatus.REJECTED) rejectionReason = null
            if (!birForm) birFormKey = null
        }
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(row))
        Mockito.`when`(userRepository.findById(userId)).thenReturn(
            Optional.of(User(id = userId, email = "seller@test.local", passwordHash = "unused", name = "S", role = Role.PHOTOGRAPHER)),
        )
        // doAnswer: re-stubbing with when() would run the previous answer with null arguments.
        Mockito.doAnswer {
            "https://signed.example/${it.getArgument<String>(0)}?ttl=${it.getArgument<Duration>(1).seconds}&name=${it.getArgument<String>(2)}"
        }.`when`(storage).presignedDownloadUrl(anyArg(), anyArg(), anyArg())
        return row
    }

    private fun decisions(): List<AdminDecisionLog> = Mockito.mockingDetails(decisionRepository).invocations
        .filter { it.method.name == "save" }.map { it.arguments[0] as AdminDecisionLog }

    private fun messages(): List<PhotographerMessage> = Mockito.mockingDetails(messageRepository).invocations
        .filter { it.method.name == "save" }.map { it.arguments[0] as PhotographerMessage }

    // 1
    @Test
    fun `the admin view makes a 5-minute download link for each present file and writes one seller_details_viewed row naming the files, with no link or key in reason or meta`() {
        val row = withRecord(birForm = false)

        val dto = service().adminView(adminId, userId)

        val files = checkNotNull(dto.files)
        listOf(files.idFront, files.idBack, files.selfie).forEach { assertTrue("ttl=300&" in it!!, it) }
        assertEquals("https://signed.example/${row.idFrontKey}?ttl=300&name=id_front.jpg", files.idFront)
        assertEquals("https://signed.example/${row.selfieKey}?ttl=300&name=selfie.jpg", files.selfie)
        assertNull(files.birForm)
        assertEquals(OffsetDateTime.now(clock).plusMinutes(5), dto.linksExpireAt)
        assertEquals("seller@test.local", dto.email)
        assertEquals(4L, dto.revision)

        val logged = decisions().single()
        assertEquals("seller_details_viewed", logged.decision)
        assertEquals(adminId, logged.adminId)
        assertEquals(userId, logged.targetUserId)
        assertNull(logged.reason)
        assertEquals(mapOf("files" to listOf("id_front", "id_back", "selfie"), "revision" to 4L), logged.meta)
        assertFalse("seller-details/" in logged.meta.toString() || "http" in logged.meta.toString())
        assertTrue(messages().isEmpty())
    }

    @Test
    fun `the admin view of a photographer with no record answers 404 and logs nothing`() {
        assertFailsWith<NotFoundException> { service().adminView(adminId, userId) }
        assertTrue(decisions().isEmpty())
    }

    // 2
    @Test
    fun `the admin list has no links, and puts submitted first, oldest first`() {
        val row = withRecord()
        Mockito.`when`(repository.findForAdmin(Mockito.eq(SellerDetailsStatus.SUBMITTED) ?: SellerDetailsStatus.SUBMITTED, anyArg()))
            .thenReturn(PageImpl(listOf(row)))
        val user = userRepository.findById(userId).get()
        Mockito.`when`(userRepository.findAllById(listOf(userId))).thenReturn(listOf(user))

        val page = service().adminList("submitted", PaginationParams.of(0, 20))

        assertEquals(1L, page.total)
        val item = page.items.single()
        assertEquals("seller@test.local", item.email)
        assertEquals("submitted", item.status)
        assertEquals("philsys", item.idType)
        assertFalse("seller-details/" in page.toString() || "http" in page.toString())
        Mockito.verify(storage, Mockito.never()).presignedDownloadUrl(anyArg(), anyArg(), anyArg())
        assertTrue(decisions().isEmpty())
        val names = com.quickpitik.dto.admin.AdminSellerDetailsRowDto::class.java.declaredFields.map { it.name.lowercase() }
        assertTrue(names.none { "key" in it || "url" in it || "link" in it || "file" in it }, names.toString())
    }

    @Test
    fun `the admin list refuses an unknown status filter`() {
        val error = assertFailsWith<ValidationException> { service().adminList("missing", PaginationParams.of(0, 20)) }
        assertEquals("status", error.field)
    }

    // 3
    @Test
    fun `approve needs SUBMITTED and the current revision, a stale revision answers 409 and changes nothing`() {
        val row = withRecord()
        val stale = assertFailsWith<ApiException> { service().approve(adminId, userId, 3) }
        assertEquals(HttpStatus.CONFLICT, stale.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, stale.code)
        assertEquals("Seller details have changed. Reload before deciding.", stale.message)
        assertEquals(SellerDetailsStatus.SUBMITTED, row.status)
        Mockito.verify(repository, Mockito.never()).save(anyArg<SellerDetails>())
        assertTrue(decisions().isEmpty())

        val dto = service().approve(adminId, userId, 4)

        Mockito.verify(userRepository, Mockito.times(2)).lockById(userId) // the stale try and this one
        assertEquals("approved", dto.status)
        assertNull(dto.files)
        assertEquals(SellerDetailsStatus.APPROVED, row.status)
        assertEquals(adminId, row.reviewedBy)
        assertEquals(OffsetDateTime.now(clock), row.reviewedAt)
        assertEquals(4L, row.revision)
        val logged = decisions().single()
        assertEquals("seller_details_approved", logged.decision)
        assertEquals(mapOf("revision" to 4L), logged.meta)
        assertTrue(messages().isEmpty())

        for (status in listOf(SellerDetailsStatus.APPROVED, SellerDetailsStatus.REJECTED)) {
            withRecord(status)
            val wrong = assertFailsWith<ApiException> { service().approve(adminId, userId, 4) }
            assertEquals(HttpStatus.CONFLICT, wrong.status)
        }
    }

    // 4
    @Test
    fun `reject works from SUBMITTED or APPROVED, needs a reason and the current revision, stores the reason, and sends an ADMIN_MESSAGE with it`() {
        for (from in listOf(SellerDetailsStatus.SUBMITTED, SellerDetailsStatus.APPROVED)) {
            Mockito.clearInvocations(decisionRepository, messageRepository)
            val row = withRecord(from)

            val dto = service().reject(adminId, userId, 4, "  The selfie does not show the ID.  ")

            assertEquals("rejected", dto.status)
            assertEquals(SellerDetailsStatus.REJECTED, row.status)
            assertEquals("The selfie does not show the ID.", row.rejectionReason)
            assertEquals(adminId, row.reviewedBy)
            val logged = decisions().single()
            assertEquals("seller_details_rejected", logged.decision)
            assertEquals("The selfie does not show the ID.", logged.reason)
            assertEquals(mapOf("revision" to 4L), logged.meta)
            val message = messages().single()
            assertEquals(userId, message.photographerId)
            assertEquals(PhotographerMessageKind.ADMIN_MESSAGE.wire, message.kindWire)
            assertEquals("Seller details not accepted", message.title)
            assertEquals(
                "Your seller details were not accepted. Reason: The selfie does not show the ID. New uploads are paused " +
                    "until you send them again in your photographer settings on quickpitik.com.",
                message.body,
            )
            assertEquals(adminId, message.sourceAdminId)
            assertEquals(logged.id, message.sourceDecisionId)
        }

        withRecord(SellerDetailsStatus.REJECTED)
        assertEquals(HttpStatus.CONFLICT, assertFailsWith<ApiException> { service().reject(adminId, userId, 4, "again") }.status)
        withRecord()
        assertEquals(HttpStatus.CONFLICT, assertFailsWith<ApiException> { service().reject(adminId, userId, 5, "stale") }.status)

        val validator = Validation.buildDefaultValidatorFactory().validator
        assertFalse(validator.validate(SellerDetailsRejectRequest(revision = 4, reason = "   ")).isEmpty())
        assertFalse(validator.validate(SellerDetailsRejectRequest(revision = 4, reason = "x".repeat(501))).isEmpty())
        assertFalse(validator.validate(SellerDetailsRejectRequest(revision = null, reason = "ok")).isEmpty())
        assertTrue(validator.validate(SellerDetailsRejectRequest(revision = 4, reason = "ok")).isEmpty())
    }

    // 5
    @Test
    fun `after a reject the upload gate refuses, after a resubmission it passes`() {
        val row = withRecord()
        service().reject(adminId, userId, 4, "blurry")
        assertTrue(row.status == SellerDetailsStatus.REJECTED, "the gate (existsByUserIdAndStatusNot REJECTED) refuses this row")

        service().submit(userId, form())

        assertEquals(SellerDetailsStatus.SUBMITTED, row.status, "the gate passes a SUBMITTED row")
        assertEquals(5L, row.revision)
        assertNull(row.rejectionReason)
    }
}
