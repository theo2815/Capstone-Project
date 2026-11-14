package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.admin.toAdminDto
import com.quickpitik.dto.photographer.SellerDetailsDto
import com.quickpitik.dto.photographer.toDto
import com.quickpitik.entity.SellerReasonCode
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
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.PayoutMethod
import com.quickpitik.entity.RepeatMatch
import com.quickpitik.entity.RepeatMatchKind
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.IdentityProperties
import com.quickpitik.dto.photographer.LiveCheckRequest
import com.quickpitik.entity.LivenessPurpose
import com.quickpitik.entity.LivenessSession
import com.quickpitik.service.ai.FaceCompareResult
import com.quickpitik.service.ai.LivenessResult
import com.quickpitik.service.ai.RekognitionIdentityClient
import com.quickpitik.service.liveness.LivenessSessionService
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.ReservedHandleRepository
import com.quickpitik.repository.SocialLinkRepository
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.moderation.ImageModerationService
import com.quickpitik.service.moderation.WordListChecker
import com.quickpitik.service.reference.RegionsService
import org.springframework.beans.factory.ObjectProvider
import software.amazon.awssdk.core.exception.SdkClientException
import software.amazon.awssdk.services.rekognition.model.LivenessSessionStatus
import java.math.BigDecimal
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
import org.springframework.transaction.TransactionDefinition
import org.springframework.transaction.TransactionSystemException
import org.springframework.transaction.support.AbstractPlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionStatus
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
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SellerDetailsServiceTest {
    /** A real Spring manager, so transaction synchronization runs; [commitFailure] makes the commit itself fail. */
    class FakeTxManager : AbstractPlatformTransactionManager() {
        var commitFailure: RuntimeException? = null
        override fun doGetTransaction(): Any = Any()
        override fun doBegin(transaction: Any, definition: TransactionDefinition) = Unit
        override fun doCommit(status: DefaultTransactionStatus) { commitFailure?.let { throw it } }
        override fun doRollback(status: DefaultTransactionStatus) = Unit
    }

    private lateinit var repository: SellerDetailsRepository
    private lateinit var userRepository: UserRepository
    private lateinit var storage: StorageService
    private lateinit var txManager: FakeTxManager
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
        txManager = Mockito.spy(FakeTxManager()) // spied so tests can check the commit order
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.empty())
        Mockito.`when`(repository.save(anyArg<SellerDetails>())).thenAnswer { it.getArgument(0) }
        decisionRepository = Mockito.mock(AdminDecisionLogRepository::class.java)
        Mockito.`when`(decisionRepository.save(anyArg<AdminDecisionLog>())).thenAnswer { it.getArgument(0) }
        messageRepository = Mockito.mock(PhotographerMessageRepository::class.java)
        Mockito.`when`(messageRepository.save(anyArg<PhotographerMessage>())).thenAnswer { it.getArgument(0) }
        sessions = Mockito.mock(LivenessSessionService::class.java)
        identityClient = Mockito.mock(RekognitionIdentityClient::class.java)
        Mockito.`when`(sessions.requireOpen(userId, SESSION, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null))
            .thenReturn(LivenessSession(SESSION, userId, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null, "1.2", consentAt))
        Mockito.`when`(sessions.claim(userId, SESSION, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null)).thenReturn(true)
        Mockito.`when`(identityClient.livenessResult(SESSION)).thenReturn(LivenessResult(LivenessSessionStatus.SUCCEEDED, 99.5f, reference))
        Mockito.`when`(identityClient.compareFaces(anyArg(), anyArg())).thenReturn(FaceCompareResult(1, 99.8f))
        Mockito.`when`(identityClient.detectLines(anyArg())).thenReturn(listOf("REPUBLIC OF THE PHILIPPINES", "DELA CRUZ, JUAN ANDRES", ID_NUMBER))
    }

    private lateinit var sessions: LivenessSessionService
    private lateinit var identityClient: RekognitionIdentityClient
    private val consentAt = OffsetDateTime.parse("2026-11-07T01:58:00Z")
    private val reference = byteArrayOf(9, 9, 9)

    @Suppress("UNCHECKED_CAST")
    private fun identityChecks(aiEnabled: Boolean = true): IdentityCheckService {
        val provider = Mockito.mock(ObjectProvider::class.java) as ObjectProvider<RekognitionIdentityClient>
        Mockito.`when`(provider.getIfAvailable()).thenReturn(identityClient)
        return IdentityCheckService(sessions, provider, AiApiProperties(enabled = aiEnabled), identityFormats)
    }

    // A set format for every type: with none, every live check would go to an admin (R-1).
    private val identityFormats = IdentityProperties(idNumberFormats = SellerIdType.entries.associateWith { "[A-Z0-9]{4,30}" })
    private fun matchHashes(normalised: String) = SellerIdType.entries.map { identityChecks().idNumberHmac(it, normalised) }

    // The real decision-log and identity-check services over mocks, so tests read what they would write.
    private fun service(
        identity: IdentityProperties = IdentityProperties(),
        profiles: PhotographerSettingsService = Mockito.mock(PhotographerSettingsService::class.java),
        aiEnabled: Boolean = true,
    ) = SellerDetailsService(
        repository, userRepository, storage, TransactionTemplate(txManager), clock,
        decisionLog(),
        StorageProperties(), identityChecks(aiEnabled), payoutRepository, identity, profiles,
    )

    private fun decisionLog() =
        AdminDecisionLogService(decisionRepository, messageRepository, Mockito.mock(ApplicationEventPublisher::class.java))

    // ─── AA-I7: a passed identity approves a waiting, complete profile ───

    private val settingsRepository: PhotographerSettingsRepository = Mockito.mock(PhotographerSettingsRepository::class.java)

    /** The real profile service over mocks, reading this test's seller-details [repository]. */
    private fun profiles() = PhotographerSettingsService(
        settingsRepository, userRepository, Mockito.mock(ReservedHandleRepository::class.java),
        Mockito.mock(RegionsService::class.java), storage, StorageProperties(),
        Mockito.mock(SocialLinkRepository::class.java).also { Mockito.`when`(it.countByUserId(userId)).thenReturn(1L) },
        payoutRepository, Mockito.mock(ApplicationEventPublisher::class.java), WordListChecker(),
        Mockito.mock(ImageModerationService::class.java), Mockito.mock(SystemFlagService::class.java),
        TransactionTemplate(txManager), repository, decisionLog(),
    )

    /** A PENDING profile; [complete] = all 8 items. Call after withRecord (it re-stubs the user). */
    private fun pendingProfile(complete: Boolean = true): PhotographerSettings {
        val user = User(id = userId, email = "seller@test.local", passwordHash = "unused", name = "S", role = Role.PHOTOGRAPHER)
            .apply { avatarUrl = "https://example.test/avatar.jpg" }
        Mockito.`when`(userRepository.lockById(userId)).thenReturn(user)
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
        Mockito.`when`(payoutRepository.countByUserIdAndQrS3KeyIsNotNull(userId)).thenReturn(1L)
        val settings = PhotographerSettings(
            userId = userId, handle = "aira", brandName = "Aira Studio", coverS3Key = "covers/c.jpg",
            watermarkS3Key = "watermarks/w.png".takeIf { complete }, regionCode = "07", provinceCode = "0722",
            verificationStatus = VerificationStatus.PENDING,
        )
        Mockito.`when`(settingsRepository.findByUserIdForUpdate(userId)).thenReturn(settings)
        Mockito.`when`(settingsRepository.save(anyArg<PhotographerSettings>())).thenAnswer { it.getArgument(0) }
        return settings
    }

    private fun approvedMessages() = messages().filter { it.kindWire == PhotographerMessageKind.VERIFICATION_APPROVED.wire }

    @Test
    fun `admin identity approval flips a pending complete profile`() {
        withRecord()
        val settings = pendingProfile()

        service(profiles = profiles()).approve(adminId, userId, 4)

        assertEquals(VerificationStatus.APPROVED, settings.verificationStatus)
        assertEquals(1L, settings.verificationRevision)
        assertNull(approvedMessages().single().sourceAdminId) // automatic: no admin on the message
    }

    @Test
    fun `admin identity approval leaves an incomplete profile alone`() {
        withRecord()
        val settings = pendingProfile(complete = false)

        service(profiles = profiles()).approve(adminId, userId, 4)

        assertEquals(VerificationStatus.PENDING, settings.verificationStatus)
        assertTrue(approvedMessages().isEmpty())
    }

    @Test
    fun `a passing live check flips a pending complete profile`() {
        val old = reviewRecord(SellerReasonCode.DECLINED)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))
        val settings = pendingProfile()

        val checked = service(profiles = profiles()).liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        assertEquals("auto_approved", checked.reviewState)
        assertEquals(VerificationStatus.APPROVED, settings.verificationStatus)
        assertEquals(1, approvedMessages().size)
    }

    @Test
    fun `a submit that lands in manual review never asks for profile approval`() {
        val profiles = Mockito.mock(PhotographerSettingsService::class.java)

        assertEquals("manual_review", service(profiles = profiles).submit(userId, form()).reviewState) // no live check: DECLINED

        Mockito.verifyNoInteractions(profiles)
    }

    // ─── AA-I6: uploadsAllowed and identityGateStartsOn ───

    @Test
    fun `uploadsAllowed follows the same gate set as the upload refusal`() {
        val expected = mapOf(
            false to mapOf(SellerDetailsStatus.MANUAL_REVIEW to true, SellerDetailsStatus.AUTO_APPROVED to true, SellerDetailsStatus.REJECTED to false, null to false),
            true to mapOf(SellerDetailsStatus.MANUAL_REVIEW to false, SellerDetailsStatus.AUTO_APPROVED to true, SellerDetailsStatus.REJECTED to false, null to false),
        )
        for ((gate, byStatus) in expected) for ((status, allowed) in byStatus) {
            Mockito.`when`(repository.findById(userId)).thenReturn(Optional.ofNullable(status?.let(::existing)))
            val dto = service(IdentityProperties(gateEnabled = gate)).get(userId)
            assertEquals(allowed, dto.uploadsAllowed, "gate=$gate status=$status")
            assertEquals(allowed, status != null && status in SellerDetailsStatus.uploadGateSet(gate), "the upload refusal's set, gate=$gate status=$status")
        }
    }

    @Test
    fun `identityGateStartsOn comes from config and is null when blank`() {
        val day = java.time.LocalDate.parse("2026-12-01")
        assertEquals(day, service(IdentityProperties(gateStartsOn = day)).get(userId).identityGateStartsOn)
        assertNull(service().get(userId).identityGateStartsOn)
    }

    @Test
    fun `submit and live check responses carry uploadsAllowed and identityGateStartsOn`() {
        val day = java.time.LocalDate.parse("2026-12-01")
        val identity = IdentityProperties(gateEnabled = true, gateStartsOn = day)
        val submitted = service(identity).submit(userId, form(livenessSessionId = SESSION))
        assertEquals("auto_approved", submitted.reviewState)
        assertTrue(submitted.uploadsAllowed)
        assertEquals(day, submitted.identityGateStartsOn)

        val old = reviewRecord(SellerReasonCode.DECLINED)
        Mockito.`when`(storage.getBytes(old.idFrontKey)).thenReturn(jpeg(32, 32))
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))
        val checked = service(identity).liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))
        assertEquals("auto_approved", checked.reviewState)
        assertTrue(checked.uploadsAllowed)
        assertEquals(day, checked.identityGateStartsOn)
    }

    private val payoutRepository: PayoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)

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
        livenessSessionId: String? = null,
        idNumber: String? = ID_NUMBER,
    ) = SellerDetailsForm(
        livenessSessionId = livenessSessionId,
        idNumber = idNumber,
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
        phoneKey = "9170000000",
        status = status,
        rejectionReason = "blurry",
        revision = 4,
        submittedAt = OffsetDateTime.parse("2026-11-01T00:00:00Z"),
        reviewedAt = OffsetDateTime.parse("2026-11-02T00:00:00Z"),
        reviewedBy = UUID.randomUUID(),
    )

    // 1
    @Test
    fun `submit stores upright JPEGs under seller-details with random names and saves MANUAL_REVIEW`() {
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
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
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

    @Test
    fun `an empty file part, as a browser sends for a blank file input, counts as not sent`() {
        fun blank(name: String) = MockMultipartFile(name, "", "application/octet-stream", ByteArray(0))

        service().submit(userId, form(idBack = blank("idBack"), birForm = blank("birForm")))
        assertNull(saved().birFormKey)
        assertNull(saved().idBackKey)

        for (field in listOf("idFront", "selfie")) {
            val error = assertFailsWith<ValidationException>(field) {
                service().submit(userId, if (field == "idFront") form(idFront = blank(field)) else form(selfie = blank(field)))
            }
            assertEquals(ErrorCodes.VALIDATION_ERROR, error.code)
            assertEquals(field, error.field)
        }
        val back = assertFailsWith<ValidationException> { service().submit(userId, form(idType = "philsys", idBack = blank("idBack"))) }
        assertEquals("idBack", back.field)
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
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
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
    fun `a storage failure before the save deletes the files already stored`() {
        Mockito.`when`(storage.put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>()))
            .thenAnswer { null }.thenThrow(IllegalStateException("r2 down"))

        assertFailsWith<IllegalStateException> { service().submit(userId, form()) }

        val (first, second) = storedKeys()
        Mockito.verify(storage).delete(first)
        Mockito.verify(storage).delete(second) // the failed put may have half-written it
        Mockito.verify(repository, Mockito.never()).save(anyArg<SellerDetails>())
    }

    @Test
    fun `a commit whose outcome is unknown keeps every file, because the saved row may point to the new ones`() {
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(existing()))
        txManager.commitFailure = TransactionSystemException("connection reset during commit")

        assertFailsWith<TransactionSystemException> { service().submit(userId, form()) }

        assertEquals(2, storedKeys().size)
        Mockito.verify(storage, Mockito.never()).delete(anyArg())
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

    private fun withRecord(status: SellerDetailsStatus = SellerDetailsStatus.MANUAL_REVIEW, birForm: Boolean = true): SellerDetails {
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
    fun `admin view links the reference image and logs that it was opened`() {
        val row = withRecord(birForm = false).apply { livenessReferenceKey = "seller-details/$userId/ref.jpg" }

        val dto = service().adminView(adminId, userId)

        assertEquals("https://signed.example/${row.livenessReferenceKey}?ttl=300&name=liveness_reference.jpg", dto.files!!.livenessReference)
        assertEquals(listOf("id_front", "id_back", "selfie", "liveness_reference"), decisions().single().meta!!["files"])
    }

    @Test
    fun `repeat matches carry email and the live suspended state`() {
        val otherId = UUID.randomUUID()
        val consent = OffsetDateTime.parse("2026-11-03T00:00:00Z")
        withRecord().apply {
            reasonCodes = listOf(SellerReasonCode.REPEAT_ACCOUNT_MATCH)
            livenessConfidence = BigDecimal("97.50")
            faceSimilarity = BigDecimal("99.20")
            autoCheckedAt = consent
            biometricConsentAt = consent
            idNumberHmac = "ab".repeat(32)
            repeatMatches = listOf(RepeatMatch(otherId, listOf(RepeatMatchKind.PHONE, RepeatMatchKind.ID_NUMBER)))
        }
        val other = User(id = otherId, email = "other@test.local", passwordHash = "unused", name = "O", role = Role.PHOTOGRAPHER)
            .apply { suspendedAt = consent }
        Mockito.`when`(userRepository.findAllById(listOf(otherId))).thenReturn(listOf(other))

        val dto = service().adminView(adminId, userId)

        assertEquals(listOf(com.quickpitik.dto.admin.RepeatMatchDto(otherId, "other@test.local", listOf("PHONE", "ID_NUMBER"), true)), dto.repeatMatches)
        assertEquals(listOf("REPEAT_ACCOUNT_MATCH"), dto.reasonCodes)
        assertEquals(BigDecimal("97.50"), dto.livenessConfidence)
        assertEquals(BigDecimal("99.20"), dto.faceSimilarity)
        assertEquals(consent, dto.autoCheckedAt)
        assertEquals(consent, dto.biometricConsentAt)
        assertTrue(dto.hasIdNumber)
        assertFalse("ab".repeat(32) in dto.toString(), "the ID-number hash never leaves the server")
    }

    @Test
    fun `admin list filters auto approved`() {
        val row = withRecord(SellerDetailsStatus.AUTO_APPROVED).apply {
            reasonCodes = listOf(SellerReasonCode.PASSED)
            livenessConfidence = BigDecimal("98.00")
            faceSimilarity = BigDecimal("99.90")
            autoCheckedAt = OffsetDateTime.parse("2026-11-03T00:00:00Z")
        }
        Mockito.`when`(repository.findForAdmin(Mockito.eq(SellerDetailsStatus.AUTO_APPROVED) ?: SellerDetailsStatus.AUTO_APPROVED, anyArg()))
            .thenReturn(PageImpl(listOf(row)))

        val item = service().adminList("auto_approved", PaginationParams.of(0, 20)).items.single()

        assertEquals("auto_approved", item.reviewState)
        assertEquals(listOf("PASSED"), item.reasonCodes)
        assertEquals(BigDecimal("98.00"), item.livenessConfidence)
        assertEquals(BigDecimal("99.90"), item.faceSimilarity)
        assertEquals(row.autoCheckedAt, item.autoCheckedAt)
    }

    @Test
    fun `the admin view of a photographer with no record answers 404 and logs nothing`() {
        assertFailsWith<NotFoundException> { service().adminView(adminId, userId) }
        assertTrue(decisions().isEmpty())
    }

    // 2
    @Test
    fun `admin list reads submitted as manual review, has no links, and puts manual review first`() {
        val row = withRecord()
        Mockito.`when`(repository.findForAdmin(Mockito.eq(SellerDetailsStatus.MANUAL_REVIEW) ?: SellerDetailsStatus.MANUAL_REVIEW, anyArg()))
            .thenReturn(PageImpl(listOf(row)))
        val user = userRepository.findById(userId).get()
        Mockito.`when`(userRepository.findAllById(listOf(userId))).thenReturn(listOf(user))

        // The legacy "submitted" filter (today's admin website) and the new "manual_review" read the same queue.
        assertEquals(1L, service().adminList("manual_review", PaginationParams.of(0, 20)).total)
        val page = service().adminList("submitted", PaginationParams.of(0, 20))

        assertEquals(1L, page.total)
        val item = page.items.single()
        assertEquals("seller@test.local", item.email)
        assertEquals("submitted", item.status)
        assertEquals("manual_review", item.reviewState)
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
    fun `approve accepts only manual review at the current revision, a stale revision answers 409 and changes nothing`() {
        val row = withRecord()
        val stale = assertFailsWith<ApiException> { service().approve(adminId, userId, 3) }
        assertEquals(HttpStatus.CONFLICT, stale.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, stale.code)
        assertEquals("Seller details have changed. Reload before deciding.", stale.message)
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
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

        for (status in listOf(SellerDetailsStatus.AUTO_APPROVED, SellerDetailsStatus.APPROVED, SellerDetailsStatus.REJECTED)) {
            withRecord(status)
            val wrong = assertFailsWith<ApiException> { service().approve(adminId, userId, 4) }
            assertEquals(HttpStatus.CONFLICT, wrong.status)
        }
    }

    // 4
    @Test
    fun `reject accepts manual review auto approved and approved, needs a reason and the current revision, stores the reason, and sends an ADMIN_MESSAGE with it`() {
        for (from in listOf(SellerDetailsStatus.MANUAL_REVIEW, SellerDetailsStatus.AUTO_APPROVED, SellerDetailsStatus.APPROVED)) {
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
        assertTrue(row.status == SellerDetailsStatus.REJECTED, "the gate (existsByUserIdAndStatusIn uploadGateSet) refuses this row")

        service().submit(userId, form())

        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status, "the gate passes a MANUAL_REVIEW row")
        assertEquals(5L, row.revision)
        assertNull(row.rejectionReason)
    }

    // ─── AA-I1: review states ────────────────────────────────────────────

    @Test
    fun `submit with no live check saves manual review declined`() {
        service().submit(userId, form(phone = "+63 917 123 4567"))

        val row = saved()
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.DECLINED), row.reasonCodes) // AA-I3: "Have an admin check instead"
        assertEquals(OffsetDateTime.now(clock), row.autoCheckedAt)
        assertNull(row.biometricConsentAt)
        assertNull(row.livenessReferenceKey)
        assertEquals("9171234567", row.phoneKey)
        Mockito.verifyNoInteractions(sessions, identityClient)
    }

    @Test
    fun `a resubmission refreshes the phone key and replaces stale reason codes`() {
        val old = existing(SellerDetailsStatus.MANUAL_REVIEW).apply {
            reasonCodes = listOf(SellerReasonCode.FACE_MISMATCH)
            phoneKey = "9170000000"
        }
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))

        service().submit(userId, form(phone = "0918 765 4321"))

        assertEquals(listOf(SellerReasonCode.DECLINED), saved().reasonCodes)
        assertEquals("9187654321", old.phoneKey)
    }

    @Test
    fun `status stays submitted for manual review and approved for auto approved while reviewState carries the detailed state`() {
        val wire = SellerDetailsStatus.entries.associateWith { it.toWire() to it.reviewState() }
        assertEquals(
            mapOf(
                SellerDetailsStatus.MANUAL_REVIEW to ("submitted" to "manual_review"),
                SellerDetailsStatus.AUTO_APPROVED to ("approved" to "auto_approved"),
                SellerDetailsStatus.APPROVED to ("approved" to "approved"),
                SellerDetailsStatus.REJECTED to ("rejected" to "rejected"),
            ),
            wire,
        )
        assertEquals(setOf(SellerDetailsStatus.AUTO_APPROVED, SellerDetailsStatus.APPROVED),
            SellerDetailsStatus.entries.filter { it.passesIdentity() }.toSet())
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, SellerDetailsStatus.fromWire("submitted"))
        assertEquals(SellerDetailsStatus.AUTO_APPROVED, SellerDetailsStatus.fromWire("auto_approved"))
        assertNull(SellerDetailsStatus.fromWire("missing"))

        val row = withRecord(SellerDetailsStatus.AUTO_APPROVED)
        assertEquals("approved", row.toDto().status)
        assertEquals("auto_approved", row.toDto().reviewState)
        assertEquals("missing", SellerDetailsDto.MISSING.reviewState)
        assertEquals("auto_approved", row.toAdminDto("a@b.c").reviewState)
    }

    @Test
    fun `phone key is the last ten digits`() {
        assertEquals("9171234567", SellerDetailsService.phoneKey("+63 917 123 4567"))
        assertEquals("9171234567", SellerDetailsService.phoneKey("09171234567"))
    }

    // ─── AA-I3: identity decided at submit, and the live-check-only route ─

    private fun putBytes(key: String): ByteArray = puts().single { it.arguments[0] == key }.arguments[1] as ByteArray

    /** A MANUAL_REVIEW record that already holds a reference image from an earlier live check. */
    private fun reviewRecord(vararg reasons: SellerReasonCode) = existing(SellerDetailsStatus.MANUAL_REVIEW).apply {
        legalName = "Juan Dela Cruz" // printed on the fixture's ID front
        livenessReferenceKey = "seller-details/$userId/old-reference.jpg"
        reasonCodes = reasons.toList()
        Mockito.`when`(storage.getBytes(idFrontKey)).thenReturn(jpeg())
    }

    @Test
    fun `live check keeps the id front back selfie and bir files and deletes only the old reference`() {
        val old = reviewRecord(SellerReasonCode.DECLINED)
        val kept = listOf(old.idFrontKey, old.idBackKey!!, old.selfieKey, old.birFormKey!!)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))

        service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        val row = saved()
        assertEquals(kept, listOf(row.idFrontKey, row.idBackKey, row.selfieKey, row.birFormKey))
        val newReference = storedKeys().single()
        assertEquals(newReference, row.livenessReferenceKey)
        Mockito.verify(storage).delete("seller-details/$userId/old-reference.jpg")
        (kept + newReference).forEach { Mockito.verify(storage, Mockito.never()).delete(it) }
    }

    @Test
    fun `a live check that gets no new still keeps the earlier one`() {
        val old = reviewRecord(SellerReasonCode.LIVENESS_LOW)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))
        Mockito.`when`(identityClient.livenessResult(SESSION)).thenThrow(SdkClientException.create("aws down"))

        service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        val row = saved()
        assertEquals(listOf(SellerReasonCode.AI_ERROR), row.reasonCodes)
        assertEquals("seller-details/$userId/old-reference.jpg", row.livenessReferenceKey)
        assertTrue(storedKeys().isEmpty())
        Mockito.verify(storage, Mockito.never()).delete(anyArg())
    }

    @Test
    fun `passing checks save auto approved with PASSED`() {
        val dto = service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        assertEquals(SellerDetailsStatus.AUTO_APPROVED, row.status)
        assertEquals(listOf(SellerReasonCode.PASSED), row.reasonCodes)
        assertEquals(BigDecimal("99.50"), row.livenessConfidence)
        assertEquals(BigDecimal("99.80"), row.faceSimilarity)
        assertEquals(SESSION, row.livenessSessionId)
        assertEquals(consentAt, row.biometricConsentAt)
        assertEquals(OffsetDateTime.now(clock), row.autoCheckedAt)
        assertTrue(row.livenessReferenceKey!!.startsWith("seller-details/$userId/"))
        assertTrue(putBytes(row.livenessReferenceKey!!).contentEquals(reference))
        // Compared against the stored (re-encoded) ID front, never the holding-ID selfie.
        Mockito.verify(identityClient).compareFaces(reference, putBytes(row.idFrontKey))
        assertEquals("approved", dto.status)
        assertEquals("auto_approved", dto.reviewState)
    }

    @Test
    fun `resubmission deletes the old reference image after commit and keeps the new one`() {
        val old = reviewRecord(SellerReasonCode.FACE_MISMATCH)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))

        service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        val order = Mockito.inOrder(txManager, storage)
        order.verify(txManager).commit(anyArg())
        order.verify(storage).delete("seller-details/$userId/old-reference.jpg")
        Mockito.verify(storage, Mockito.never()).delete(row.livenessReferenceKey!!)
        assertEquals(SellerDetailsStatus.AUTO_APPROVED, row.status)
    }

    @Test
    fun `live check on a manual review record re-runs the checks on the stored id front without new files`() {
        val old = reviewRecord(SellerReasonCode.DECLINED)
        val storedFront = jpeg(32, 32)
        Mockito.`when`(storage.getBytes(old.idFrontKey)).thenReturn(storedFront)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))

        val dto = service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        Mockito.verify(identityClient).compareFaces(reference, storedFront)
        assertEquals(1, storedKeys().size, "only the new reference image is stored")
        val row = saved()
        assertEquals(SellerDetailsStatus.AUTO_APPROVED, row.status)
        assertEquals(listOf(SellerReasonCode.PASSED), row.reasonCodes)
        assertEquals(5L, row.revision)
        assertEquals("auto_approved", dto.reviewState)
    }

    @Test
    fun `a submitted legal name that is not on the id front goes to manual review naming it`() {
        service().submit(userId, form(livenessSessionId = SESSION).apply { legalName = "Maria Santos" })

        val row = saved()
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.ID_NAME_MISMATCH), row.reasonCodes)
    }

    @Test
    fun `live check compares the stored legal name with the id front`() {
        val old = reviewRecord(SellerReasonCode.DECLINED).apply { legalName = "Maria Santos" }
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))

        service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        assertEquals(listOf(SellerReasonCode.ID_NAME_MISMATCH), saved().reasonCodes)
    }

    // ─── R-8: repeated failed tries go to an admin ───

    /** [failures] live checks that fail the face match, then one that passes, all on one MANUAL_REVIEW record. */
    private fun passAfterFailedTries(failures: Int): SellerDetails {
        val row = reviewRecord(SellerReasonCode.DECLINED)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(row))
        Mockito.`when`(identityClient.compareFaces(anyArg(), anyArg())).thenReturn(FaceCompareResult(1, 50f))
        repeat(failures) {
            service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))
            assertEquals(listOf(SellerReasonCode.FACE_MISMATCH), row.reasonCodes)
        }
        Mockito.`when`(identityClient.compareFaces(anyArg(), anyArg())).thenReturn(FaceCompareResult(1, 99.8f))
        service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))
        return row
    }

    @Test
    fun `a passing live check after three failed tries goes to an admin`() {
        val row = passAfterFailedTries(3)

        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.REPEATED_FAILED_CHECKS), row.reasonCodes)
        assertEquals(3, row.failedAutoChecks)
    }

    @Test
    fun `a passing live check after two failed tries is approved`() {
        val row = passAfterFailedTries(2)

        assertEquals(SellerDetailsStatus.AUTO_APPROVED, row.status)
        assertEquals(listOf(SellerReasonCode.PASSED), row.reasonCodes)
        assertEquals(2, row.failedAutoChecks)
    }

    @Test
    fun `an aws error or ai off does not count as a failed try`() {
        val row = reviewRecord(SellerReasonCode.DECLINED)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(row))
        Mockito.doThrow(SdkClientException.create("aws down")).`when`(identityClient).livenessResult(SESSION)
        repeat(3) { service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER)) }
        assertEquals(listOf(SellerReasonCode.AI_ERROR), row.reasonCodes)
        repeat(3) { service(aiEnabled = false).liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER)) }
        assertEquals(listOf(SellerReasonCode.AI_DISABLED), row.reasonCodes)
        assertEquals(0, row.failedAutoChecks)

        Mockito.doReturn(LivenessResult(LivenessSessionStatus.SUCCEEDED, 99.5f, reference)).`when`(identityClient).livenessResult(SESSION)
        service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        assertEquals(SellerDetailsStatus.AUTO_APPROVED, row.status)
    }

    @Test
    fun `the admin detail shows the failed-try count`() {
        assertEquals(2, existing().apply { failedAutoChecks = 2 }.toAdminDto("seller@test.local").failedAutoChecks)
    }

    @Test
    fun `live check outside manual review answers invalid state transition`() {
        for (status in listOf(null, SellerDetailsStatus.AUTO_APPROVED, SellerDetailsStatus.APPROVED, SellerDetailsStatus.REJECTED)) {
            Mockito.`when`(repository.findById(userId)).thenReturn(Optional.ofNullable(status?.let(::existing)))

            val error = assertFailsWith<ApiException>(status.toString()) { service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER)) }

            assertEquals(HttpStatus.CONFLICT, error.status)
            assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, error.code)
        }
        Mockito.verifyNoInteractions(sessions, identityClient) // refused before the session is used up
        Mockito.verify(repository, Mockito.never()).save(anyArg<SellerDetails>())
    }

    @Test
    fun `an admin decision during the live check answers invalid state transition and keeps the admin's state`() {
        val decided = listOf(
            existing(SellerDetailsStatus.REJECTED),
            existing(SellerDetailsStatus.APPROVED),
            existing(SellerDetailsStatus.MANUAL_REVIEW).apply { revision = 5 }, // a resubmission from another tab
        )
        for (later in decided) {
            Mockito.clearInvocations(storage, repository)
            val first = reviewRecord()
            Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(first), Optional.of(later))

            val error = assertFailsWith<ApiException> { service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER)) }

            assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, error.code)
            Mockito.verify(repository, Mockito.never()).save(anyArg<SellerDetails>())
            val created = storedKeys().single()
            Mockito.verify(storage).delete(created) // this request's reference image
            Mockito.verify(storage, Mockito.times(1)).delete(anyArg())
        }
    }

    @Test
    fun `a resubmission after an admin rejection lands in manual review with PREVIOUSLY_REJECTED even when every check passes`() {
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(existing(SellerDetailsStatus.REJECTED)))

        service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.PREVIOUSLY_REJECTED), row.reasonCodes)
        assertEquals(BigDecimal("99.80"), row.faceSimilarity, "the checks still run, so the admin sees the scores")
    }

    @Test
    fun `a live check on a previously rejected record keeps PREVIOUSLY_REJECTED`() {
        val old = reviewRecord(SellerReasonCode.DECLINED, SellerReasonCode.PREVIOUSLY_REJECTED)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))

        service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        val row = saved()
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.PREVIOUSLY_REJECTED), row.reasonCodes)
    }

    @Test
    fun `an admin approval clears PREVIOUSLY_REJECTED for a later resubmission`() {
        val approved = existing(SellerDetailsStatus.APPROVED).apply { reasonCodes = listOf(SellerReasonCode.PREVIOUSLY_REJECTED) }
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(approved))

        service().submit(userId, form(livenessSessionId = SESSION))

        assertEquals(SellerDetailsStatus.AUTO_APPROVED, saved().status)
    }

    @Test
    fun `a second submit with the same session answers LIVENESS_SESSION_INVALID`() {
        Mockito.`when`(sessions.claim(userId, SESSION, LivenessPurpose.PHOTOGRAPHER_IDENTITY, null)).thenReturn(false)

        val error = assertFailsWith<ApiException> { service().submit(userId, form(livenessSessionId = SESSION)) }

        assertEquals(ErrorCodes.LIVENESS_SESSION_INVALID, error.code)
        Mockito.verifyNoInteractions(identityClient)
        assertNothingStored()
    }

    @Test
    fun `a live check needs a session id`() {
        val validator = Validation.buildDefaultValidatorFactory().validator
        assertFalse(validator.validate(LiveCheckRequest(null)).isEmpty())
        assertFalse(validator.validate(LiveCheckRequest("  ")).isEmpty())
        assertTrue(validator.validate(LiveCheckRequest(SESSION, ID_NUMBER)).isEmpty())
    }

    // ─── AA-I4: the typed ID number, kept only as a keyed hash ───────────

    @Test
    fun `a live check needs an id number`() {
        val validator = Validation.buildDefaultValidatorFactory().validator
        assertFalse(validator.validate(LiveCheckRequest(SESSION, null)).isEmpty())
        assertFalse(validator.validate(LiveCheckRequest(SESSION, " ")).isEmpty())
    }

    @Test
    fun `submit stores only the keyed hash of the typed number`() {
        val dto = service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        assertEquals(listOf(SellerReasonCode.PASSED), row.reasonCodes)
        assertTrue(Regex("[0-9a-f]{64}").matches(row.idNumberHmac.orEmpty()), row.idNumberHmac)
        val raw = "N0112345678"
        assertFalse(raw in row.toString() || raw in dto.toString() || ID_NUMBER in dto.toString())
    }

    @Test
    fun `a declined submit still stores the hash`() {
        service().submit(userId, form())

        assertTrue(Regex("[0-9a-f]{64}").matches(saved().idNumberHmac.orEmpty()))
    }

    @Test
    fun `a bad id number is refused before the session is used and nothing is stored`() {
        val error = assertFailsWith<ValidationException> { service().submit(userId, form(livenessSessionId = SESSION, idNumber = "N0")) }

        assertEquals("idNumber", error.field)
        Mockito.verifyNoInteractions(sessions, identityClient)
        assertNothingStored()
    }

    @Test
    fun `a live check replaces the hash with the new typed number`() {
        val old = reviewRecord(SellerReasonCode.DECLINED).apply { idNumberHmac = "0".repeat(64) }
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))

        service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        val row = saved()
        assertEquals(listOf(SellerReasonCode.PASSED), row.reasonCodes)
        assertTrue(Regex("[0-9a-f]{64}").matches(row.idNumberHmac.orEmpty()))
        assertNotEquals("0".repeat(64), row.idNumberHmac)
    }

    // AA-I5: repeat accounts and payout names.
    private val other = UUID.randomUUID()

    private fun payout(name: String = "Juan Dela Cruz", number: String = "09171234567") =
        PayoutAccount(userId = userId, method = PayoutMethod.GCASH, accountNumber = number, accountName = name)

    @Test
    fun `same id hash on another account goes to manual review naming it`() {
        Mockito.`when`(repository.findUserIdsByIdNumberHmacIn(anyArg(), Mockito.eq(userId) ?: userId)).thenReturn(listOf(other))

        val dto = service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.REPEAT_ACCOUNT_MATCH), row.reasonCodes)
        assertEquals(listOf(RepeatMatch(other, listOf(RepeatMatchKind.ID_NUMBER))), row.repeatMatches)
        // Matched on the number's hash under every type, the saved one among them, and under the user lock.
        val hashes = matchHashes("N0112345678")
        assertTrue(row.idNumberHmac in hashes)
        val order = Mockito.inOrder(userRepository, repository)
        order.verify(userRepository).lockById(userId)
        order.verify(repository).findUserIdsByIdNumberHmacIn(hashes, userId)
        assertEquals("manual_review", dto.reviewState)
    }

    @Test
    fun `the same id number under another id type on another account goes to manual review naming it`() {
        val passportHash = identityChecks().idNumberHmac(SellerIdType.PASSPORT, "N0112345678")
        // Like the database: only the other account's row, stored under the passport hash, matches.
        Mockito.`when`(repository.findUserIdsByIdNumberHmacIn(anyArg(), Mockito.eq(userId) ?: userId))
            .thenAnswer { if (passportHash in it.getArgument<List<String>>(0)) listOf(other) else emptyList() }

        service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.REPEAT_ACCOUNT_MATCH), row.reasonCodes)
        assertEquals(listOf(RepeatMatch(other, listOf(RepeatMatchKind.ID_NUMBER))), row.repeatMatches)
        assertNotEquals(passportHash, row.idNumberHmac) // stored under the chosen type, as before
    }

    // R-9: two accounts submitting one number at once take turns, so the second sees the first.
    @Test
    fun `the id number lock is taken before the repeat lookup`() {
        service().submit(userId, form(livenessSessionId = SESSION))

        val hashes = matchHashes("N0112345678")
        val order = Mockito.inOrder(userRepository, repository)
        order.verify(userRepository).lockById(userId)
        order.verify(repository).lockIdNumber(hashes.first())
        order.verify(repository).findUserIdsByIdNumberHmacIn(hashes, userId)
    }

    // Same gap for the phone: the phone lock comes before the id lock (one order, no deadlock) and the lookup.
    @Test
    fun `the phone lock is taken before the id lock and the phone lookup`() {
        service().submit(userId, form(livenessSessionId = SESSION))

        val phoneKey = saved().phoneKey
        val order = Mockito.inOrder(repository)
        order.verify(repository).lockPhoneKey(phoneKey)
        order.verify(repository).lockIdNumber(anyArg())
        order.verify(repository).findUserIdsByPhoneKey(phoneKey, userId)
    }

    @Test
    fun `no typed number takes no id lock`() {
        service().submit(userId, form(idNumber = null))

        saved()
        Mockito.verify(repository, Mockito.never()).lockIdNumber(anyArg())
    }

    @Test
    fun `shared payout number and phone merge into one match`() {
        Mockito.`when`(payoutRepository.findAllByUserIdOrderByCreatedAtAsc(userId)).thenReturn(listOf(payout()))
        Mockito.`when`(payoutRepository.findUserIdsByAccountNumbers(listOf("09171234567"), userId)).thenReturn(listOf(other))
        Mockito.`when`(repository.findUserIdsByPhoneKey("9171234567", userId)).thenReturn(listOf(other))

        service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        assertEquals(listOf(RepeatMatch(other, listOf(RepeatMatchKind.PAYOUT_NUMBER, RepeatMatchKind.PHONE))), row.repeatMatches)
        assertEquals(listOf(SellerReasonCode.REPEAT_ACCOUNT_MATCH), row.reasonCodes)
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
    }

    @Test
    fun `payout holder name mismatch goes to manual review`() {
        Mockito.`when`(payoutRepository.findAllByUserIdOrderByCreatedAtAsc(userId))
            .thenReturn(listOf(payout(), payout(name = "Maria Santos", number = "09179999999")))

        service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.PAYOUT_NAME_MISMATCH), row.reasonCodes)
        assertEquals(emptyList(), row.repeatMatches)
    }

    @Test
    fun `no payout account skips the name check`() {
        service().submit(userId, form(livenessSessionId = SESSION))

        val row = saved()
        assertEquals(SellerDetailsStatus.AUTO_APPROVED, row.status)
        assertEquals(listOf(SellerReasonCode.PASSED), row.reasonCodes)
        Mockito.verify(payoutRepository, Mockito.never()).findUserIdsByAccountNumbers(anyArg(), anyArg())
    }

    @Test
    fun `a repeat match never refuses the submit`() {
        Mockito.`when`(repository.findUserIdsByIdNumberHmacIn(anyArg(), Mockito.eq(userId) ?: userId)).thenReturn(listOf(other))
        Mockito.`when`(repository.findUserIdsByPhoneKey("9171234567", userId)).thenReturn(listOf(other, UUID.randomUUID()))

        val dto = service().submit(userId, form())

        assertEquals("submitted", dto.status)
        val row = saved()
        assertEquals(listOf(SellerReasonCode.DECLINED, SellerReasonCode.REPEAT_ACCOUNT_MATCH), row.reasonCodes)
        assertEquals(2, row.repeatMatches.size)
        assertEquals(listOf(RepeatMatchKind.ID_NUMBER, RepeatMatchKind.PHONE), row.repeatMatches.first { it.userId == other }.matchedOn)
    }

    @Test
    fun `a live check runs the repeat checks on the stored phone and name`() {
        val old = reviewRecord(SellerReasonCode.DECLINED)
        Mockito.`when`(repository.findById(userId)).thenReturn(Optional.of(old))
        Mockito.`when`(repository.findUserIdsByPhoneKey("9170000000", userId)).thenReturn(listOf(other))
        Mockito.`when`(payoutRepository.findAllByUserIdOrderByCreatedAtAsc(userId)).thenReturn(listOf(payout(name = "Juan Dela Cruz")))

        service().liveCheck(userId, LiveCheckRequest(SESSION, ID_NUMBER))

        val row = saved()
        assertEquals(listOf(SellerReasonCode.REPEAT_ACCOUNT_MATCH), row.reasonCodes)
        assertEquals(listOf(RepeatMatch(other, listOf(RepeatMatchKind.PHONE))), row.repeatMatches)
    }

    private companion object {
        const val SESSION = "s-1"
        const val ID_NUMBER = "N01-12-345678"
    }
}
