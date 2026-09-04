package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.ai.FaceDetection
import com.quickpitik.dto.ai.FacesDetectResult
import com.quickpitik.entity.SelfieQualityTestStatus
import com.quickpitik.entity.UserSelfie
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.io.ByteArrayOutputStream
import java.awt.image.BufferedImage
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// Two behaviours locked here:
//
//  1. A size cap. Unlike avatar/cover this service does not decode on the happy
//     path, so the cap is not about heap — it bounds what gets pushed to S3 and
//     forwarded to ai-api.
//  2. EXIF pass-through. A selfie with no orientation tag must be stored byte-
//     for-byte as uploaded; re-encoding every selfie would cost quality for no
//     reason. (The rotation branch itself is covered by ExifOrientationTest.)
class SelfieServiceTest {

    private lateinit var userSelfieRepository: UserSelfieRepository
    private lateinit var storageService: StorageService
    private lateinit var aiApiClient: AiApiClient

    private val userId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        userSelfieRepository = Mockito.mock(UserSelfieRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        aiApiClient = Mockito.mock(AiApiClient::class.java)
    }

    // Default is ai-api disabled: the quality gate short-circuits, so those tests
    // exercise the upload plumbing without needing an inference stub. Pass
    // aiEnabled = true to drive the gate itself.
    private fun service(aiEnabled: Boolean = false) = SelfieService(
        userSelfieRepository,
        storageService,
        StorageProperties(),
        aiApiClient,
        AiApiProperties(enabled = aiEnabled),
    )

    private fun jpegBytes(w: Int = 24, h: Int = 24): ByteArray {
        val out = ByteArrayOutputStream()
        ImageIO.write(BufferedImage(w, h, BufferedImage.TYPE_INT_RGB), "jpeg", out)
        return out.toByteArray()
    }

    @Test
    fun `selfie over 5MB is rejected before storage or ai-api`() {
        val ex = assertFailsWith<ApiException> {
            service().upload(userId, ByteArray(5 * 1024 * 1024 + 1), "image/jpeg", "big.jpg")
        }

        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, ex.status)
        assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, ex.code)
        Mockito.verifyNoInteractions(storageService)
        Mockito.verifyNoInteractions(aiApiClient)
        // The cap is checked before the per-user count query too.
        Mockito.verifyNoInteractions(userSelfieRepository)
    }

    @Test
    fun `a selfie with no EXIF orientation is stored byte-for-byte`() {
        val original = jpegBytes()
        Mockito.`when`(userSelfieRepository.countByUserId(userId)).thenReturn(0L)
        Mockito.`when`(userSelfieRepository.save(anyArg<UserSelfie>()))
            .thenAnswer { it.arguments[0] as UserSelfie }
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://selfie")

        var storedKey: String? = null
        var storedBytes: ByteArray? = null
        var storedType: String? = null
        Mockito.`when`(storageService.put(anyArg<String>(), anyArg<ByteArray>(), anyArg<String>()))
            .thenAnswer { inv ->
                storedKey = inv.getArgument(0)
                storedBytes = inv.getArgument(1)
                storedType = inv.getArgument(2)
                null
            }

        service().upload(userId, original, "image/jpeg", "selfie.jpg")

        assertTrue(original.contentEquals(storedBytes), "bytes must pass through unmodified")
        assertEquals("image/jpeg", storedType)
        assertTrue(storedKey!!.endsWith(".jpg"))
    }

    // ai-api writes its error copy for API consumers ("No faces detected in
    // image"). It used to be forwarded verbatim into the runner's 4xx body.
    @Test
    fun `an ai-api rejection surfaces our copy, never ai-api's`() {
        stubHappyPathPlumbing()
        Mockito.`when`(aiApiClient.facesDetect(anyArg(), anyArg(), anyArg()))
            .thenThrow(
                AiApiException(
                    status = HttpStatus.UNPROCESSABLE_ENTITY,
                    aiCode = "NO_FACES",
                    message = "No faces detected in image",
                ),
            )

        val ex = assertFailsWith<ValidationException> {
            service(aiEnabled = true).upload(userId, jpegBytes(), "image/jpeg", "selfie.jpg")
        }

        assertEquals(ErrorCodes.SELFIE_REJECTED, ex.code)
        assertEquals("No face detected — make sure your face is centered and well-lit.", ex.message)
        Mockito.verify(storageService, Mockito.never()).put(anyArg(), anyArg(), anyArg())
    }

    @Test
    fun `an ai-api LOW_QUALITY rejection surfaces our copy`() {
        stubHappyPathPlumbing()
        Mockito.`when`(aiApiClient.facesDetect(anyArg(), anyArg(), anyArg()))
            .thenThrow(
                AiApiException(
                    status = HttpStatus.UNPROCESSABLE_ENTITY,
                    aiCode = "LOW_QUALITY",
                    message = "Face crop below quality threshold 0.42",
                ),
            )

        val ex = assertFailsWith<ValidationException> {
            service(aiEnabled = true).upload(userId, jpegBytes(), "image/jpeg", "selfie.jpg")
        }

        assertEquals("Image quality too low — try a sharper, better-lit selfie.", ex.message)
    }

    // With AI_API_ENABLED=false the selfie stores fine but nothing ever looked at
    // it. qualityScore=0 alone couldn't say so — hence the explicit status.
    @Test
    fun `a selfie uploaded with ai-api off reads back as untested`() {
        stubHappyPathPlumbing()

        val dto = service(aiEnabled = false).upload(userId, jpegBytes(), "image/jpeg", "selfie.jpg")

        assertEquals(SelfieQualityTestStatus.UNTESTED.wire, dto.qualityTestStatus)
        Mockito.verifyNoInteractions(aiApiClient)
    }

    @Test
    fun `a selfie that clears the gate reads back as passed`() {
        stubHappyPathPlumbing()
        Mockito.`when`(aiApiClient.facesDetect(anyArg(), anyArg(), anyArg()))
            .thenReturn(FacesDetectResult(faces = listOf(FaceDetection(confidence = 0.94))))

        val dto = service(aiEnabled = true).upload(userId, jpegBytes(), "image/jpeg", "selfie.jpg")

        assertEquals(SelfieQualityTestStatus.PASSED.wire, dto.qualityTestStatus)
    }

    private fun stubHappyPathPlumbing() {
        Mockito.`when`(userSelfieRepository.countByUserId(userId)).thenReturn(0L)
        Mockito.`when`(userSelfieRepository.save(anyArg<UserSelfie>()))
            .thenAnswer { it.arguments[0] as UserSelfie }
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://selfie")
    }
}
