package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.Role
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.entity.User
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.moderation.ImageModerationResult
import com.quickpitik.service.moderation.ImageModerationService
import com.quickpitik.service.moderation.ImageModerationVerdict
import com.quickpitik.service.photographer.PhotographerSettingsService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.any
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mockito.`when`
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.time.Duration
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// AA-I11 on real Postgres: the moderation call runs with no transaction open, a
// refused upload still leaves its flag, and an unchecked one commits key + flag together.
class ProfileImageModerationIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var service: PhotographerSettingsService
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var settings: PhotographerSettingsRepository
    @Autowired private lateinit var flags: FlagRepository
    @MockitoBean private lateinit var moderation: ImageModerationService
    @MockitoBean private lateinit var storage: StorageService

    private val txOpenDuringCheck = mutableListOf<Boolean>()

    private fun photographer(): UUID = users.save(
        User(email = "aa-i11-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Pg", role = Role.PHOTOGRAPHER),
    ).id

    private fun verdict(v: ImageModerationVerdict, vararg labels: String) {
        `when`(moderation.check(any() ?: byteArrayOf())).thenAnswer {
            txOpenDuringCheck += TransactionSynchronizationManager.isActualTransactionActive()
            ImageModerationResult(v, labels.toList())
        }
        `when`(storage.presignedGetUrl(anyString(), any(Duration::class.java) ?: Duration.ZERO)).thenReturn("https://cover.test/x")
    }

    private fun jpeg(): ByteArray = ByteArrayOutputStream().also {
        ImageIO.write(BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB), "jpg", it)
    }.toByteArray()

    private fun openFlags(userId: UUID, reason: SystemFlagReason) =
        flags.findByTargetKindWireAndTargetIdAndReasonAndStatusWireIn(
            FlagTargetKind.USER.wire, userId, reason.wire, listOf(FlagStatus.OPEN.wire),
        )

    @Test
    fun `the flag row survives a refused cover upload`() {
        val userId = photographer()
        verdict(ImageModerationVerdict.FLAGGED, "Hate Symbols")

        val ex = assertFailsWith<ValidationException> { service.uploadCover(userId, jpeg(), "image/jpeg") }

        assertEquals(ErrorCodes.IMAGE_NOT_ALLOWED, ex.code)
        assertEquals(listOf(false), txOpenDuringCheck)
        assertEquals("cover: Hate Symbols", openFlags(userId, SystemFlagReason.PROFILE_IMAGE_FLAGGED).single().note)
        assertNull(settings.findById(userId).orElse(null)?.coverS3Key)
    }

    @Test
    fun `an unchecked cover commits the image key and its flag together`() {
        val userId = photographer()
        verdict(ImageModerationVerdict.UNCHECKED)

        service.uploadCover(userId, jpeg(), "image/jpeg")

        assertFalse(txOpenDuringCheck.single())
        assertNotNull(settings.findById(userId).orElseThrow().coverS3Key)
        assertEquals(1, openFlags(userId, SystemFlagReason.PROFILE_IMAGE_UNCHECKED).size)
    }
}
