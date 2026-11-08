package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.moderation.ImageModerationResult
import com.quickpitik.service.moderation.ImageModerationService
import com.quickpitik.service.moderation.ImageModerationVerdict
import com.quickpitik.support.testTransactionTemplate
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.io.ByteArrayInputStream
import java.util.Base64
import java.util.Optional
import java.util.UUID
import javax.imageio.ImageIO
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

// The MIME allow-list does not bound the decode: ImageIO sniffs the real format,
// so a BMP/TGA declared as image/png sails past the Content-Type check and then
// expands into a BufferedImage orders of magnitude larger than the upload.
// Spring's 25 MB multipart ceiling is far too loose to be the only guard.
class AvatarServiceTest {

    private val userRepository: UserRepository = Mockito.mock(UserRepository::class.java)
    private val storageService: StorageService = Mockito.mock(StorageService::class.java)
    private val userDtoMapper: UserDtoMapper = Mockito.mock(UserDtoMapper::class.java)

    private val moderation: ImageModerationService = Mockito.mock(ImageModerationService::class.java)
    private val flags: SystemFlagService = Mockito.mock(SystemFlagService::class.java)

    private fun service(mapper: UserDtoMapper = userDtoMapper) =
        AvatarService(userRepository, storageService, mapper, moderation, flags, testTransactionTemplate())

    @Test
    fun `valid WebP avatar is stored as a 512 pixel JPEG`() {
        // Exact synthetic 16x16 WebP that failed staging; no external fixture path.
        val webp = Base64.getDecoder().decode(
            "UklGRjIAAABXRUJQVlA4ICYAAACQAQCdASoQABAAAUAmJYgCdLoAA5gA/vPfZrQtCBz+N5vNasQAAA==",
        )
        val user = User(email = "runner@test.local", passwordHash = "unused", name = "Runner", role = Role.RUNNER)
        Mockito.`when`(userRepository.findById(user.id)).thenReturn(Optional.of(user))
        Mockito.`when`(storageService.presignedGetUrl(anyArg(), anyArg())).thenReturn("https://avatar.test/current")
        val mapper = UserDtoMapper(storageService, StorageProperties())

        val result = service(mapper).upload(user.id, webp, "image/webp")

        val key = ArgumentCaptor.forClass(String::class.java)
        val bytes = ArgumentCaptor.forClass(ByteArray::class.java)
        val contentType = ArgumentCaptor.forClass(String::class.java)
        Mockito.verify(storageService).put(key.capture() ?: "", bytes.capture() ?: byteArrayOf(), contentType.capture() ?: "")
        assertEquals("image/jpeg", contentType.value)
        assertContentEquals(byteArrayOf(-1, -40, -1), bytes.value.take(3).toByteArray())
        val image = assertNotNull(ImageIO.read(ByteArrayInputStream(bytes.value)))
        assertEquals(512, image.width)
        assertEquals(512, image.height)
        assertTrue(key.value.startsWith("avatars/${user.id}/"))
        assertTrue(key.value.endsWith(".jpg"))
        assertEquals(key.value, user.avatarS3Key)
        Mockito.verify(userRepository).save(user)
        assertEquals(user.id, result.id)
        assertEquals(user.email, result.email)
        assertEquals(user.name, result.name)
        assertEquals("https://avatar.test/current", result.avatarUrl)
        Mockito.verify(storageService).presignedGetUrl(key.value, StorageProperties().presignedTtl.avatar)
    }

    @Test
    fun `avatar over 5MB is rejected before decode`() {
        val ex = assertFailsWith<ApiException> {
            service().upload(UUID.randomUUID(), ByteArray(5 * 1024 * 1024 + 1), "image/png")
        }

        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE, ex.status)
        assertEquals(ErrorCodes.PAYLOAD_TOO_LARGE, ex.code)
        // Nothing was decoded, looked up, or stored.
        Mockito.verifyNoInteractions(storageService)
        Mockito.verifyNoInteractions(userRepository)
    }

    @Test
    fun `an undersized but undecodable payload still fails on decode, not on size`() {
        // Just under the cap: proves the size gate let it through and the
        // existing UNSUPPORTED_MEDIA_TYPE path is still what catches garbage.
        val ex = assertFailsWith<com.quickpitik.exception.ValidationException> {
            service().upload(UUID.randomUUID(), ByteArray(1024), "image/png")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
    }

    @Test
    fun `truncated WebP avatar is rejected as unsupported before storage`() {
        // RIFF/WEBP/VP8 chunk header from the valid fixture, with no frame bytes.
        val bytes = Base64.getDecoder().decode("UklGRjIAAABXRUJQVlA4ICYAAAA=")

        val error = assertFailsWith<com.quickpitik.exception.ValidationException> {
            service().upload(UUID.randomUUID(), bytes, "image/webp")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
        assertEquals("file", error.field)
        Mockito.verifyNoInteractions(userRepository, storageService)
    }

    // AA-I11: photographer avatars are checked before they are stored; runners' never are.
    private fun pngBytes(): ByteArray = java.io.ByteArrayOutputStream().also {
        ImageIO.write(java.awt.image.BufferedImage(8, 8, java.awt.image.BufferedImage.TYPE_INT_RGB), "png", it)
    }.toByteArray()

    private fun photographer(): User {
        val user = User(email = "pg@test.local", passwordHash = "unused", name = "Pg", role = Role.PHOTOGRAPHER)
        Mockito.`when`(userRepository.findById(user.id)).thenReturn(Optional.of(user))
        return user
    }

    private fun realMapper() = UserDtoMapper(storageService, StorageProperties())

    private fun verdict(v: ImageModerationVerdict, vararg labels: String) =
        Mockito.`when`(moderation.check(anyArg())).thenReturn(ImageModerationResult(v, labels.toList()))

    @Test
    fun `a flagged photographer avatar is refused with IMAGE_NOT_ALLOWED, nothing stored, PROFILE_IMAGE_FLAGGED raised`() {
        val user = photographer()
        verdict(ImageModerationVerdict.FLAGGED, "Explicit")
        Mockito.`when`(flags.raise(FlagTargetKind.USER, user.id, SystemFlagReason.PROFILE_IMAGE_FLAGGED, "avatar: Explicit"))
            .thenReturn(true)

        val ex = assertFailsWith<ValidationException> { service().upload(user.id, pngBytes(), "image/png") }

        assertEquals(ErrorCodes.IMAGE_NOT_ALLOWED, ex.code)
        assertEquals("file", ex.field)
        Mockito.verify(flags).raise(FlagTargetKind.USER, user.id, SystemFlagReason.PROFILE_IMAGE_FLAGGED, "avatar: Explicit")
        Mockito.verify(flags, Mockito.never()).appendNote(anyArg(), anyArg(), anyArg(), anyArg())
        Mockito.verifyNoInteractions(storageService)
        Mockito.verify(userRepository, Mockito.never()).save(anyArg())
        assertEquals(null, user.avatarS3Key)
    }

    @Test
    fun `a second flagged attempt adds a line to the open flag`() {
        val user = photographer()
        verdict(ImageModerationVerdict.FLAGGED, "Violence")
        // raise answers false: a PROFILE_IMAGE_FLAGGED flag is already open for this user.

        assertFailsWith<ValidationException> { service().upload(user.id, pngBytes(), "image/png") }

        val note = ArgumentCaptor.forClass(String::class.java)
        Mockito.verify(flags).appendNote(
            Mockito.eq(FlagTargetKind.USER) ?: FlagTargetKind.USER,
            Mockito.eq(user.id) ?: user.id,
            Mockito.eq(SystemFlagReason.PROFILE_IMAGE_FLAGGED) ?: SystemFlagReason.PROFILE_IMAGE_FLAGGED,
            note.capture() ?: "",
        )
        assertTrue(note.value.endsWith(" avatar: Violence"), note.value)
    }

    @Test
    fun `an unchecked avatar is stored and PROFILE_IMAGE_UNCHECKED raised`() {
        val user = photographer()
        verdict(ImageModerationVerdict.UNCHECKED)

        service(realMapper()).upload(user.id, pngBytes(), "image/png")

        assertTrue(user.avatarS3Key!!.startsWith("avatars/${user.id}/"))
        Mockito.verify(userRepository).save(user)
        Mockito.verify(flags).raise(FlagTargetKind.USER, user.id, SystemFlagReason.PROFILE_IMAGE_UNCHECKED, "avatar")
    }

    @Test
    fun `a runner avatar is never sent to moderation`() {
        val user = User(email = "r@test.local", passwordHash = "unused", name = "R", role = Role.RUNNER)
        Mockito.`when`(userRepository.findById(user.id)).thenReturn(Optional.of(user))

        service(realMapper()).upload(user.id, pngBytes(), "image/png")

        Mockito.verifyNoInteractions(moderation, flags)
        Mockito.verify(userRepository).save(user)
    }

    private fun <T> anyArg(): T = Mockito.any()
}
