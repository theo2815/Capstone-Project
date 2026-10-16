package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.UserRepository
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

    private fun service() = AvatarService(userRepository, storageService, userDtoMapper)

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

        val result = AvatarService(userRepository, storageService, mapper).upload(user.id, webp, "image/webp")

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

    private fun <T> anyArg(): T = Mockito.any()
}
