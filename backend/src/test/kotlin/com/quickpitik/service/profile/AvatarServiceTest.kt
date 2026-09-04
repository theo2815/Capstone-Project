package com.quickpitik.service.profile

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

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
}
