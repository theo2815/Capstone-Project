package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataIntegrityViolationException
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertFailsWith

/**
 * `uq_photos_photographer_content_hash` (V24), exercised for real.
 *
 * `PhotoUploadServiceTest` mocks `PhotoRepository`, so it can only assert that
 * the service *asks* about a duplicate — the index that actually guarantees
 * uniqueness under a race has never run in a test. That index is the whole
 * safety story for duplicate uploads: two clients uploading the same shot can't
 * see each other, so the service-level check is a TOCTOU race that only the
 * database resolves.
 *
 * The partial-index predicate is the subtle part. `WHERE photographer_id IS NOT
 * NULL AND content_hash IS NOT NULL` means NULLs are excluded entirely — every
 * pre-V24 row, plus any row orphaned by `ON DELETE SET NULL`. Get that wrong
 * and either the backfill breaks or orphans start colliding.
 */
class PhotoDedupIndexIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var photoRepository: PhotoRepository

    @Autowired
    private lateinit var eventRepository: EventRepository

    @Autowired
    private lateinit var userRepository: UserRepository

    private lateinit var eventId: UUID
    private lateinit var photographerId: UUID
    private lateinit var otherPhotographerId: UUID

    @BeforeEach
    fun seed() {
        eventId = eventRepository.save(newEvent()).id
        photographerId = userRepository.save(newPhotographer()).id
        otherPhotographerId = userRepository.save(newPhotographer()).id
    }

    @Test
    fun `the same photographer cannot store one hash twice`() {
        val hash = randomHash()
        photoRepository.saveAndFlush(newPhoto(photographerId, hash))

        assertFailsWith<DataIntegrityViolationException> {
            photoRepository.saveAndFlush(newPhoto(photographerId, hash))
        }
    }

    // The boundary is (photographer_id, content_hash), not content_hash alone —
    // two photographers shooting the same runner from the same angle would
    // otherwise block each other.
    @Test
    fun `a different photographer may store the same hash`() {
        val hash = randomHash()
        photoRepository.saveAndFlush(newPhoto(photographerId, hash))

        photoRepository.saveAndFlush(newPhoto(otherPhotographerId, hash))
    }

    // Rows that predate V24 have no hash at all. If the index covered NULLs,
    // the second of these would collide and any backfill would be impossible.
    @Test
    fun `rows with no hash never collide`() {
        photoRepository.saveAndFlush(newPhoto(photographerId, contentHash = null))

        photoRepository.saveAndFlush(newPhoto(photographerId, contentHash = null))
    }

    // ON DELETE SET NULL orphans a photo's photographer_id. Those rows must
    // drop out of the index rather than piling up against each other.
    @Test
    fun `orphaned rows never collide`() {
        val hash = randomHash()
        photoRepository.saveAndFlush(newPhoto(photographerId = null, contentHash = hash))

        photoRepository.saveAndFlush(newPhoto(photographerId = null, contentHash = hash))
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun randomHash(): String = UUID.randomUUID().toString().replace("-", "").repeat(2)

    private fun newPhoto(photographerId: UUID?, contentHash: String?) = Photo(
        eventId = eventId,
        photographerId = photographerId,
        s3Key = "test/${UUID.randomUUID()}.jpg",
        contentHash = contentHash,
        pricePhp = BigDecimal("125.00"),
    )

    private fun newEvent() = Event(
        slug = "dedup-index-test-${UUID.randomUUID()}",
        name = "Dedup Index Test Run",
        date = LocalDate.now(),
        location = "Cebu City",
        status = EventStatus.ACTIVE,
    )

    private fun newPhotographer() = User(
        email = "shooter-${UUID.randomUUID()}@test.local",
        passwordHash = "\$2a\$12\$stub",
        name = "Test Photographer",
        role = Role.PHOTOGRAPHER,
    )
}
