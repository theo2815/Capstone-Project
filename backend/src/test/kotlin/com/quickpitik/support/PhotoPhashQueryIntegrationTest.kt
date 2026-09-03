package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.random.Random
import kotlin.test.assertEquals

/**
 * `PhotoRepository.findNearestByPhash` (V42), executed against real Postgres.
 *
 * The query is native SQL — `bit_count(CAST((phash # :h) AS bit(64)))` — and
 * nothing in the Mockito suite ever binds it. This proves the cast syntax,
 * the parameter binding, the LIVE filter, and that a negative bigint (sign
 * bit set) survives the round trip through bit(64).
 */
class PhotoPhashQueryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var photoRepository: PhotoRepository

    @Autowired
    private lateinit var eventRepository: EventRepository

    @Autowired
    private lateinit var userRepository: UserRepository

    @Test
    fun `nearest fingerprint is the LIVE row with the smallest hamming distance`() {
        val eventId = eventRepository.save(newEvent()).id
        val exact = userRepository.save(newPhotographer()).id
        val near = userRepository.save(newPhotographer()).id
        val hidden = userRepository.save(newPhotographer()).id
        // Sign bit forced on: the case a naive cast gets wrong.
        val base = Random.nextLong() or Long.MIN_VALUE

        photoRepository.saveAndFlush(newPhoto(eventId, exact, base, PhotoStatus.LIVE))
        photoRepository.saveAndFlush(newPhoto(eventId, near, base xor 0b111L, PhotoStatus.LIVE))
        photoRepository.saveAndFlush(newPhoto(eventId, hidden, base, PhotoStatus.HIDDEN))

        val row = photoRepository.findNearestByPhash(base).single()
        assertEquals(exact, row[0])
        assertEquals(eventId, row[1])
        assertEquals(0, (row[2] as Number).toInt())

        val offByOne = photoRepository.findNearestByPhash(base xor 1L).single()
        assertEquals(exact, offByOne[0])
        assertEquals(1, (offByOne[2] as Number).toInt())
    }

    // V43: a copy cropped to the runner hashes like the registered centre
    // crop, not like the marked frame — the query must take the best of the
    // three columns, and rows without the V43 columns must still rank by phash.
    @Test
    fun `a centre-crop hash wins over a far marked hash`() {
        val eventId = eventRepository.save(newEvent()).id
        val marked = userRepository.save(newPhotographer()).id
        val cropped = userRepository.save(newPhotographer()).id
        val base = Random.nextLong() or Long.MIN_VALUE

        photoRepository.saveAndFlush(newPhoto(eventId, marked, base xor 0b11L, PhotoStatus.LIVE))
        photoRepository.saveAndFlush(
            newPhoto(eventId, cropped, base.inv(), PhotoStatus.LIVE).also { it.phashCentre = base },
        )

        val row = photoRepository.findNearestByPhash(base).single()
        assertEquals(cropped, row[0])
        assertEquals(0, (row[2] as Number).toInt())
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun newPhoto(eventId: UUID, photographerId: UUID, phash: Long, status: PhotoStatus) = Photo(
        eventId = eventId,
        photographerId = photographerId,
        s3Key = "test/${UUID.randomUUID()}.jpg",
        pricePhp = BigDecimal("125.00"),
        status = status,
    ).also { it.phash = phash }

    private fun newEvent() = Event(
        slug = "phash-query-test-${UUID.randomUUID()}",
        name = "Phash Query Test Run",
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
