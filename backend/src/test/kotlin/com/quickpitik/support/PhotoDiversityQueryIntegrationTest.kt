package com.quickpitik.support

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginationParams
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
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals

class PhotoDiversityQueryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var photoRepository: PhotoRepository

    @Autowired
    private lateinit var eventRepository: EventRepository

    @Autowired
    private lateinit var userRepository: UserRepository

    @Test
    fun `gallery represents every photographer before repeating one`() {
        val eventId = eventRepository.save(newEvent()).id
        val photographers = List(3) { userRepository.save(newPhotographer()).id }
        val snapshot = OffsetDateTime.now(ZoneOffset.UTC).withNano(0)
        val counts = listOf(4, 3, 1)

        photographers.zip(counts).forEachIndexed { photographerIndex, (photographerId, count) ->
            repeat(count) { photoIndex ->
                photoRepository.save(
                    newPhoto(
                        eventId = eventId,
                        photographerId = photographerId,
                        capturedAt = snapshot.minusHours(photographerIndex.toLong()).minusMinutes(photoIndex.toLong()),
                        publishedAt = snapshot.minusMinutes(1),
                    ),
                )
            }
        }
        photoRepository.saveAndFlush(
            newPhoto(
                eventId = eventId,
                photographerId = photographers.first(),
                capturedAt = snapshot.plusHours(1),
                publishedAt = snapshot.plusMinutes(1),
            ),
        )

        val page = photoRepository.findForEventNoBib(
            eventId = eventId,
            snapshotAt = snapshot,
            seed = snapshot.toEpochSecond(),
            pageable = OffsetLimitPageable(PaginationParams.of(0, 5)),
        )
        val ids = page.content.map { it.photographerId }

        assertEquals(8, page.totalElements)
        assertEquals(photographers.toSet(), ids.take(3).toSet())
        assertEquals(3, ids.take(3).distinct().size)
        assertEquals(photographers.take(2).toSet(), ids.drop(3).take(2).toSet())
    }

    private fun newPhoto(
        eventId: UUID,
        photographerId: UUID,
        capturedAt: OffsetDateTime,
        publishedAt: OffsetDateTime,
    ) = Photo(
        eventId = eventId,
        photographerId = photographerId,
        s3Key = "test/${UUID.randomUUID()}.jpg",
        pricePhp = BigDecimal("125.00"),
        status = PhotoStatus.LIVE,
        capturedAt = capturedAt,
        uploadedAt = capturedAt,
        publishedAt = publishedAt,
    )

    private fun newEvent() = Event(
        slug = "photo-diversity-${UUID.randomUUID()}",
        name = "Photo Diversity Test Run",
        date = LocalDate.now(),
        location = "Cebu City",
        status = EventStatus.ACTIVE,
    )

    private fun newPhotographer() = User(
        email = "diversity-${UUID.randomUUID()}@test.local",
        passwordHash = "\$2a\$12\$stub",
        name = "Test Photographer",
        role = Role.PHOTOGRAPHER,
    )
}
