package com.quickpitik.common

import com.quickpitik.entity.Photo
import com.quickpitik.entity.PhotoBibEmbed
import com.quickpitik.entity.PhotoSpan
import com.quickpitik.entity.PhotoStatus
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import org.slf4j.LoggerFactory
import org.springframework.boot.ApplicationArguments
import org.springframework.boot.ApplicationRunner
import org.springframework.stereotype.Component
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID

@Component
class BootstrapPhotosRunner(
    private val eventRepository: EventRepository,
    private val photoRepository: PhotoRepository,
) : ApplicationRunner {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun run(args: ApplicationArguments?) {
        // Fetch all events
        val events = eventRepository.findAll()
        if (events.isEmpty()) {
            log.info("Mock photos bootstrap skipped — no events exist in database")
            return
        }

        // Check if there are already photos seeded
        if (photoRepository.count() > 0) {
            log.info("Mock photos bootstrap skipped — photos already exist in database")
            return
        }

        val event = events.first()
        log.info("Seeding 5 mock photos for event: {} ({})", event.name, event.id)

        val mockPhotos = listOf(
            Triple("2948", BigDecimal("5.00"), "Cebu Marathon 2026 Runner #2948 - Leading the Pack"),
            Triple("1234", BigDecimal("10.00"), "Cebu Marathon 2026 Runner #1234 - High-fiving Supporters"),
            Triple("4201", BigDecimal("21.10"), "Cebu Marathon 2026 Runner #4201 - Reaching the Half Marathon Mark"),
            Triple("1111", BigDecimal("42.20"), "Cebu Marathon 2026 Runner #1111 - Crossing the Finish Line in Cebu"),
            Triple("2948", BigDecimal("42.20"), "Cebu Marathon 2026 Runner #2948 - Crossing the Finish Line in Cebu")
        )

        mockPhotos.forEachIndexed { index, (bib, km, alt) ->
            val photo = Photo(
                id = UUID.randomUUID(),
                eventId = event.id,
                s3Key = "photos/mock_runner_${index + 1}.jpg",
                thumbnailS3Key = "photos/mock_runner_${index + 1}_thumb.jpg",
                watermarkS3Key = "photos/mock_runner_${index + 1}_watermark.jpg",
                pricePhp = BigDecimal("150.00"),
                status = PhotoStatus.LIVE,
                spanWire = PhotoSpan.DEFAULT.wire,
                tone = 0,
                km = km,
                capturedAt = OffsetDateTime.now().minusMinutes((120 - index * 10).toLong()),
                uploadedAt = OffsetDateTime.now(),
                altText = alt
            )
            photo.bibs.add(PhotoBibEmbed(bib, BigDecimal("0.99")))
            photoRepository.save(photo)
        }

        // Update photo count in the event
        eventRepository.save(event.apply {
            photoCount = mockPhotos.size
        })

        log.info("Successfully seeded 5 premium mock runner photos with bib tags!")
    }
}
