package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals

// AX-11: the admin Overview trend's uploads series counts the photos committed each Manila day.
class PhotoUploadTrendIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var users: UserRepository

    @Autowired
    private lateinit var events: EventRepository

    @Autowired
    private lateinit var photos: PhotoRepository

    @Autowired
    private lateinit var jdbc: JdbcTemplate

    @Autowired
    private lateinit var transactions: PlatformTransactionManager

    private val manila = ZoneId.of("Asia/Manila")

    @Test
    fun `uploads are counted per Manila day, only inside the window`() {
        val photographer = users.save(
            User(email = "trend-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Trend", role = Role.PHOTOGRAPHER),
        )
        val event = events.save(
            Event(slug = "trend-${UUID.randomUUID()}", name = "Trend", date = LocalDate.of(2001, 2, 3), location = "Cebu", status = EventStatus.ACTIVE),
        )
        // 2001: a year no other test uploads in, because the container is shared across classes.
        fun utc(day: Int, hour: Int) = OffsetDateTime.of(2001, 2, day, hour, 0, 0, 0, ZoneOffset.UTC)
        listOf(
            utc(2, 17), // 01:00 Manila on 3 Feb, still 2 Feb in UTC
            utc(3, 4), utc(3, 4), // noon Manila, 3 Feb
            utc(3, 17), // 01:00 Manila on 4 Feb, a race-morning upload
            utc(4, 16), // 00:00 Manila on 5 Feb, exactly where the window ends
            utc(15, 4), // after the window
        ).forEach {
            photos.save(Photo(eventId = event.id, photographerId = photographer.id, s3Key = "trend/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("100.00"), uploadedAt = it))
        }
        val from = LocalDate.of(2001, 2, 3).atStartOfDay(manila).toOffsetDateTime()
        val to = LocalDate.of(2001, 2, 5).atStartOfDay(manila).toOffsetDateTime()

        val byDay = TransactionTemplate(transactions).execute {
            // The servers (CI, Railway) run in UTC; pin it so this machine's own zone cannot hide a UTC day cut.
            jdbc.execute("SET LOCAL TIME ZONE 'UTC'")
            photos.uploadCountsByDayBetween(from, to, manila.id).associate { it[0].toString() to (it[1] as Number).toLong() }
        }

        assertEquals(mapOf("2001-02-03" to 3L, "2001-02-04" to 1L), byDay)
    }
}
