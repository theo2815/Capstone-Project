package com.quickpitik.support

import com.quickpitik.repository.EventPhotographerRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import java.util.UUID
import kotlin.test.assertEquals

/** AA-R14: the "price varies" query on real Postgres. */
class RacePriceRaisedQueryIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var coverage: EventPhotographerRepository

    private fun id(sql: String, vararg args: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *args)!!

    private fun photographer() = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Raised Test', 'PHOTOGRAPHER')",
        "raised-${UUID.randomUUID()}@test.local")

    private fun race() = id("INSERT INTO events (slug, name, date, location, status, price_per_photo, event_type) " +
        "VALUES (?, 'Raised Run', current_date + 10, 'Cebu', 'ACTIVE', 125, 'ROAD_RACE')", "raised-${UUID.randomUUID()}")

    private fun cover(event: UUID, price: Int?, photos: Int) {
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, price_php, photo_count) VALUES (?, ?, ?, ?)",
            event, photographer(), price, photos)
    }

    @Test
    fun `raised price query finds only events with a raised row that has photos`() {
        val raised = race().also { cover(it, 180, 4); cover(it, null, 9) }
        val raisedNoPhotos = race().also { cover(it, 200, 0) }
        val floorOnly = race().also { cover(it, null, 3) }
        val notAsked = race().also { cover(it, 150, 2) }

        val found = coverage.findEventIdsWithRaisedPrice(listOf(raised, raisedNoPhotos, floorOnly))

        assertEquals(listOf(raised), found)
        assertEquals(false, notAsked in found)
    }
}
