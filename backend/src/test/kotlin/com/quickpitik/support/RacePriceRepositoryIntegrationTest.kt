package com.quickpitik.support

import com.quickpitik.entity.EventPhotographerId
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/** AA-R11: V97 and the race price queries on real Postgres. */
class RacePriceRepositoryIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var events: EventRepository
    @Autowired private lateinit var coverage: EventPhotographerRepository

    private fun id(sql: String, vararg args: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *args)!!

    private fun photographer() = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Price Test', 'PHOTOGRAPHER')",
        "price-${UUID.randomUUID()}@test.local")

    private fun race() = id("INSERT INTO events (slug, name, date, location, status, price_per_photo, event_type) " +
        "VALUES (?, 'Price Run', current_date + 10, 'Cebu', 'ACTIVE', 125, 'ROAD_RACE')", "price-${UUID.randomUUID()}")

    private fun photo(event: UUID, photographer: UUID) =
        id("INSERT INTO photos (event_id, photographer_id, s3_key, price_php) VALUES (?, ?, ?, 125)", event, photographer, "rp/${UUID.randomUUID()}")

    private fun price(photo: UUID): BigDecimal = jdbc.queryForObject("SELECT price_php FROM photos WHERE id = ?", BigDecimal::class.java, photo)!!

    @Test
    fun `V97 adds the price columns and the history table`() {
        val columns = jdbc.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'event_photographer' AND column_name IN ('price_php', 'price_changed_at')",
            String::class.java,
        )
        assertEquals(setOf("price_php", "price_changed_at"), columns.toSet())
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM information_schema.tables WHERE table_name = 'race_price_changes'", Int::class.java))
    }

    @Test
    fun `the re-price touches only this photographer's photos in this race`() {
        val (me, other) = photographer() to photographer()
        val (raceA, raceB) = race() to race()
        val mine = List(2) { photo(raceA, me) }
        val theirs = photo(raceA, other)
        val myOtherRace = photo(raceB, me)

        val changed = tx.execute { photos.updatePriceByEventIdAndPhotographerId(raceA, me, BigDecimal("200.00")) }

        assertEquals(2, changed)
        mine.forEach { assertEquals(0, BigDecimal("200.00").compareTo(price(it))) }
        assertEquals(0, BigDecimal("125").compareTo(price(theirs)))
        assertEquals(0, BigDecimal("125").compareTo(price(myOtherRace)))
    }

    @Test
    fun `the locks and the scalar floor read work on real rows`() {
        val me = photographer()
        val race = race()
        val ids = List(3) { photo(race, me) }
        photo(race, photographer())
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id) VALUES (?, ?)", race, me)

        tx.execute {
            // Postgres orders uuid by bytes (= its hex text); Kotlin's UUID.compareTo uses signed longs.
            assertEquals(ids.sortedBy { it.toString() }, photos.lockIdsByEventIdAndPhotographerIdForUpdate(race, me))
            assertNotNull(coverage.findByIdForUpdate(EventPhotographerId(race, me)))
            assertEquals(0, BigDecimal("125").compareTo(events.findPricePerPhoto(race)))
        }
    }
}
