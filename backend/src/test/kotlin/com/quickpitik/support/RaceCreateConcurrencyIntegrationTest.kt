package com.quickpitik.support

import com.quickpitik.dto.photographer.CreateRaceRequest
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.RACE_CREATION_LOCK_KEY
import com.quickpitik.service.events.RaceEventService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.TestPropertySource
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.sql.DataSource
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** AA-R3: the race-creation advisory lock and the new race queries, on real Postgres. */
@TestPropertySource(properties = ["app.races.self-serve-enabled=true"])
class RaceCreateConcurrencyIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var dataSource: DataSource
    @Autowired private lateinit var races: RaceEventService
    @Autowired private lateinit var events: EventRepository

    // An approved photographer whose identity passed (AUTO_APPROVED seller details).
    private fun photographer(): UUID {
        val id = jdbc.queryForObject(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Race Test', 'PHOTOGRAPHER') RETURNING id",
            UUID::class.java, "race-${UUID.randomUUID()}@test.local",
        )!!
        jdbc.update("INSERT INTO photographer_settings (user_id, verification_status) VALUES (?, 'APPROVED')", id)
        jdbc.update(
            "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, address, phone, " +
                "phone_key, status, revision, submitted_at) VALUES (?, 'Juan Dela Cruz', 'PASSPORT', 'seller-details/x/f.jpg', " +
                "'seller-details/x/s.jpg', 'Cebu', '09171234567', ?, 'AUTO_APPROVED', 1, now())",
            id, UUID.randomUUID().toString().take(10),
        )
        return id
    }

    private fun request(title: String, date: LocalDate) = CreateRaceRequest(
        title = title, date = date.toString(), location = "IT Park, Cebu City",
        regionCode = "region-7", provinceCode = "cebu", eventType = "ROAD_RACE",
    )

    @Test
    fun `two photographers creating the same race at once leave one row`() {
        val (a, b) = photographer() to photographer()
        // A unique title per run, so other classes' rows never count.
        val title = "Lock Test Marathon ${UUID.randomUUID().toString().take(8).filter(Char::isLetter).ifEmpty { "x" }}"
        val date = LocalDate.now().plusDays(30)
        val pool = Executors.newFixedThreadPool(2)
        // Hold the race lock so both creates pass the unlocked pre-check and then queue on the lock.
        val pause = dataSource.connection.apply { autoCommit = false }
        try {
            pause.createStatement().use { it.execute("SELECT pg_advisory_xact_lock($RACE_CREATION_LOCK_KEY)") }
            val futures = listOf(a, b).map { id -> pool.submit<String> { races.create(id, request(title, date), null).outcome } }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
            while (waitingOnRaceLock() < 2) {
                check(System.nanoTime() < deadline) { "both creates never queued on the race lock" }
                Thread.sleep(20)
            }
            pause.rollback() // releases the lock: one create inserts, the other re-checks inside the lock

            assertEquals(listOf("created", "match"), futures.map { it.get(30, TimeUnit.SECONDS) }.sorted())
        } finally {
            pause.close()
            pool.shutdownNow()
        }
        val rows = jdbc.queryForList("SELECT submitted_by FROM events WHERE name = ?", UUID::class.java, title)
        assertEquals(1, rows.size)
        assertTrue(rows.single() in setOf(a, b))
        assertEquals(1L, events.countRacesSubmittedSince(rows.single(), OffsetDateTime.now().minusHours(1)))
        assertEquals(
            1,
            jdbc.queryForObject("SELECT count(*) FROM event_photographer ep JOIN events e ON e.id = ep.event_id WHERE e.name = ?",
                Int::class.java, title),
            "only the winner gets a coverage row",
        )
    }

    @Test
    fun `join adds coverage and an alias once, and the candidate query reads the alias`() {
        val (submitter, joiner) = photographer() to photographer()
        val date = LocalDate.now().plusDays(40)
        val title = "Alias Test Run ${UUID.randomUUID().toString().take(8).filter(Char::isLetter).ifEmpty { "x" }}"
        val race = races.create(submitter, request(title, date), null).event!!

        races.join(joiner, race.id, "Sugbo Alias Night Run")
        races.join(joiner, race.id, "Sugbo Alias Night Run")

        val candidates = events.findRaceCandidates(date.minusDays(1), date.plusDays(1)).map { it.id }
        assertTrue(race.id in candidates)
        assertEquals(listOf("alias night run sugbo"), events.findAliasKeys(listOf(race.id)).map { it.getNameKey() })
        assertEquals(
            2,
            jdbc.queryForObject("SELECT count(*) FROM event_photographer WHERE event_id = ?", Int::class.java, race.id),
            "submitter and joiner, once each",
        )
    }

    private fun waitingOnRaceLock(): Int = jdbc.queryForObject(
        "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND objid::bigint = ? AND NOT granted",
        Int::class.java, RACE_CREATION_LOCK_KEY,
    )!!
}
