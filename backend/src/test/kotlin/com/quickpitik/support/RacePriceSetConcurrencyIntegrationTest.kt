package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.events.RacePriceService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.TestPropertySource
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.sql.DataSource
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** AA-R11 (review F10): the race price is checked under the locks, against the floor read after them. */
@TestPropertySource(properties = ["app.races.self-serve-enabled=true"])
class RacePriceSetConcurrencyIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var dataSource: DataSource
    @Autowired private lateinit var prices: RacePriceService
    @Autowired private lateinit var tx: TransactionTemplate
    @Autowired private lateinit var coverage: EventPhotographerRepository
    @Autowired private lateinit var photos: PhotoRepository

    private fun id(sql: String, vararg args: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *args)!!

    // An approved photographer covering a race with one photo in it.
    private fun coveredRace(): Pair<UUID, UUID> {
        val me = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Price Test', 'PHOTOGRAPHER')",
            "price-${UUID.randomUUID()}@test.local")
        jdbc.update("INSERT INTO photographer_settings (user_id, verification_status) VALUES (?, 'APPROVED')", me)
        val race = id("INSERT INTO events (slug, name, date, location, status, price_per_photo, event_type) " +
            "VALUES (?, 'Price Run', current_date + 10, 'Cebu', 'ACTIVE', 125, 'ROAD_RACE')", "price-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id) VALUES (?, ?)", race, me)
        jdbc.update("INSERT INTO photos (event_id, photographer_id, s3_key, price_php) VALUES (?, ?, ?, 125)", race, me, "rp/${UUID.randomUUID()}")
        return me to race
    }

    @Test
    fun `a floor change that commits first bounds my price`() {
        val (me, race) = coveredRace()
        val pool = Executors.newSingleThreadExecutor()
        // The admin path (AA-R12) locks the race's photos first, lowers the floor to 100, and holds.
        val admin = dataSource.connection.apply { autoCommit = false }
        try {
            admin.prepareStatement("SELECT id FROM photos WHERE event_id = ? ORDER BY id FOR NO KEY UPDATE").use {
                it.setObject(1, race); it.executeQuery().close()
            }
            admin.prepareStatement("UPDATE events SET price_per_photo = 100 WHERE id = ?").use { it.setObject(1, race); it.executeUpdate() }
            val put = pool.submit<BigDecimal> { prices.setMyPrice(me, race, BigDecimal("250")).effectivePhp }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
            while (waitingOnALock() < 1) {
                check(System.nanoTime() < deadline) { "the price change never waited on the photo locks" }
                Thread.sleep(20)
            }
            admin.commit()

            val error = assertFailsWith<ExecutionException> { put.get(30, TimeUnit.SECONDS) }.cause
            assertTrue(error is ValidationException && error.field == "pricePhp", "expected a 400 against the new ceiling, got $error")
            assertTrue("200.00" in error.message!!, error.message)
        } finally {
            admin.close()
            pool.shutdownNow()
        }
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM race_price_changes WHERE event_id = ?", Int::class.java, race))
    }

    @Test
    fun `two parallel price changes store one and refuse the other with the cooldown`() {
        val (me, race) = coveredRace()
        val start = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        val results = try {
            val futures = listOf("150", "200").map { value ->
                pool.submit<Any> {
                    start.await()
                    runCatching { prices.setMyPrice(me, race, BigDecimal(value)) }.fold({ it }, { it })
                }
            }
            start.countDown()
            futures.map { it.get(30, TimeUnit.SECONDS) }
        } finally {
            pool.shutdownNow()
        }

        val refused = results.filterIsInstance<ConflictException>()
        assertEquals(1, refused.size, "results: $results")
        assertEquals(ErrorCodes.RACE_PRICE_COOLDOWN, refused.single().code)
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM race_price_changes WHERE event_id = ?", Int::class.java, race))
    }

    @Test
    fun `a price change during an upload leaves the new photo at the new price`() {
        val (me, race) = coveredRace()
        val priceHeld = CountDownLatch(1)
        val releasePrice = CountDownLatch(1)
        val pool = Executors.newFixedThreadPool(2)
        val newPhoto = UUID.randomUUID()
        try {
            // Thread A: the photographer's price change, holding its row lock open.
            val change = pool.submit {
                tx.execute {
                    prices.setMyPrice(me, race, BigDecimal("200"))
                    priceHeld.countDown()
                    check(releasePrice.await(20, TimeUnit.SECONDS)) { "price change was not released" }
                }
            }
            check(priceHeld.await(20, TimeUnit.SECONDS)) { "price change did not take its locks" }
            // Thread B: persistNew's transaction, the photo seeded at the floor read before the change.
            val upload = pool.submit {
                tx.execute {
                    jdbc.update("INSERT INTO photos (id, event_id, photographer_id, s3_key, price_php) VALUES (?, ?, ?, ?, 125)",
                        newPhoto, race, me, "rp/${UUID.randomUUID()}")
                    coverage.upsertOnUpload(race, me, OffsetDateTime.now())
                    photos.reseedRacePrice(newPhoto)
                }
            }
            val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20)
            while (waitingOnALock() < 1) {
                check(System.nanoTime() < deadline) { "the upload never waited on the coverage row lock" }
                Thread.sleep(20)
            }
            releasePrice.countDown()
            change.get(30, TimeUnit.SECONDS)
            upload.get(30, TimeUnit.SECONDS)
        } finally {
            releasePrice.countDown()
            pool.shutdownNow()
        }
        assertEquals(BigDecimal("200.00"), jdbc.queryForObject("SELECT price_php FROM photos WHERE id = ?", BigDecimal::class.java, newPhoto))
    }

    private fun waitingOnALock(): Int = jdbc.queryForObject(
        "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' AND datname = current_database()",
        Int::class.java,
    )!!
}
