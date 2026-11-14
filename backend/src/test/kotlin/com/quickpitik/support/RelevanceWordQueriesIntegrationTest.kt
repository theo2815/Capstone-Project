package com.quickpitik.support

import com.quickpitik.repository.BatchRelevanceRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.util.UUID
import kotlin.test.assertEquals

// RV-1.8a (plan 5.9, G13, P13): the words that go into the summary, against real Postgres. The container is shared,
// so fixtures sit in 2003 and every query is scoped to this class's own events and batches.
class RelevanceWordQueriesIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    @Test
    fun `top words leave out words under the share`() {
        val event = event()
        // 34 counted photos: 3% of them is 1.02 photos, so a word on one photo is under the share and one on two is not.
        val b = batch(event, counted = 34, words = mapOf("COMMON" to 34, "BANNER" to 20, "PAIR" to 2, "RARE" to 1))

        assertEquals(listOf("COMMON" to 34, "BANNER" to 20, "PAIR" to 2), batches.topWords(b, 3, 30).map { it.word to it.photos })
        assertEquals(listOf("COMMON", "BANNER"), batches.topWords(b, 3, 2).map { it.word }, "the most common first, up to the limit")
        assertEquals(listOf("COMMON", "BANNER"), batches.topWords(b, 50, 30).map { it.word }, "50% of 34 is 17 photos")
    }

    @Test
    fun `event words count the other photographers' batches only`() {
        val event = event()
        val mine = batch(event, counted = 10, words = mapOf("CEBU" to 10, "SUGBO" to 10))
        // Ten photos of one word count as one photographer.
        batch(event, counted = 10, words = mapOf("CEBU" to 10))
        // CEBU on 1 of 50 photos is under 3% of this batch: not counted for it.
        batch(event, counted = 50, words = mapOf("CEBU" to 1, "SUGBO" to 50))
        batch(event, counted = 5, words = emptyMap())
        // Another event's batch never counts.
        batch(event(), counted = 10, words = mapOf("CEBU" to 10, "SUGBO" to 10))

        val counts = batches.eventWords(event, mine, arrayOf("CEBU", "SUGBO", "MISSING"), 3)
            .associate { it.word to it.photographers }
        assertEquals(mapOf("CEBU" to 1, "SUGBO" to 1), counts)
        assertEquals(3, batches.otherPhotographers(event, mine))
    }

    private fun event(): UUID = id(
        "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2003-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
        "relwq-${UUID.randomUUID()}",
    )

    // A batch of [counted] photos whose words were counted, with the given per-batch word counts.
    private fun batch(event: UUID, counted: Int, words: Map<String, Int>): UUID {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "relwq-${UUID.randomUUID()}@test.local",
        )
        tx.execute { batches.ensureBatch(event, photographer) }
        val batch = batches.findByEventIdAndPhotographerId(event, photographer)!!.id
        jdbc.update(
            "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) " +
                "SELECT ?, ?, 'relwq/' || gen_random_uuid(), 125, 'LIVE' FROM generate_series(1, ?)",
            event, photographer, counted,
        )
        jdbc.update(
            "INSERT INTO relevance_photo_checks (photo_id, batch_id, words_counted_at) " +
                "SELECT id, ?, TIMESTAMPTZ '2003-03-04 05:06:07+00' FROM photos WHERE event_id = ? AND photographer_id = ?",
            batch, event, photographer,
        )
        words.forEach { (w, n) -> jdbc.update("INSERT INTO batch_relevance_words (batch_id, word, photos) VALUES (?, ?, ?)", batch, w, n) }
        return batch
    }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!
}
