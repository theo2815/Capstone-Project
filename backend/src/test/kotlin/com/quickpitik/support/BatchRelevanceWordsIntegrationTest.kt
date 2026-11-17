package com.quickpitik.support

import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.relevance.RelevanceWordWriter
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// RV-1.2: the words kept from the text read, against real Postgres (plan 5.5). The container is shared, so
// fixtures sit in 2003 and assertions look only at this class's own rows.
class BatchRelevanceWordsIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    private val writer by lazy {
        RelevanceWordWriter(batches, RelevanceSwitch(RelevanceProperties(mode = RelevanceMode.RECORD_ONLY), AdminProperties(),
            AiApiProperties()), tx, Clock.fixed(NOW.toInstant(), ZoneOffset.UTC))
    }

    @Test
    fun `a photo read twice is counted once`() {
        val b = batch()
        val p = photo(b)
        writer.record(p, b.event, b.photographer, listOf("Cebu", "Marathon"))
        writer.record(p, b.event, b.photographer, listOf("Cebu", "Marathon"))

        assertEquals(mapOf("CEBU" to 1, "MARATHON" to 1), words(b))
        assertEquals(1L, batches.countWordPhotos(b.id))
        assertEquals(NOW.toInstant(), (jdbc.queryForObject("SELECT words_counted_at FROM relevance_photo_checks WHERE photo_id = ?",
            java.sql.Timestamp::class.java, p))!!.toInstant())
    }

    @Test
    fun `two photos with the same word make a count of two`() {
        val b = batch()
        writer.record(photo(b), b.event, b.photographer, listOf("Cebu", "Marathon"))
        writer.record(photo(b), b.event, b.photographer, listOf("CEBU", "Sugbo"))

        assertEquals(mapOf("CEBU" to 2, "MARATHON" to 1, "SUGBO" to 1), words(b))
        assertEquals(2L, batches.countWordPhotos(b.id))
    }

    @Test
    fun `rare words are purged once, settle-days after the final scoring, and common words stay`() {
        val b = batch()
        // 34 counted photos: 3% of them is 1.02 photos, so a word on one photo is rare and a word on two is not.
        repeat(34) { i -> writer.record(photo(b), b.event, b.photographer, listOfNotNull("COMMON", "RARE".takeIf { i == 0 },
            "PAIR".takeIf { i < 2 })) }
        settle(b, NOW.minusDays(8))

        assertEquals(1, purge(), "one batch of this class purged")
        assertEquals(mapOf("COMMON" to 34, "PAIR" to 2), words(b))
        assertNotNull(purgedAt(b))

        // Once: a word that arrives after the purge is left alone.
        jdbc.update("INSERT INTO batch_relevance_words (batch_id, word, photos) VALUES (?, 'LATE', 1)", b.id)
        assertEquals(0, purge())
        assertEquals(mapOf("COMMON" to 34, "PAIR" to 2, "LATE" to 1), words(b))
    }

    @Test
    fun `words are not purged before the settle days have passed`() {
        val b = batch()
        repeat(34) { i -> writer.record(photo(b), b.event, b.photographer, listOfNotNull("COMMON", "RARE".takeIf { i == 0 })) }
        settle(b, NOW.minusDays(6))
        val unscored = batch().also { writer.record(photo(it), it.event, it.photographer, listOf("RARE")) }

        assertEquals(0, purge())
        assertEquals(mapOf("COMMON" to 34, "RARE" to 1), words(b))
        assertNull(purgedAt(b))
        assertEquals(mapOf("RARE" to 1), words(unscored), "never finally scored: never purged")
    }

    @Test
    fun `an erased photo loses its check row`() {
        val b = batch()
        val (erased, kept) = List(2) { photo(b) }
        writer.record(erased, b.event, b.photographer, listOf("CEBU"))
        writer.record(kept, b.event, b.photographer, listOf("CEBU"))

        assertEquals(1, tx.execute { batches.deletePhotoCheck(erased) })
        assertEquals(listOf(kept), jdbc.queryForList("SELECT photo_id FROM relevance_photo_checks WHERE batch_id = ?",
            UUID::class.java, b.id))
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun batch(): Batch {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "relwords-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Run', DATE '2003-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relwords-${UUID.randomUUID()}",
        )
        tx.execute { batches.ensureBatch(event, photographer) }
        return Batch(batches.findByEventIdAndPhotographerId(event, photographer)!!.id, event, photographer)
    }

    private fun photo(b: Batch): UUID {
        val id = UUID.randomUUID()
        jdbc.update(
            "INSERT INTO photos (id, event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, ?, 125, 'LIVE')",
            id, b.event, b.photographer, "relwords/$id",
        )
        return id
    }

    private fun settle(b: Batch, at: OffsetDateTime) =
        jdbc.update("UPDATE batch_relevance SET final_scored_at = ? WHERE id = ?", at, b.id)

    // Other classes' settled batches share the container, so the limit is wide and only this class's rows count.
    private fun purge(): Int {
        val mine = jdbc.queryForList("SELECT br.id FROM batch_relevance br JOIN users u ON u.id = br.photographer_id " +
            "WHERE u.email LIKE 'relwords-%' AND br.words_purged_at IS NULL", UUID::class.java).toSet()
        tx.execute { batches.purgeRareWords(3, NOW.minusDays(7), 100_000, NOW) }
        val left = jdbc.queryForList("SELECT id FROM batch_relevance WHERE words_purged_at IS NULL", UUID::class.java).toSet()
        return (mine - left).size
    }

    private fun purgedAt(b: Batch): Any? = jdbc.queryForObject("SELECT words_purged_at FROM batch_relevance WHERE id = ?",
        Any::class.java, b.id)

    private fun words(b: Batch): Map<String, Int> =
        jdbc.queryForList("SELECT word, photos FROM batch_relevance_words WHERE batch_id = ?", b.id)
            .associate { it["word"] as String to (it["photos"] as Number).toInt() }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2003-03-04T05:06:07.123456Z")
    }
}
