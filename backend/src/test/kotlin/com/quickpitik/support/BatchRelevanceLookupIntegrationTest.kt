package com.quickpitik.support

import com.quickpitik.repository.BatchCounts
import com.quickpitik.repository.BatchRelevanceRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.random.Random
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertNotEquals
import kotlin.test.assertNull

// PE-1.3: the look-up statement and the copy counts against real Postgres. Every test builds its own
// users and events, and the look-up only reads one event, so nothing here assumes an empty table.
class BatchRelevanceLookupIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: TransactionTemplate

    private val base = Random.nextLong() or Long.MIN_VALUE

    @Test
    fun `a clean hash within four bits of an earlier photo by another photographer is a match on the clean pair`() {
        val e = event()
        val original = photo(e, photographer(), at(0), clean = base)
        val me = Batch(e, photographer())
        val copy = photo(e, me.photographer, at(1), clean = base xor 0b1111L)

        assertEquals(1, lookUp(me))
        assertMatch(copy, original, 4, "CLEAN")
    }

    @Test
    fun `a centre hash match is found when the clean hashes differ`() {
        val e = event()
        val centre = Random.nextLong()
        val original = photo(e, photographer(), at(0), clean = base.inv(), centre = centre)
        val me = Batch(e, photographer())
        val crop = photo(e, me.photographer, at(1), clean = base, centre = centre xor 1L)

        lookUp(me)
        assertMatch(crop, original, 1, "CENTRE")
    }

    @Test
    fun `a saved preview matches the other photo's marked hash`() {
        val e = event()
        val original = photo(e, photographer(), at(0), clean = base.inv(), marked = base)
        val me = Batch(e, photographer())
        val preview = photo(e, me.photographer, at(1), clean = base)

        lookUp(me)
        assertMatch(preview, original, 0, "MARKED")
    }

    // R2-61
    @Test
    fun `an older photo with only a marked hash still matches and one with no hashes is skipped`() {
        val e = event()
        photo(e, photographer(), at(0))
        val markedOnly = photo(e, photographer(), at(1), marked = base xor 0b11L)
        val me = Batch(e, photographer())
        val p = photo(e, me.photographer, at(2), clean = base)

        lookUp(me)
        assertMatch(p, markedOnly, 2, "MARKED")
    }

    @Test
    fun `a hash with the sign bit set is compared bit for bit`() {
        val e = event()
        val original = photo(e, photographer(), at(0), clean = base)
        val me = Batch(e, photographer())
        val p = photo(e, me.photographer, at(1), clean = base xor 0b111L)
        lookUp(me)
        assertMatch(p, original, 3, "CLEAN")

        val e2 = event()
        photo(e2, photographer(), at(0), clean = -1L)
        val me2 = Batch(e2, photographer())
        val far = photo(e2, me2.photographer, at(1), clean = 0L)
        assertEquals(1, lookUp(me2))
        assertNoMatch(far)
    }

    @Test
    fun `the earlier upload is never matched against the later one`() {
        val e = event()
        val first = Batch(e, photographer())
        val original = photo(e, first.photographer, at(0), clean = base)
        photo(e, photographer(), at(1), clean = base)

        assertEquals(1, lookUp(first))
        assertNoMatch(original)
    }

    // R2-50: the SQL is a strict <.
    @Test
    fun `two photos by different photographers with the same upload time give no hit`() {
        val e = event()
        photo(e, photographer(), at(0), clean = base)
        val me = Batch(e, photographer())
        val p = photo(e, me.photographer, at(0), clean = base)

        lookUp(me)
        assertNoMatch(p)
    }

    @Test
    fun `own photos, other events, trashed photos and photos with no photographer are never a match`() {
        val e = event()
        val me = Batch(e, photographer())
        photo(e, me.photographer, at(0), clean = base)
        photo(event(), photographer(), at(0), clean = base)
        photo(e, photographer(), at(0), clean = base, deleted = true)
        photo(e, null, at(0), clean = base)
        val p = photo(e, me.photographer, at(1), clean = base)

        assertEquals(2, lookUp(me), "both own photos are looked up")
        assertNoMatch(p)
    }

    // R2-59 (and PE-1.2's rescoped R2-11): a photo under a privacy removal review neither counts nor matches.
    @Test
    fun `a photo under a removal review is neither looked up nor a match`() {
        val e = event()
        val other = photographer()
        photo(e, other, at(0), clean = base, removal = "REVIEW")
        val me = Batch(e, photographer())
        val p = photo(e, me.photographer, at(1), clean = base)
        photo(e, me.photographer, at(2), clean = base.inv(), removal = "REVIEW")

        assertEquals(1, lookUp(me))
        assertNoMatch(p)
        assertEquals(0L, batches.countPendingLookups(e, me.photographer))
        assertEquals(1L, counts(me).photosLive)
    }

    @Test
    fun `distance six is stored as a weak match and distance eight is stored and distance nine is not`() {
        fun nearestAt(bits: Long): Pair<UUID, UUID> {
            val e = event()
            val original = photo(e, photographer(), at(0), clean = base)
            val me = Batch(e, photographer())
            val p = photo(e, me.photographer, at(1), clean = base xor bits)
            lookUp(me)
            return p to original
        }
        nearestAt(0b111111L).let { (p, o) -> assertMatch(p, o, 6, "CLEAN") }
        nearestAt(0b11111111L).let { (p, o) -> assertMatch(p, o, 8, "CLEAN") }
        nearestAt(0b111111111L).let { (p, _) -> assertNoMatch(p) }
    }

    @Test
    fun `the nearest match wins and ties go to the earlier upload`() {
        val e = event()
        photo(e, photographer(), at(0), clean = base xor 0b111L)
        val nearEarly = photo(e, photographer(), at(1), clean = base xor 0b11L)
        photo(e, photographer(), at(2), clean = base xor 0b1100L)
        val me = Batch(e, photographer())
        val p = photo(e, me.photographer, at(3), clean = base)

        lookUp(me)
        assertMatch(p, nearEarly, 2, "CLEAN")
    }

    @Test
    fun `the same file uploaded earlier by another photographer in another event is an exact match`() {
        val hash = sha()
        val original = photo(event(), photographer(), at(0), clean = base.inv(), contentHash = hash)
        val e = event()
        val me = Batch(e, photographer())
        val p = photo(e, me.photographer, at(1), clean = base, contentHash = hash)

        lookUp(me)
        assertEquals(original, row(p)["exact_photo_id"])
    }

    // The uploader's own earlier copy cannot exist: V24's unique index (photographer_id, content_hash)
    // refuses it, so it is not built here. A copy whose photographer is gone is the nearest case.
    @Test
    fun `a trashed copy and a copy with no photographer are not`() {
        val hash = sha()
        photo(event(), photographer(), at(0), contentHash = hash, deleted = true)
        photo(event(), null, at(0), contentHash = hash)
        val e = event()
        val me = Batch(e, photographer())
        val p = photo(e, me.photographer, at(1), clean = base, contentHash = hash)

        lookUp(me)
        assertNull(row(p)["exact_photo_id"])
    }

    @Test
    fun `processing photos and photos without a clean hash are not looked up`() {
        val e = event()
        val me = Batch(e, photographer())
        photo(e, me.photographer, at(0), clean = base, status = "PROCESSING")
        photo(e, me.photographer, at(1), marked = base)
        photo(e, me.photographer, at(2), clean = base, status = "HIDDEN")

        assertEquals(0L, batches.countPendingLookups(e, me.photographer))
        assertEquals(0, lookUp(me))
    }

    @Test
    fun `a second run inserts nothing and the limit is respected in upload order`() {
        val e = event()
        val me = Batch(e, photographer())
        val third = photo(e, me.photographer, at(2), clean = base)
        val first = photo(e, me.photographer, at(0), clean = base)
        val second = photo(e, me.photographer, at(1), clean = base)

        assertEquals(2, lookUp(me, limit = 2))
        assertEquals(setOf(first, second), lookedUp(me))
        assertEquals(1, lookUp(me, limit = 2))
        assertEquals(setOf(first, second, third), lookedUp(me))
        assertEquals(0, lookUp(me, limit = 2))
    }

    @Test
    fun `pending look-ups count down to zero`() {
        val e = event()
        val me = Batch(e, photographer())
        repeat(3) { photo(e, me.photographer, at(it.toLong()), clean = base) }

        assertEquals(3L, batches.countPendingLookups(e, me.photographer))
        lookUp(me, limit = 2)
        assertEquals(1L, batches.countPendingLookups(e, me.photographer))
        lookUp(me, limit = 2)
        assertEquals(0L, batches.countPendingLookups(e, me.photographer))
    }

    // R2-50: distance 4 is tight, 5 is weak.
    @Test
    fun `counts ignore trashed photos and split tight from weak at the given distances`() {
        val e = event()
        val me = Batch(e, photographer())
        val m = photo(e, photographer(), at(0))
        listOf(4, 5, 8).forEach { lookedUpRow(me, photo(e, me.photographer, at(1)), m, it) }
        lookedUpRow(me, photo(e, me.photographer, at(1), indexing = "INDEXED"), null, null, exact = m)
        lookedUpRow(me, photo(e, me.photographer, at(1), deleted = true), m, 0)
        photo(e, me.photographer, at(1), status = "PROCESSING")
        photo(e, me.photographer, at(1))

        assertEquals(
            Counts(photosLive = 5, photosProcessing = 1, indexedPhotos = 1, lookups = 4, tightHits = 1, weakHits = 2, exactMatches = 1),
            counts(me),
        )
        assertEquals(2L, counts(me, tight = 5).tightHits)
        assertEquals(1L, counts(me, tight = 5).weakHits)
        assertEquals(0L, counts(me, tight = 4, weak = 4).weakHits)
    }

    @Test
    fun `a looked-up photo that was hidden afterwards still counts`() {
        val e = event()
        val m = photo(e, photographer(), at(0), clean = base)
        val me = Batch(e, photographer())
        val p = photo(e, me.photographer, at(1), clean = base)
        lookUp(me)
        jdbc.update("UPDATE photos SET status = 'HIDDEN' WHERE id = ?", p)

        val c = counts(me)
        assertEquals(0L, c.photosLive)
        assertEquals(1L, c.lookups)
        assertEquals(1L, c.tightHits)
        assertEquals(m, row(p)["match_photo_id"])
    }

    // R2-23: each look-up transaction runs with a statement timeout that ends with the transaction.
    @Test
    fun `a local statement timeout cancels a slow statement and ends with its transaction`() {
        val shown = tx.execute {
            batches.setLocalStatementTimeout(5_000)
            jdbc.queryForObject("SHOW statement_timeout", String::class.java)
        }
        assertEquals("5s", shown)
        assertNotEquals("5s", jdbc.queryForObject("SHOW statement_timeout", String::class.java))

        val failure = assertFails {
            tx.execute {
                batches.setLocalStatementTimeout(50)
                jdbc.queryForObject("SELECT pg_sleep(2)::text", String::class.java)
            }
        }
        assertContains(generateSequence(failure) { it.cause }.joinToString { it.message.orEmpty() }, "statement timeout")
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private inner class Batch(val event: UUID, val photographer: UUID) {
        val id: UUID = id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer)
    }

    private data class Counts(
        val photosLive: Long, val photosProcessing: Long, val indexedPhotos: Long, val lookups: Long,
        val tightHits: Long, val weakHits: Long, val exactMatches: Long,
    )

    private fun counts(b: Batch, tight: Int = 4, weak: Int = 8): Counts =
        batches.batchCounts(b.event, b.photographer, tight, weak).let {
            Counts(it.photosLive, it.photosProcessing, it.indexedPhotos, it.lookups, it.tightHits, it.weakHits, it.exactMatches)
        }

    private fun lookUp(b: Batch, limit: Int = 100, weakMax: Int = 8): Int =
        tx.execute { batches.lookUpNext(b.id, b.event, b.photographer, limit, weakMax, NOW) }!!

    private fun photographer(): UUID = id(
        "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
        "rel-${UUID.randomUUID()}@test.local",
    )

    private fun event(): UUID = id(
        "INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Rel Run', ?, 'Cebu', 'ACTIVE')",
        "rel-${UUID.randomUUID()}", LocalDate.of(2002, 6, 9),
    )

    private fun photo(
        event: UUID,
        photographer: UUID?,
        uploadedAt: OffsetDateTime,
        clean: Long? = null,
        centre: Long? = null,
        marked: Long? = null,
        status: String = "LIVE",
        deleted: Boolean = false,
        removal: String = "NONE",
        contentHash: String? = null,
        indexing: String = "PENDING",
    ): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, uploaded_at, status, deleted_at, " +
            "removal_state, content_hash, indexing_status, phash, phash_clean, phash_centre) " +
            "VALUES (?, ?, ?, 125, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        event, photographer, "rel/${UUID.randomUUID()}", uploadedAt, status, if (deleted) NOW else null,
        removal, contentHash, indexing, marked, clean, centre,
    )

    private fun lookedUpRow(b: Batch, photo: UUID, match: UUID?, distance: Int?, exact: UUID? = null) = jdbc.update(
        "INSERT INTO batch_relevance_photos (photo_id, batch_id, looked_up_at, match_photo_id, match_distance, match_pair, exact_photo_id) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
        photo, b.id, NOW, match, distance, if (match == null) null else "CLEAN", exact,
    )

    private fun row(photo: UUID): Map<String, Any?> =
        jdbc.queryForMap("SELECT * FROM batch_relevance_photos WHERE photo_id = ?", photo)

    private fun lookedUp(b: Batch): Set<UUID> =
        jdbc.queryForList("SELECT photo_id FROM batch_relevance_photos WHERE batch_id = ?", UUID::class.java, b.id).toSet()

    private fun assertMatch(photo: UUID, match: UUID, distance: Int, pair: String) {
        val r = row(photo)
        assertEquals(match, r["match_photo_id"])
        assertEquals(distance, (r["match_distance"] as Number).toInt())
        assertEquals(pair, r["match_pair"])
    }

    private fun assertNoMatch(photo: UUID) {
        val r = row(photo)
        assertNull(r["match_photo_id"])
        assertNull(r["match_distance"])
        assertNull(r["match_pair"])
    }

    private fun sha() = UUID.randomUUID().toString().replace("-", "").repeat(2)

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2002-06-09T08:00:00.123456Z")
        fun at(seconds: Long): OffsetDateTime = NOW.minusHours(1).plusSeconds(seconds)
    }
}
