package com.quickpitik.support

import com.quickpitik.dto.photos.PhotoMetadataFields
import com.quickpitik.entity.PhotoMetadata
import com.quickpitik.entity.PhotoMetadataState
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotoMetadataRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// PE-1.5: the scoring reads beyond the copy counts (plan 5.3 items 2 to 4) against real Postgres. Every
// test builds its own users and events and reads only its own batch, so nothing assumes an empty table.
class BatchRelevanceEvidenceIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var batches: BatchRelevanceRepository
    @Autowired private lateinit var metadata: PhotoMetadataRepository
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var flags: FlagRepository

    @Test
    fun `a saved capture time comes back equal through the entity mapping`() {
        val e = event()
        val me = photographer()
        val p = photo(e, me)
        val taken = LocalDateTime.of(2026, 9, 28, 6, 35, 20)
        // saveAndFlush commits in its own transaction; the read below runs in a fresh persistence context.
        metadata.saveAndFlush(PhotoMetadata(
            photoId = p,
            fields = PhotoMetadataFields(takenAtLocal = taken, takenAtOffset = "+08:00"),
            state = PhotoMetadataState.READY,
        ))

        val row = batches.metadataRows(e, me).single()
        assertEquals(p, row.photoId)
        assertEquals(PhotoMetadataState.READY, row.state)
        assertEquals(taken, row.fields?.takenAtLocal)
        assertEquals("+08:00", row.fields?.takenAtOffset)

        // First observation of the stored shape (plan 4, PE-0 probe p1); recorded in the PE-1.5 journal.
        val stored = jdbc.queryForObject(
            "SELECT jsonb_typeof(fields->'takenAtLocal') || ' ' || CAST(fields->'takenAtLocal' AS text) FROM photo_metadata WHERE photo_id = ?",
            String::class.java, p,
        )
        assertNotNull(stored)
        println("PE-1.5 stored takenAtLocal: $stored")
    }

    @Test
    fun `pending and processing rows come back with no fields, and trashed photos are left out`() {
        val e = event()
        val me = photographer()
        val pending = photo(e, me).also { meta(it, "PENDING") }
        val processing = photo(e, me).also { meta(it, "PROCESSING") }
        photo(e, me, deleted = true).also { meta(it, "READY", taken = true) }
        // Not batch evidence either: still processing, under a removal review, or someone else's.
        photo(e, me, status = "PROCESSING").also { meta(it, "READY", taken = true) }
        photo(e, me, removal = "REVIEW").also { meta(it, "READY", taken = true) }
        photo(e, photographer()).also { meta(it, "READY", taken = true) }

        val rows = batches.metadataRows(e, me).associateBy { it.photoId }
        assertEquals(setOf(pending, processing), rows.keys)
        assertEquals(PhotoMetadataState.PENDING, rows.getValue(pending).state)
        assertEquals(PhotoMetadataState.PROCESSING, rows.getValue(processing).state)
        assertNull(rows.getValue(pending).fields)
        assertNull(rows.getValue(processing).fields)
    }

    @Test
    fun `a hidden photo's capture time, marker and bibs still count`() {
        val e = event()
        val other = photographer()
        val theirs = photo(e, other).also { bibs(it, "101") }
        val me = photographer()
        val hidden = photo(e, me, status = "HIDDEN").also {
            meta(it, "READY", taken = true, xmp = theirs)
            bibs(it, "101")
        }

        assertEquals(listOf(hidden), batches.metadataRows(e, me).map { it.photoId })
        assertEquals(1L, batches.xmpMatches(e, me))
        assertEquals(1L to 1L, overlap(e, me))
    }

    @Test
    fun `a marker naming another photographer's photo counts and one naming the uploader's own photo does not`() {
        val e = event()
        val me = photographer()
        val mine = photo(e, me)
        val theirs = photo(e, photographer(), status = "PROCESSING")
        val inOtherEvent = photo(event(), photographer())
        val trashed = photo(e, photographer(), deleted = true)
        val noOwner = photo(e, null)
        photo(e, me).also { meta(it, "READY", xmp = theirs) }
        photo(e, me).also { meta(it, "READY", xmp = inOtherEvent) }
        photo(e, me).also { meta(it, "READY", xmp = mine) }
        photo(e, me).also { meta(it, "READY", xmp = trashed) }
        photo(e, me).also { meta(it, "READY", xmp = noOwner) }

        // Any status and any event on the named side; trashed and ownerless photos are no evidence.
        assertEquals(2L, batches.xmpMatches(e, me))
    }

    @Test
    fun `a marker naming no existing photo counts nothing`() {
        val e = event()
        val me = photographer()
        photo(e, me).also { meta(it, "READY", xmp = UUID.randomUUID()) }

        assertEquals(0L, batches.xmpMatches(e, me))
    }

    @Test
    fun `bib overlap counts a photo once however many of its words match`() {
        val e = event()
        photo(e, photographer()).also { bibs(it, "101", "102") }
        val me = photographer()
        photo(e, me).also { bibs(it, "101", "102", "999") }
        photo(e, me).also { bibs(it, "555") }

        assertEquals(2L to 1L, overlap(e, me))
    }

    @Test
    fun `a word seen only in another event does not overlap`() {
        val e = event()
        photo(event(), photographer()).also { bibs(it, "101") }
        val me = photographer()
        photo(e, me).also { bibs(it, "101") }
        // The uploader's own photos are not "other photographers" either.
        photo(e, me).also { bibs(it, "101") }

        assertEquals(2L to 0L, overlap(e, me))
    }

    @Test
    fun `a batch with no bib rows has zero bibbed photos`() {
        val e = event()
        photo(e, photographer()).also { bibs(it, "101") }
        val me = photographer()
        photo(e, me)

        assertEquals(0L to 0L, overlap(e, me))
    }

    // ─── PE-2.3: the admin's evidence read ───────────────────────────────

    @Test
    fun `copy samples come nearest first, leave out trashed photos and keep hidden ones`() {
        val e = event()
        val theirs = photo(e, photographer())
        val me = photographer()
        val batch = batch(e, me)
        val far = photo(e, me).also { lookup(it, batch, theirs, 3) }
        val nearest = photo(e, me).also { lookup(it, batch, theirs, 0) }
        val exactOnly = photo(e, me).also { lookup(it, batch, exact = theirs) }
        val hidden = photo(e, me, status = "HIDDEN").also { lookup(it, batch, theirs, 1) }
        photo(e, me, deleted = true).also { lookup(it, batch, theirs, 0) }
        photo(e, me, removal = "REVIEW").also { lookup(it, batch, theirs, 0) }
        photo(e, me, status = "PROCESSING").also { lookup(it, batch, theirs, 0) }
        photo(e, me).also { lookup(it, batch) } // looked up, nothing found: no sample

        val rows = batches.copySamples(batch, 12)
        assertEquals(listOf(nearest, hidden, far, exactOnly), rows.map { it.photoId })
        assertEquals(listOf(0, 1, 3, null), rows.map { it.distance })
        assertEquals(theirs, rows.first().matchPhotoId)
        assertEquals("CLEAN", rows.first().pair)
        assertNull(rows.last().matchPhotoId)
        assertEquals(theirs, rows.last().exactPhotoId)
        assertEquals(listOf(nearest, hidden), batches.copySamples(batch, 2).map { it.photoId })
    }

    @Test
    fun `batches are found by their flag`() {
        val me = photographer()
        val flag = flag(me)
        val linked = listOf(batch(event(), me, flag), batch(event(), me, flag))
        batch(event(), me)
        batch(event(), photographer(), flag(photographer()))

        assertEquals(linked.toSet(), batches.findByFlagId(flag).map { it.id }.toSet())
    }

    @Test
    fun `the photos a flag hid are counted per event and restored ones are not`() {
        val me = photographer()
        val flag = flag(me)
        val e1 = event()
        val e2 = event()
        hide(flag, photo(e1, me))
        hide(flag, photo(e1, me))
        hide(flag, photo(e1, me), restored = true)
        hide(flag, photo(e2, me))
        hide(flag(me), photo(e2, me)) // another flag's hide

        val counts = flags.countHiddenPhotosByEvent(flag).associate { it.eventId to it.hidden }
        assertEquals(mapOf(e1 to 2L, e2 to 1L), counts)
    }

    private fun batch(event: UUID, photographer: UUID, flag: UUID? = null): UUID = id(
        "INSERT INTO batch_relevance (event_id, photographer_id, flag_id) VALUES (?, ?, ?)", event, photographer, flag,
    )

    private fun lookup(photo: UUID, batch: UUID, match: UUID? = null, distance: Int? = null, exact: UUID? = null) = jdbc.update(
        "INSERT INTO batch_relevance_photos (photo_id, batch_id, looked_up_at, match_photo_id, match_distance, match_pair, exact_photo_id) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
        photo, batch, NOW, match, distance, match?.let { "CLEAN" }, exact,
    )

    private fun flag(photographer: UUID): UUID =
        id("INSERT INTO flags (target_kind, target_id, reason) VALUES ('user', ?, 'batch_relevance_low')", photographer)

    private fun hide(flag: UUID, photo: UUID, restored: Boolean = false) = jdbc.update(
        "INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at, restored_at) VALUES (?, ?, ?, ?)",
        flag, photo, NOW, if (restored) NOW else null,
    )

    private fun overlap(event: UUID, photographer: UUID): Pair<Long, Long> =
        batches.bibOverlap(event, photographer).let { it.bibbedPhotos to it.bibOverlapPhotos }

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
        status: String = "LIVE",
        deleted: Boolean = false,
        removal: String = "NONE",
    ): UUID = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, uploaded_at, status, deleted_at, removal_state) " +
            "VALUES (?, ?, ?, 125, ?, ?, ?, ?)",
        event, photographer, "rel/${UUID.randomUUID()}", NOW, status, if (deleted) NOW else null, removal,
    )

    private fun meta(photo: UUID, state: String, taken: Boolean = false, xmp: UUID? = null) = jdbc.update(
        "INSERT INTO photo_metadata (photo_id, state, fields, xmp_photo_id) VALUES (?, ?, CAST(? AS jsonb), ?)",
        photo, state, if (taken) """{"takenAtLocal":"2002-06-09T07:00:00"}""" else null, xmp,
    )

    private fun bibs(photo: UUID, vararg words: String) = words.forEach {
        jdbc.update("INSERT INTO photo_bibs (photo_id, bib_number, ocr_confidence) VALUES (?, ?, 0.99)", photo, it)
    }

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2002-06-09T08:00:00.123456Z")
    }
}
