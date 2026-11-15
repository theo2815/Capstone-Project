package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.AdminProperties
import com.quickpitik.config.AiApiProperties
import com.quickpitik.config.RelevanceMode
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.relevance.AdminRelevanceService
import com.quickpitik.service.relevance.RelevancePhotographerHideService
import com.quickpitik.service.relevance.RelevanceSwitch
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.http.HttpStatus
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// RV-3.3 (G8): a photographer hides the blurry photos of their own batch and undoes it. Their holds are PHOTOGRAPHER
// rows; undo restores only those, by the one restore rule; an admin's batch restore never touches them. The container is
// shared, so fixtures sit in 2010 and assertions look only at this class's own rows.
class RelevancePhotographerHideIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var cases: RelevanceCaseRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var admin: AdminRelevanceService
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var tx: org.springframework.transaction.support.TransactionTemplate
    @MockitoBean lateinit var storage: StorageService
    private lateinit var adminId: UUID

    @BeforeEach fun setUp() {
        Mockito.doAnswer { "https://synthetic.test/${it.arguments[0]}" }.`when`(storage).presignedGetUrl(anyArg(), anyArg())
        adminId = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Admin', 'ADMIN')",
            "relhide-${UUID.randomUUID()}@test.local")
    }
    private fun <T> anyArg(): T = Mockito.any()

    @Test
    fun `a photographer hides their blurry photos and they leave the gallery`() {
        val b = batch()
        val c = case(b, "FITS", "NONE")
        // Postgres orders uuids by their bytes, which is their text order (java.util.UUID.compareTo is signed).
        val blurry = List(2) { photo(b, sharpness = 10.0) }.sortedBy { it.toString() }
        photo(b, sharpness = 50.0)
        photo(b, sharpness = null, error = "BAD_IMAGE")
        val hiddenAlready = photo(b, sharpness = 5.0, status = "HIDDEN")

        val list = service().blurry(b.photographer, c, PaginationParams.of(0, 50))
        assertEquals(blurry, list.items.map { it.photoId })
        assertEquals(2L, list.total)
        assertEquals("https://synthetic.test/wm/${blurry[0]}", list.items[0].thumbnailUrl, "the watermarked preview, never the original")
        assertEquals("HIDDEN", status(hiddenAlready), "a photo already out of the gallery is not listed")

        assertEquals(2, service().hide(b.photographer, c, blurry).affected)
        assertEquals(listOf("HIDDEN", "HIDDEN"), blurry.map { status(it) })
        assertEquals(listOf(listOf("PHOTOGRAPHER", null, c)), holdsOf(b).map { listOf(it[1], it[2], it[3]) }.distinct())
        assertEquals(0L, service().blurry(b.photographer, c, PaginationParams.of(0, 50)).total)
        // A second hide changes nothing; after an undo, a new hide makes the same row live again.
        assertEquals(0, service().hide(b.photographer, c, blurry).affected)
        service().unhide(b.photographer, c, listOf(blurry[0]))
        assertEquals("LIVE", status(blurry[0]))
        assertEquals(1, service().hide(b.photographer, c, listOf(blurry[0])).affected)
        assertEquals(2, holdsOf(b).size)
        assertEquals(listOf(null, null), holdsOf(b).map { it[2] })
    }

    @Test
    fun `undo brings back only photos nothing else holds`() {
        val b = batch()
        val c = case(b, "FITS", "NONE")
        val free = photo(b, sharpness = 10.0)
        val alsoAuto = photo(b, sharpness = 10.0)
        val alsoAdmin = photo(b, sharpness = 10.0)
        service().hide(b.photographer, c, listOf(free, alsoAuto, alsoAdmin))
        hold(b, alsoAuto, "AUTO", c)
        hold(b, alsoAdmin, "ADMIN", c)

        assertEquals(1, service().unhide(b.photographer, c, listOf(free, alsoAuto, alsoAdmin)).affected)

        assertEquals(listOf("LIVE", "HIDDEN", "HIDDEN"), listOf(free, alsoAuto, alsoAdmin).map { status(it) })
        val rows = holdsOf(b).associate { (it[0] to it[1]) to it[2] }
        assertEquals(5, rows.size)
        assertEquals("PHOTOGRAPHER", rows[alsoAuto to "PHOTOGRAPHER"])
        assertEquals(null, rows[alsoAuto to "AUTO"], "another kind's hold stays live")
        assertEquals(null, rows[alsoAdmin to "ADMIN"])
    }

    @Test
    fun `an admin's batch restore leaves the photographer's hides in place`() {
        val b = batch()
        val c = case(b, "JUNK_DUMP", "HIDE_ALL")
        val both = photo(b, sharpness = 10.0, status = "HIDDEN").also { hold(b, it, "AUTO", c) }
        val autoOnly = photo(b, sharpness = 50.0, status = "HIDDEN").also { hold(b, it, "AUTO", c) }
        service().hide(b.photographer, c, listOf(both))

        admin.restore(adminId, c, null)

        assertEquals("LIVE", status(autoOnly))
        assertEquals("HIDDEN", status(both))
        assertEquals(null, holdsOf(b).single { it[0] == both && it[1] == "PHOTOGRAPHER" }[2])
        // The photographer can still undo it, now that nothing else holds it (the case is closed, still auto-hide).
        assertEquals(1, service().unhide(b.photographer, c, listOf(both)).affected)
        assertEquals("LIVE", status(both))
    }

    @Test
    fun `ids outside the caller's batch are refused`() {
        val b = batch()
        val c = case(b, "FITS", "NONE")
        val own = photo(b, sharpness = 10.0)
        val stranger = batch()
        val theirs = photo(stranger, sharpness = 10.0)
        val sameOwnerOtherEvent = photo(batch(photographer = b.photographer), sharpness = 10.0)

        for (ids in listOf(listOf(own, theirs), listOf(own, sameOwnerOtherEvent), listOf(own, UUID.randomUUID()), emptyList(),
            List(201) { UUID.randomUUID() })) {
            assertEquals("photoIds", assertFailsWith<ValidationException> { service().hide(b.photographer, c, ids) }.field)
            assertEquals("photoIds", assertFailsWith<ValidationException> { service().unhide(b.photographer, c, ids) }.field)
        }
        assertEquals(listOf("LIVE", "LIVE", "LIVE"), listOf(own, theirs, sameOwnerOtherEvent).map { status(it) })
        assertEquals(0, holdsOf(b).size + holdsOf(stranger).size)
        // Another photographer's case, and an unknown one, are not found.
        assertFailsWith<NotFoundException> { service().hide(stranger.photographer, c, listOf(theirs)) }
        assertFailsWith<NotFoundException> { service().blurry(stranger.photographer, c, PaginationParams.of(0, 50)) }
        assertFailsWith<NotFoundException> { service().unhide(b.photographer, UUID.randomUUID(), listOf(own)) }
    }

    @Test
    fun `the routes are refused for a record-only case`() {
        val b = batch()
        val c = case(b, "FITS", "NONE", mode = "RECORD_ONLY")
        val p = photo(b, sharpness = 10.0)

        assertEquals(ACTIONS_OFF, refused { service().hide(b.photographer, c, listOf(p)) })
        assertEquals(ACTIONS_OFF, refused { service().unhide(b.photographer, c, listOf(p)) })
        assertFailsWith<NotFoundException> { service().blurry(b.photographer, c, PaginationParams.of(0, 50)) }
        assertEquals("LIVE", status(p))
        assertEquals(0, holdsOf(b).size)
    }

    @Test
    fun `with the mode off or record-only nothing runs`() {
        val b = batch()
        val c = case(b, "FITS", "NONE")
        val p = photo(b, sharpness = 10.0)

        for (mode in listOf(RelevanceMode.OFF, RelevanceMode.RECORD_ONLY)) {
            assertEquals(ACTIONS_OFF, refused { service(mode).hide(b.photographer, c, listOf(p)) }, "$mode")
            assertEquals(ACTIONS_OFF, refused { service(mode).unhide(b.photographer, c, listOf(p)) }, "$mode")
            assertFailsWith<NotFoundException> { service(mode).blurry(b.photographer, c, PaginationParams.of(0, 50)) }
        }
        assertEquals("LIVE", status(p))
        assertEquals(0, holdsOf(b).size)
    }

    private data class Batch(val id: UUID, val event: UUID, val photographer: UUID)

    private fun refused(block: () -> Unit): Pair<HttpStatus, String> =
        assertFailsWith<ApiException> { block() }.let { it.status to it.code }

    private fun service(mode: RelevanceMode = RelevanceMode.AUTO_HIDE): RelevancePhotographerHideService {
        val props = RelevanceProperties(mode = mode, junk = RelevanceProperties.Junk(blurCutoff = 20.0, darkCutoff = 5.0))
        return RelevancePhotographerHideService(cases, holds, photos,
            RelevanceSwitch(props, AdminProperties(flagsEnabled = true), AiApiProperties(enabled = false)), props, storage,
            StorageProperties(), Clock.systemUTC(), tx)
    }

    private fun batch(photographer: UUID? = null): Batch {
        val owner = photographer ?: id("INSERT INTO users (email, password_hash, name, role, terms_version) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER', '1.7')",
            "relhide-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rel Hide Run', DATE '2010-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "relhide-${UUID.randomUUID()}")
        return Batch(id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, owner), event, owner)
    }

    private fun case(b: Batch, verdict: String, action: String, mode: String = "AUTO_HIDE"): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict, decided_at, applied_mode, action, applied_at) " +
            "VALUES (?, 'QUIET', ?, 600, 'APPLIED', ?, ?, ?, ?, ?)",
        b.id, NOW, verdict, NOW, mode, action, NOW,
    )

    // A photo of the batch, measured with this sharpness (null with an error: it could not be measured).
    private fun photo(b: Batch, sharpness: Double?, error: String? = null, status: String = "LIVE"): UUID {
        val photo = id("INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status) VALUES (?, ?, ?, 125, ?)",
            b.event, b.photographer, "original/${UUID.randomUUID()}", status)
        jdbc.update("UPDATE photos SET watermark_s3_key = ?, thumbnail_s3_key = ? WHERE id = ?", "wm/$photo", "th/$photo", photo)
        jdbc.update("INSERT INTO relevance_photo_checks (photo_id, batch_id, measured_at, fg_sharpness, sharpness, brightness, person_conf, race_conf, measure_error) " +
            "VALUES (?, ?, ?, ?, ?, 50, 0.9, 0.9, ?)", photo, b.id, if (error == null) NOW else null, sharpness, sharpness, error)
        return photo
    }

    private fun hold(b: Batch, photo: UUID, heldBy: String, caseId: UUID) {
        jdbc.update("INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, case_id, hidden_at) VALUES (?, ?, ?, ?, ?)",
            b.id, photo, heldBy, caseId, NOW)
    }

    // photo_id, held_by, restored_by, case_id
    private fun holdsOf(b: Batch): List<List<Any?>> = jdbc.queryForList(
        "SELECT photo_id, held_by, restored_by, case_id FROM flag_hidden_photos WHERE batch_id = ? ORDER BY photo_id, held_by", b.id,
    ).map { listOf(it["photo_id"], it["held_by"], it["restored_by"], it["case_id"]) }

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private companion object {
        val NOW: OffsetDateTime = OffsetDateTime.parse("2010-03-04T05:06:07.123456Z")
        val ACTIONS_OFF = HttpStatus.CONFLICT to ErrorCodes.RELEVANCE_ACTIONS_OFF
    }
}
