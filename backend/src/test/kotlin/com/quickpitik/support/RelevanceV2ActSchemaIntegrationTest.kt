package com.quickpitik.support

import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.RelevanceHoldRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import org.flywaydb.core.Flyway
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.core.io.support.PathMatchingResourcePatternResolver
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.data.jpa.repository.Query
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.datasource.SingleConnectionDataSource
import org.springframework.transaction.support.TransactionTemplate
import java.time.OffsetDateTime
import java.util.UUID
import javax.sql.DataSource
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

// RV-2.1: migration 2 of Photo Relevance v2 (part 2, A.1): flag_hidden_photos becomes the one hold table, the
// decision log gets an automatic actor, and appeals, strikes and the relevance note exist. Fixtures sit in 2003.
class RelevanceV2ActSchemaIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var dataSource: DataSource
    @Autowired private lateinit var tx: TransactionTemplate
    @Autowired private lateinit var flags: FlagRepository
    @Autowired private lateinit var photos: PhotoRepository
    @Autowired private lateinit var holds: RelevanceHoldRepository
    @Autowired private lateinit var decisions: AdminDecisionLogService

    @Test
    fun `the migration keeps every existing flag hold and its restore still works`() {
        val (previous, mine) = versions()
        val schema = "relevance_act_${UUID.randomUUID().toString().replace("-", "")}"
        try {
            flywayFor(schema, previous).migrate()
            val old = SingleConnectionDataSource(dataSource.connection, true)
            try {
                val db = JdbcTemplate(old)
                db.execute("SET search_path TO $schema, public")
                fun id(sql: String, vararg p: Any) = db.queryForObject("$sql RETURNING id", UUID::class.java, *p)!!
                val user = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'x', 'Rv', 'PHOTOGRAPHER')", "rv-${UUID.randomUUID()}@test.local")
                val event = id("INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Rv Run', DATE '2003-02-03', 'Cebu', 'ACTIVE')", "rv-${UUID.randomUUID()}")
                fun photo() = id("INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status, watermark_s3_key) VALUES (?, ?, ?, 125, 'HIDDEN', 'wm')", event, user, "rv/${UUID.randomUUID()}")
                fun flag() = id("INSERT INTO flags (target_kind, target_id, reason, status) VALUES ('user', ?, 'batch_relevance_low', 'open')", user)
                val (dismissed, other) = List(2) { flag() }
                val (free, shared, wasRestored) = List(3) { photo() }
                val hidden = OffsetDateTime.parse("2003-02-03T04:05:06Z")
                listOf(free, shared, wasRestored).forEach {
                    db.update("INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at, restored_at) VALUES (?, ?, ?, ?)",
                        dismissed, it, hidden, if (it == wasRestored) hidden.plusDays(1) else null)
                }
                db.update("INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at) VALUES (?, ?, ?)", other, shared, hidden)

                flywayFor(schema, mine).migrate()
                assertEquals(mine, latestVersion(schema))

                // Every row kept as it was, now a FLAG hold with its own id and no batch.
                assertEquals(
                    listOf("$dismissed|$free|FLAG|false", "$dismissed|$shared|FLAG|false", "$dismissed|$wasRestored|FLAG|true", "$other|$shared|FLAG|false").sorted(),
                    db.queryForList(
                        "SELECT flag_id || '|' || photo_id || '|' || held_by || '|' || (restored_at IS NOT NULL) FROM flag_hidden_photos " +
                            "WHERE id IS NOT NULL AND batch_id IS NULL AND case_id IS NULL AND restored_by IS NULL AND hidden_at = ?",
                        String::class.java, hidden,
                    ).sorted(),
                )
                assertEquals(listOf("id"), db.queryForList(
                    "SELECT a.attname FROM pg_constraint c JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY (c.conkey) " +
                        "WHERE c.conname = 'flag_hidden_photos_pkey' AND c.connamespace = ?::regnamespace",
                    String::class.java, schema,
                ))
                // Version 1's insert still finds its conflict target, and its bulk restore still follows the rule.
                db.update("INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at) VALUES (?, ?, now()) ON CONFLICT (flag_id, photo_id) DO NOTHING", dismissed, free)
                assertEquals(4, db.queryForObject("SELECT count(*) FROM flag_hidden_photos", Int::class.java))
                assertEquals(1, db.update(realQuery("unhideRecordedForFlag"), dismissed, dismissed, dismissed))
                assertEquals(mapOf(free to "LIVE", shared to "HIDDEN", wasRestored to "HIDDEN"),
                    listOf(free, shared, wasRestored).associateWith { db.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, it) })
            } finally {
                JdbcTemplate(old).execute("RESET search_path")
                old.destroy()
            }
        } finally {
            jdbc.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    @Test
    fun `a hold row belongs to a flag or to a batch, never both`() {
        val f = fixture()
        val photo = photo(f)
        val flag = flag(f)
        val batch = batch(f)
        hold(flag = flag, batch = null, photo = photo, heldBy = "FLAG")
        hold(flag = null, batch = batch, photo = photo, heldBy = "AUTO")
        hold(flag = null, batch = batch, photo = photo, heldBy = "ADMIN")
        hold(flag = null, batch = batch, photo = photo, heldBy = "PHOTOGRAPHER")
        assertThrows<DataIntegrityViolationException> { hold(flag = flag(f), batch = batch(f), photo = photo, heldBy = "AUTO") }
        assertThrows<DataIntegrityViolationException> { hold(flag = null, batch = null, photo = photo, heldBy = "AUTO") }
        assertThrows<DataIntegrityViolationException> { hold(flag = flag(f), batch = null, photo = photo, heldBy = "AUTO") }
        assertThrows<DataIntegrityViolationException> { hold(flag = null, batch = batch(f), photo = photo, heldBy = "FLAG") }
        assertThrows<DataIntegrityViolationException> { hold(flag = null, batch = batch(f), photo = photo, heldBy = "SOMEONE") }
        // One hold per batch, photo and kind.
        assertThrows<DataIntegrityViolationException> { hold(flag = null, batch = batch, photo = photo, heldBy = "AUTO") }
    }

    @Test
    fun `a batch hold is restored with a restorer or not at all`() {
        val f = fixture()
        val batch = batch(f)
        fun restored(at: OffsetDateTime?, by: String?) = jdbc.update(
            "INSERT INTO flag_hidden_photos (batch_id, photo_id, held_by, hidden_at, restored_at, restored_by) VALUES (?, ?, 'AUTO', now(), ?, ?)",
            batch, photo(f), at, by,
        )
        restored(null, null)
        restored(OffsetDateTime.now(), "ADMIN")
        assertThrows<DataIntegrityViolationException> { restored(OffsetDateTime.now(), null) }
        assertThrows<DataIntegrityViolationException> { restored(null, "AUTO") }
        assertThrows<DataIntegrityViolationException> { restored(OffsetDateTime.now(), "FLAG") }
        // A flag row keeps version 1's shape: restored_at alone.
        jdbc.update("INSERT INTO flag_hidden_photos (flag_id, photo_id, hidden_at, restored_at) VALUES (?, ?, now(), now())", flag(f), photo(f))
    }

    @Test
    fun `a photo held by a batch stays hidden when a flag that also held it is dismissed`() {
        val f = fixture()
        val photo = photo(f, "HIDDEN")
        val flag = flag(f)
        hold(flag = flag, batch = null, photo = photo, heldBy = "FLAG")
        hold(flag = null, batch = batch(f), photo = photo, heldBy = "AUTO")

        assertTrue(flags.existsOtherBatchHold(photo, flag))
        assertEquals(0, tx.execute { photos.unhideRecordedForFlag(flag) })
        assertEquals("HIDDEN", status(photo))
    }

    @Test
    fun `a photo held by a flag stays hidden when its batch hold is restored`() {
        val f = fixture()
        val batch = batch(f)
        val (alsoFlagged, onlyBatch, unwatermarked) = listOf(photo(f, "HIDDEN"), photo(f, "LIVE"), photo(f, "PROCESSING", watermark = null))
        hold(flag = flag(f), batch = null, photo = alsoFlagged, heldBy = "FLAG")
        val case = id("INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage) VALUES (?, 'QUIET', TIMESTAMPTZ '2003-02-03 04:05:06+00', 40, 'APPLIED')", batch)
        val now = OffsetDateTime.parse("2003-02-04T04:05:06Z")

        val held = tx.execute { holds.hideChunk(batch, case, f.event, f.photographer, "AUTO", now, 10).also { assertEquals(2, holds.hidePhotos(it)) } }!!
        assertEquals(setOf(alsoFlagged, onlyBatch, unwatermarked), held.toSet())
        assertEquals(listOf("HIDDEN"), listOf(alsoFlagged, onlyBatch, unwatermarked).map(::status).distinct())
        assertEquals(3L, holds.countLiveHolds(batch, "AUTO"))
        assertEquals(0L, holds.countLiveHolds(batch, "ADMIN"))
        // A second pass finds nothing new to hold.
        assertEquals(emptyList<UUID>(), tx.execute { holds.hideChunk(batch, case, f.event, f.photographer, "AUTO", now, 10) })

        val released = tx.execute {
            holds.restoreChunk(batch, "AUTO", "ADMIN", now, 2).also { holds.unhideReleased(it) }
        }!!
        assertEquals(2, released.size)
        val rest = tx.execute { holds.restoreChunk(batch, "AUTO", "ADMIN", now, 2).also { holds.unhideReleased(it) } }!!
        assertEquals(setOf(alsoFlagged, onlyBatch, unwatermarked), (released + rest).toSet())
        assertEquals(0L, holds.countLiveHolds(batch, "AUTO"))
        assertEquals(mapOf(alsoFlagged to "HIDDEN", onlyBatch to "LIVE", unwatermarked to "PROCESSING"),
            listOf(alsoFlagged, onlyBatch, unwatermarked).associateWith(::status))
        // A photo a person restored under this case is not held again by the same case.
        assertEquals(emptyList<UUID>(), tx.execute { holds.hideChunk(batch, case, f.event, f.photographer, "AUTO", now, 10) })
    }

    @Test
    fun `an automatic decision is logged with no admin, and a row with neither is refused`() {
        val f = fixture()
        val logged = tx.execute { decisions.logAutomaticUserDecision(f.photographer, "relevance_hidden", "junk", mapOf("caseId" to "c")) }!!
        assertNull(logged.adminId)
        assertTrue(logged.automatic)
        assertEquals("true|null", jdbc.queryForObject(
            "SELECT automatic || '|' || coalesce(admin_id::text, 'null') FROM admin_decision_log WHERE id = ?", String::class.java, logged.id,
        ))
        assertThrows<DataIntegrityViolationException> {
            jdbc.update("INSERT INTO admin_decision_log (admin_id, target_user_id, decision) VALUES (NULL, ?, 'nobody')", f.photographer)
        }
        // An admin's row is unchanged: not automatic by default.
        val admin = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'x', 'Ad', 'ADMIN')", "rv-${UUID.randomUUID()}@test.local")
        val row = id("INSERT INTO admin_decision_log (admin_id, target_user_id, decision) VALUES (?, ?, 'approved')", admin, f.photographer)
        assertFalse(jdbc.queryForObject("SELECT automatic FROM admin_decision_log WHERE id = ?", Boolean::class.java, row)!!)
    }

    @Test
    fun `the new message kind is accepted and an unknown kind is still refused`() {
        val f = fixture()
        val case = id("INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage) VALUES (?, 'QUIET', TIMESTAMPTZ '2003-02-03 04:05:06+00', 40, 'APPLIED')", batch(f))
        // Every kind the code knows, the 21 of V73 and the new one, fits the check.
        PhotographerMessageKind.entries.forEach {
            jdbc.update("INSERT INTO photographer_messages (photographer_id, kind, body) VALUES (?, ?, 'x')", f.photographer, it.wire)
        }
        assertEquals(22, PhotographerMessageKind.entries.size)
        jdbc.update("INSERT INTO photographer_messages (photographer_id, kind, body, relevance_case_id) VALUES (?, 'relevance_notice', 'x', ?)", f.photographer, case)
        assertThrows<DataIntegrityViolationException> {
            jdbc.update("INSERT INTO photographer_messages (photographer_id, kind, body) VALUES (?, 'bogus_kind', 'x')", f.photographer)
        }
    }

    @Test
    fun `one appeal per case and one open appeal per batch`() {
        val f = fixture()
        val batch = batch(f)
        fun case() = id("INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage) VALUES (?, 'QUIET', TIMESTAMPTZ '2003-02-03 04:05:06+00', 40, 'CLOSED')", batch)
        fun appeal(case: UUID, status: String = "OPEN", reason: String = "These are mine") = jdbc.update(
            "INSERT INTO relevance_appeals (case_id, batch_id, photographer_id, reason, status) VALUES (?, ?, ?, ?, ?)",
            case, batch, f.photographer, reason, status,
        )
        val first = case()
        appeal(first, "LOST")
        assertThrows<DataIntegrityViolationException> { appeal(first) }
        val (second, third) = List(2) { case() }
        appeal(second)
        assertThrows<DataIntegrityViolationException> { appeal(third) }
        assertThrows<DataIntegrityViolationException> { appeal(case(), "LOST", reason = "") }
        assertThrows<DataIntegrityViolationException> { appeal(case(), "LOST", reason = "x".repeat(1001)) }
        jdbc.update("UPDATE relevance_appeals SET status = 'WON' WHERE case_id = ?", second)
        appeal(third)
    }

    @Test
    fun `a strike survives the deletion of its event's batch row`() {
        val f = fixture()
        val batch = batch(f)
        val case = id("INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage) VALUES (?, 'QUIET', TIMESTAMPTZ '2003-02-03 04:05:06+00', 40, 'APPLIED')", batch)
        jdbc.update("INSERT INTO relevance_strikes (photographer_id, case_id, event_id, verdict, confirmed_at) VALUES (?, ?, ?, 'JUNK_DUMP', now())", f.photographer, case, f.event)
        assertThrows<DataIntegrityViolationException> {
            jdbc.update("INSERT INTO relevance_strikes (photographer_id, case_id, event_id, verdict, confirmed_at) VALUES (?, ?, ?, 'JUNK_DUMP', now())", f.photographer, case, f.event)
        }

        jdbc.update("DELETE FROM batch_relevance WHERE id = ?", batch)

        assertEquals(0, count("SELECT count(*) FROM relevance_cases WHERE id = ?", case))
        assertEquals(1, count("SELECT count(*) FROM relevance_strikes WHERE case_id = ? AND event_id = ?", case, f.event))
    }

    private class Fixture(val event: UUID, val photographer: UUID)

    private fun fixture(): Fixture {
        val photographer = id("INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rv', 'PHOTOGRAPHER')", "rv-${UUID.randomUUID()}@test.local")
        val event = id("INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Rv Run', DATE '2003-02-03', 'Cebu', 'ACTIVE')", "rv-${UUID.randomUUID()}")
        return Fixture(event, photographer)
    }

    private fun photo(f: Fixture, status: String = "LIVE", watermark: String? = "wm") = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, status, watermark_s3_key) VALUES (?, ?, ?, 125, ?, ?)",
        f.event, f.photographer, "rv/${UUID.randomUUID()}", status, watermark,
    )

    private fun flag(f: Fixture) =
        id("INSERT INTO flags (target_kind, target_id, reason, status) VALUES ('user', ?, 'batch_relevance_low', 'open')", f.photographer)

    // A fresh batch row needs a fresh (event, photographer) pair.
    private fun batch(f: Fixture) = id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", fixture().event, f.photographer)

    private fun hold(flag: UUID?, batch: UUID?, photo: UUID, heldBy: String) = jdbc.update(
        "INSERT INTO flag_hidden_photos (flag_id, batch_id, photo_id, held_by, hidden_at) VALUES (?, ?, ?, ?, now())",
        flag, batch, photo, heldBy,
    )

    private fun status(photo: UUID): String = jdbc.queryForObject("SELECT status FROM photos WHERE id = ?", String::class.java, photo)!!

    private fun count(sql: String, vararg params: Any): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    // The text of a version 1 repository query, with its named parameter made positional, to run in a throwaway schema.
    private fun realQuery(method: String): String =
        PhotoRepository::class.java.getMethod(method, UUID::class.java).getAnnotation(Query::class.java).value.replace(":flagId", "?")

    // This migration's version and the one before it, read from the classpath, so a rename at merge needs no edit here.
    private fun versions(suffix: String = "__relevance_v2_act.sql"): Pair<String, String> {
        val all = PathMatchingResourcePatternResolver().getResources("classpath:db/migration/V*__*.sql").map { it.filename!! }
        fun version(name: String) = name.removePrefix("V").substringBefore("__").toInt()
        val mine = version(all.single { it.endsWith(suffix) })
        val previous = all.map(::version).filter { it < mine }.max()
        return previous.toString() to mine.toString()
    }

    private fun latestVersion(schema: String): String = jdbc.queryForObject(
        "SELECT version FROM $schema.flyway_schema_history WHERE success AND version IS NOT NULL ORDER BY installed_rank DESC LIMIT 1",
        String::class.java,
    )!!

    private fun flywayFor(schema: String, target: String): Flyway = Flyway.configure()
        .dataSource(dataSource).locations("classpath:db/migration").defaultSchema(schema).schemas(schema).target(target).load()
}
