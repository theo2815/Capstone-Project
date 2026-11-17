package com.quickpitik.support

import org.flywaydb.core.Flyway
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.core.io.support.PathMatchingResourcePatternResolver
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.jdbc.core.JdbcTemplate
import java.util.UUID
import javax.sql.DataSource
import kotlin.test.assertEquals

// RV-1.1: the record-only schema of Photo Relevance v2 (plan 5.4) on top of the version before it, its
// one-in-flight and one-current rules, and what its foreign keys delete. Fixtures sit in 2003 (shared container).
class RelevanceV2SchemaIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var dataSource: DataSource

    @Test
    fun `the migration adds the word, photo check, case, sample and alert tables on top of the version before it`() {
        val (previous, mine) = versions()
        val schema = "relevance_v2_${UUID.randomUUID().toString().replace("-", "")}"
        fun tables() = jdbc.queryForList(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = ? AND table_name IN " +
                "('batch_relevance_words', 'relevance_photo_checks', 'relevance_cases', 'relevance_case_photos', 'relevance_alerts') ORDER BY 1",
            String::class.java, schema,
        )
        fun columns() = jdbc.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = ? AND table_name = 'batch_relevance' " +
                "AND is_nullable = 'YES' AND column_name IN ('check_claim', 'check_claim_until', 'check_served_at', " +
                "'checked_upload_at', 'checked_photos', 'words_purged_at') ORDER BY 1",
            String::class.java, schema,
        )
        fun indexes() = jdbc.queryForList(
            "SELECT indexname FROM pg_indexes WHERE schemaname = ? AND indexname IN ('idx_relevance_photo_checks_batch', " +
                "'idx_relevance_cases_batch', 'uq_relevance_cases_in_flight', 'uq_relevance_cases_current', 'idx_relevance_alerts_unseen') ORDER BY 1",
            String::class.java, schema,
        )
        try {
            flywayFor(schema, previous).migrate()
            assertEquals(emptyList(), tables())
            assertEquals(emptyList(), columns())

            flywayFor(schema, mine).migrate()

            assertEquals(mine, latestVersion(schema))
            assertEquals(
                listOf("batch_relevance_words", "relevance_alerts", "relevance_case_photos", "relevance_cases", "relevance_photo_checks"),
                tables(),
            )
            assertEquals(
                listOf("check_claim", "check_claim_until", "check_served_at", "checked_photos", "checked_upload_at", "words_purged_at"),
                columns(),
            )
            assertEquals(
                listOf("idx_relevance_alerts_unseen", "idx_relevance_cases_batch", "idx_relevance_photo_checks_batch",
                    "uq_relevance_cases_current", "uq_relevance_cases_in_flight"),
                indexes(),
            )
        } finally {
            jdbc.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    @Test
    fun `a batch can have one check in flight and one current case, never two of either`() {
        val f = fixture()
        val batch = batch(f)
        case(batch, "APPLIED")
        // An applied (current) case does not block a new check.
        case(batch, "SAMPLING")
        assertThrows<DataIntegrityViolationException> { case(batch, "DECIDING") }
        assertThrows<DataIntegrityViolationException> { case(batch, "APPLIED") }
        // Closed cases pile up freely.
        case(batch, "CLOSED")
        case(batch, "CLOSED")
        assertEquals(4, count("SELECT count(*) FROM relevance_cases WHERE batch_id = ?", batch))
    }

    @Test
    fun `deleting a photo removes its check row and its sample rows`() {
        val f = fixture()
        val batch = batch(f)
        val (gone, kept) = List(2) { photo(f) }
        val case = case(batch, "SAMPLING")
        listOf(gone, kept).forEach { check(it, batch); sample(case, it) }

        jdbc.update("DELETE FROM photos WHERE id = ?", gone)

        assertEquals(listOf(kept), ids("SELECT photo_id FROM relevance_photo_checks WHERE batch_id = ?", batch))
        assertEquals(listOf(kept), ids("SELECT photo_id FROM relevance_case_photos WHERE case_id = ?", case))
    }

    @Test
    fun `deleting a batch removes its words, checks, cases, samples and alerts`() {
        val f = fixture()
        val batch = batch(f)
        val photo = photo(f)
        val case = filled(batch, photo)

        jdbc.update("DELETE FROM batch_relevance WHERE id = ?", batch)

        assertEquals(0, leftovers(batch, case, photo))
    }

    @Test
    fun `an alert with a used dedupe key is refused`() {
        val f = fixture()
        val batch = batch(f)
        val key = "rv-test-${UUID.randomUUID()}"
        alert(batch, null, key)
        assertThrows<DataIntegrityViolationException> { alert(batch, null, key) }
    }

    @Test
    fun `every close reason the plan names fits its column`() {
        val f = fixture()
        val batch = batch(f)
        val reasons = listOf(
            "SAME_VERDICT", "SUPERSEDED", "SKIPPED", "NO_RESULT",
            "RECHECK_LIFTED", "HELD_BY_CASE", "RESTORED_BY_ADMIN", "APPEAL_WON", "LOOKS_FINE",
        )
        reasons.forEach {
            jdbc.update(
                "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, closed_at, close_reason) " +
                    "VALUES (?, 'QUIET', TIMESTAMPTZ '2003-02-03 04:05:06+00', 40, 'CLOSED', now(), ?)",
                batch, it,
            )
        }
        assertEquals(reasons.sorted(), jdbc.queryForList(
            "SELECT close_reason FROM relevance_cases WHERE batch_id = ? ORDER BY 1", String::class.java, batch,
        ))
    }

    private class Fixture(val event: UUID, val photographer: UUID)

    private fun fixture(): Fixture {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rv', 'PHOTOGRAPHER')",
            "rv-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Rv Run', DATE '2003-02-03', 'Cebu', 'ACTIVE')",
            "rv-${UUID.randomUUID()}",
        )
        return Fixture(event, photographer)
    }

    private fun photo(f: Fixture) = id(
        "INSERT INTO photos (event_id, photographer_id, s3_key, price_php) VALUES (?, ?, ?, 125)",
        f.event, f.photographer, "rv/${UUID.randomUUID()}",
    )

    private fun batch(f: Fixture) =
        id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", f.event, f.photographer)

    private fun case(batch: UUID, stage: String) = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage) " +
            "VALUES (?, 'QUIET', TIMESTAMPTZ '2003-02-03 04:05:06+00', 40, ?)",
        batch, stage,
    )

    private fun check(photo: UUID, batch: UUID) = jdbc.update(
        "INSERT INTO relevance_photo_checks (photo_id, batch_id, words_counted_at) VALUES (?, ?, now())", photo, batch,
    )

    private fun sample(case: UUID, photo: UUID) =
        jdbc.update("INSERT INTO relevance_case_photos (case_id, photo_id, round) VALUES (?, ?, 1)", case, photo)

    private fun alert(batch: UUID, case: UUID?, key: String) = jdbc.update(
        "INSERT INTO relevance_alerts (batch_id, case_id, kind, dedupe_key, would_have) VALUES (?, ?, 'BIG_HIDE', ?, true)",
        batch, case, key,
    )

    // A batch with one row in every version 2 table; returns its case.
    private fun filled(batch: UUID, photo: UUID): UUID {
        jdbc.update("INSERT INTO batch_relevance_words (batch_id, word, photos) VALUES (?, 'CEBU', 1)", batch)
        check(photo, batch)
        val case = case(batch, "SAMPLING")
        sample(case, photo)
        alert(batch, case, "rv-test-${UUID.randomUUID()}")
        return case
    }

    private fun leftovers(batch: UUID, case: UUID, photo: UUID): Int = count(
        "SELECT (SELECT count(*) FROM batch_relevance_words WHERE batch_id = ?) " +
            "+ (SELECT count(*) FROM relevance_photo_checks WHERE photo_id = ?) " +
            "+ (SELECT count(*) FROM relevance_cases WHERE batch_id = ?) " +
            "+ (SELECT count(*) FROM relevance_case_photos WHERE case_id = ?) " +
            "+ (SELECT count(*) FROM relevance_alerts WHERE batch_id = ?)",
        batch, photo, batch, case, batch,
    )

    private fun ids(sql: String, vararg params: Any): List<UUID> = jdbc.queryForList(sql, UUID::class.java, *params)

    private fun count(sql: String, vararg params: Any): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private fun id(sql: String, vararg params: Any): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    // This migration's version and the one before it, read from the classpath, so a rename at merge needs no edit here.
    private fun versions(suffix: String = "__relevance_v2_record.sql"): Pair<String, String> {
        val all = PathMatchingResourcePatternResolver().getResources("classpath:db/migration/V*__*.sql")
            .map { it.filename!! }
        fun version(name: String) = name.removePrefix("V").substringBefore("__").toInt()
        val mine = version(all.single { it.endsWith(suffix) })
        val previous = all.map(::version).filter { it < mine }.max()
        return previous.toString() to mine.toString()
    }

    // Copied from BatchRelevanceSchemaIntegrationTest (private there).
    private fun latestVersion(schema: String): String = jdbc.queryForObject(
        "SELECT version FROM $schema.flyway_schema_history WHERE success AND version IS NOT NULL " +
            "ORDER BY installed_rank DESC LIMIT 1",
        String::class.java,
    )!!

    private fun flywayFor(schema: String, target: String): Flyway = Flyway.configure()
        .dataSource(dataSource)
        .locations("classpath:db/migration")
        .defaultSchema(schema)
        .schemas(schema)
        .target(target)
        .load()
}
