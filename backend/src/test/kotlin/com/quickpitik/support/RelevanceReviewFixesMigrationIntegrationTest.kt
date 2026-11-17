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
import kotlin.test.assertNull

// RVF-3: the review fixes' one migration (V108). Only the schema: no entity or query reads it yet.
class RelevanceReviewFixesMigrationIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var dataSource: DataSource

    @Test
    fun `a case may carry the trigger RETRY`() {
        val case = case(batch(), trigger = "RETRY")
        assertEquals("RETRY", jdbc.queryForObject("SELECT trigger FROM relevance_cases WHERE id = ?", String::class.java, case))
    }

    @Test
    fun `an alert may carry the kind WORDS_ONLY`() {
        val batch = batch()
        val alert = id(
            "INSERT INTO relevance_alerts (batch_id, case_id, kind, dedupe_key, would_have) VALUES (?, ?, 'WORDS_ONLY', ?, false)",
            batch, case(batch), "rvf3-${UUID.randomUUID()}",
        )
        assertEquals("WORDS_ONLY", jdbc.queryForObject("SELECT kind FROM relevance_alerts WHERE id = ?", String::class.java, alert))
    }

    @Test
    fun `a new case has no consent mark and no start mode`() {
        val case = case(batch())
        assertEquals(false, jdbc.queryForObject("SELECT no_consent FROM relevance_cases WHERE id = ?", Boolean::class.java, case))
        assertNull(jdbc.queryForObject("SELECT started_mode FROM relevance_cases WHERE id = ?", String::class.java, case))
        jdbc.update("UPDATE relevance_cases SET started_mode = 'AUTO_HIDE' WHERE id = ?", case)
        assertThrows<DataIntegrityViolationException> {
            jdbc.update("UPDATE relevance_cases SET started_mode = 'SOMETIMES' WHERE id = ?", case)
        }
    }

    @Test
    fun `a strike has no void date until one is set`() {
        val photographer = photographer()
        val strike = id(
            "INSERT INTO relevance_strikes (photographer_id, case_id, event_id, verdict, confirmed_at) VALUES (?, ?, ?, 'JUNK_DUMP', now())",
            photographer, UUID.randomUUID(), UUID.randomUUID(),
        )
        assertNull(jdbc.queryForObject("SELECT voided_at FROM relevance_strikes WHERE id = ?", java.time.OffsetDateTime::class.java, strike))
        jdbc.update("UPDATE relevance_strikes SET voided_at = now() WHERE id = ?", strike)
        assertEquals(1, jdbc.queryForObject("SELECT count(*) FROM relevance_strikes WHERE id = ? AND voided_at IS NOT NULL", Int::class.java, strike))
    }

    @Test
    fun `a made-up trigger and a made-up alert kind are still refused`() {
        val batch = batch()
        assertThrows<DataIntegrityViolationException> { case(batch, trigger = "AGAIN") }
        assertThrows<DataIntegrityViolationException> {
            jdbc.update(
                "INSERT INTO relevance_alerts (batch_id, kind, dedupe_key, would_have) VALUES (?, 'WORDS', ?, false)",
                batch, "rvf3-${UUID.randomUUID()}",
            )
        }
    }

    @Test
    fun `an existing case with the note NO_CONSENT gets the consent mark and other cases do not`() {
        val (previous, mine) = versions()
        val schema = "rvf3_${UUID.randomUUID().toString().replace("-", "")}"
        try {
            flywayFor(schema, previous).migrate()
            val photographer = id(
                "INSERT INTO $schema.users (email, password_hash, name, role) VALUES (?, 'unused', 'Rvf', 'PHOTOGRAPHER')",
                "rvf3-${UUID.randomUUID()}@test.local",
            )
            val event = id(
                "INSERT INTO $schema.events (slug, name, date, location, status, visibility) VALUES (?, 'Rvf Run', DATE '2002-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
                "rvf3-${UUID.randomUUID()}",
            )
            val batch = id("INSERT INTO $schema.batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer)
            fun case(note: String?) = id(
                "INSERT INTO $schema.relevance_cases (batch_id, trigger, upload_at, batch_photos, stage, verdict_note) VALUES (?, 'QUIET', now(), 40, 'CLOSED', ?)",
                batch, note,
            )
            val noConsent = case("NO_CONSENT")
            val other = case("FAILED_CALL")

            flywayFor(schema, mine).migrate()

            fun mark(case: UUID) = jdbc.queryForObject("SELECT no_consent FROM $schema.relevance_cases WHERE id = ?", Boolean::class.java, case)
            assertEquals(true, mark(noConsent))
            assertEquals(false, mark(other))
        } finally {
            jdbc.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    private fun photographer(): UUID = id(
        "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rvf', 'PHOTOGRAPHER')",
        "rvf3-${UUID.randomUUID()}@test.local",
    )

    private fun batch(): UUID {
        val event = id(
            "INSERT INTO events (slug, name, date, location, status, visibility) VALUES (?, 'Rvf Run', DATE '2002-03-04', 'Cebu', 'ACTIVE', 'PUBLIC')",
            "rvf3-${UUID.randomUUID()}",
        )
        return id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer())
    }

    private fun case(batchId: UUID, trigger: String = "QUIET"): UUID = id(
        "INSERT INTO relevance_cases (batch_id, trigger, upload_at, batch_photos, stage) VALUES (?, ?, now(), 40, 'CLOSED')",
        batchId, trigger,
    )

    private fun id(sql: String, vararg params: Any?): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    // This migration's version and the one before it, read from the classpath, so a rename at merge needs no edit here.
    private fun versions(suffix: String = "__relevance_review_fixes.sql"): Pair<String, String> {
        val all = PathMatchingResourcePatternResolver().getResources("classpath:db/migration/V*__*.sql").map { it.filename!! }
        fun version(name: String) = name.removePrefix("V").substringBefore("__").toInt()
        val mine = version(all.single { it.endsWith(suffix) })
        val previous = all.map(::version).filter { it < mine }.max()
        return previous.toString() to mine.toString()
    }

    private fun flywayFor(schema: String, target: String): Flyway = Flyway.configure()
        .dataSource(dataSource).locations("classpath:db/migration").defaultSchema(schema).schemas(schema).target(target).load()
}
