package com.quickpitik.support

import org.flywaydb.core.Flyway
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.core.io.support.PathMatchingResourcePatternResolver
import org.springframework.jdbc.core.JdbcTemplate
import java.util.UUID
import javax.sql.DataSource
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// PE-1.2: the relevance migration on top of the version before it, and what its foreign keys delete.
class BatchRelevanceSchemaIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var dataSource: DataSource

    @Test
    fun `the migration adds both tables, the content hash index and the marker column on top of the version before it`() {
        val (previous, mine) = versions()
        val schema = "relevance_${UUID.randomUUID().toString().replace("-", "")}"
        try {
            flywayFor(schema, previous).migrate()
            assertEquals(previous, latestVersion(schema))
            assertEquals(0, tables(schema))

            flywayFor(schema, mine).migrate()

            assertEquals(mine, latestVersion(schema))
            assertEquals(2, tables(schema))
            val index = jdbc.queryForObject(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = ? AND indexname = 'idx_photos_content_hash'",
                String::class.java, schema,
            )!!
            assertTrue("(content_hash)" in index && "WHERE (content_hash IS NOT NULL)" in index, index)
            assertEquals(listOf("uuid"), jdbc.queryForList(
                "SELECT data_type FROM information_schema.columns WHERE table_schema = ? AND table_name = 'photo_metadata' " +
                    "AND column_name = 'xmp_photo_id' AND is_nullable = 'YES'",
                String::class.java, schema,
            ))
        } finally {
            jdbc.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    @Test
    fun `deleting a photo removes its look-up row and deleting a batch removes its rows`() {
        val f = fixture()
        val (gone, kept) = List(2) { photo(f.event, f.photographer) }
        val batch = batch(f.event, f.photographer)
        lookup(gone, batch)
        lookup(kept, batch)

        jdbc.update("DELETE FROM photos WHERE id = ?", gone)
        assertEquals(listOf(kept), lookups(batch))

        jdbc.update("DELETE FROM batch_relevance WHERE id = ?", batch)
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM batch_relevance_photos WHERE photo_id = ?", Int::class.java, kept))
    }

    // R2-11: a privacy removal keeps the photo row as a tombstone (deleted_at set), so its look-up row stays;
    // the count and match queries (PE-1.3, PE-1.5) skip it by deleted_at.
    @Test
    fun `after a privacy removal the photo keeps its look-up row and is marked deleted`() {
        val f = fixture()
        val photo = photo(f.event, f.photographer)
        val batch = batch(f.event, f.photographer)
        lookup(photo, batch)

        jdbc.update(
            "UPDATE photos SET deleted_at = now(), deletion_started_at = now(), purged_at = now(), s3_key = '' WHERE id = ?",
            photo,
        )

        assertEquals(listOf(photo), lookups(batch))
        assertEquals(1, jdbc.queryForObject(
            "SELECT count(*) FROM photos WHERE id = ? AND deleted_at IS NOT NULL", Int::class.java, photo,
        ))
    }

    private class Fixture(val event: UUID, val photographer: UUID)

    private fun fixture(): Fixture {
        val photographer = id(
            "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Rel', 'PHOTOGRAPHER')",
            "rel-${UUID.randomUUID()}@test.local",
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Rel Run', DATE '2001-03-04', 'Cebu', 'ACTIVE')",
            "rel-${UUID.randomUUID()}",
        )
        return Fixture(event, photographer)
    }

    private fun photo(event: UUID, photographer: UUID) =
        id("INSERT INTO photos (event_id, photographer_id, s3_key, price_php) VALUES (?, ?, ?, 125)", event, photographer, "rel/${UUID.randomUUID()}")

    private fun batch(event: UUID, photographer: UUID) =
        id("INSERT INTO batch_relevance (event_id, photographer_id) VALUES (?, ?)", event, photographer)

    private fun lookup(photo: UUID, batch: UUID) =
        jdbc.update("INSERT INTO batch_relevance_photos (photo_id, batch_id, looked_up_at) VALUES (?, ?, now())", photo, batch)

    private fun lookups(batch: UUID): List<UUID> =
        jdbc.queryForList("SELECT photo_id FROM batch_relevance_photos WHERE batch_id = ?", UUID::class.java, batch)

    private fun id(sql: String, vararg params: Any): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private fun tables(schema: String): Int = jdbc.queryForObject(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = ? " +
            "AND table_name IN ('batch_relevance', 'batch_relevance_photos')",
        Int::class.java, schema,
    )!!

    // This migration's version and the one before it, read from the classpath, so a rename at merge
    // (the number is fixed then: plan R2-24) needs no edit here.
    private fun versions(): Pair<String, String> {
        val all = PathMatchingResourcePatternResolver().getResources("classpath:db/migration/V*__*.sql")
            .map { it.filename!! }
        fun version(name: String) = name.removePrefix("V").substringBefore("__").toInt()
        val mine = version(all.single { it.endsWith("__batch_relevance.sql") })
        val previous = all.map(::version).filter { it < mine }.max()
        return previous.toString() to mine.toString()
    }

    // Copied from FlywayMigrationIntegrationTest (private there; that file belongs to other lanes too).
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
