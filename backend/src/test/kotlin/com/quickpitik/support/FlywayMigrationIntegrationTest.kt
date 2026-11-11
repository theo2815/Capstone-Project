package com.quickpitik.support

import jakarta.persistence.EntityManager
import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.HandlePatchRequest
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventReviewStatus
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.RaceType
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.EventRepository
import com.quickpitik.service.events.RaceEventService
import com.quickpitik.service.photographer.PhotographerSettingsService
import org.flywaydb.core.Flyway
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.beans.factory.annotation.Value
import org.springframework.core.io.ClassPathResource
import org.springframework.core.io.support.PathMatchingResourcePatternResolver
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.datasource.init.ScriptUtils
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import javax.sql.DataSource

/**
 * The migrations apply, in order, on a database that has never seen them — and
 * the entities still match the schema they produce.
 *
 * Most of the value is in the fact that this class loads at all. Flyway runs
 * V1→V39 during context startup, then Hibernate's `ddl-auto: validate` compares
 * every `@Entity` against the resulting tables and fails the context if a column
 * is missing, renamed, or the wrong type. Thirty migrations had been applied
 * only to the developer's long-lived local database, where a hand-patched
 * column can hide a broken migration indefinitely; this is the first thing that
 * proves a clean install works.
 *
 * The assertions below are deliberately thin — they exist so the failure
 * message names what broke, rather than the whole run dying on an opaque
 * context-load error.
 */
class FlywayMigrationIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var jdbcTemplate: JdbcTemplate

    @Autowired
    private lateinit var entityManager: EntityManager

    @Autowired
    private lateinit var photographerSettings: PhotographerSettingsService

    @Autowired
    private lateinit var dataSource: DataSource

    @Autowired
    private lateinit var events: EventRepository

    @Value("\${app.admin.bootstrap-email}")
    private lateinit var bootstrapEmail: String

    @Test
    fun `account route handles are reserved by the database and rejected by the service`() {
        for (handle in listOf("reset-password", "verify", "verify-email", "confirm-email-change")) {
            assertEquals(1, jdbcTemplate.queryForObject(
                "SELECT count(*) FROM reserved_handles WHERE handle = ?", Int::class.java, handle,
            ))
            val error = assertFailsWith<ApiException> {
                photographerSettings.putHandle(UUID.randomUUID(), HandlePatchRequest("  ${handle.uppercase()}  "))
            }
            assertEquals(ErrorCodes.RESERVED_HANDLE, error.code)
        }
    }

    @Test
    fun `every migration applied cleanly`() {
        val failed = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM flyway_schema_history WHERE success = false",
            Int::class.java,
        )
        assertEquals(0, failed, "a migration is recorded as failed in flyway_schema_history")

        val applied = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM flyway_schema_history WHERE success = true AND version IS NOT NULL",
            Int::class.java,
        ) ?: 0
        // Every versioned script shipped on the classpath, so a migration Flyway
        // skipped (misnamed, out of order) fails here without a floor to maintain.
        val shipped = PathMatchingResourcePatternResolver().getResources("classpath*:db/migration/V*.sql").size
        assertEquals(shipped, applied, "applied migrations do not match the V*.sql scripts on the classpath")
    }

    @Test
    fun `photos s3_key is indexed for the upload cleanup reference check`() {
        val definition = jdbcTemplate.queryForObject(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'photos' AND indexname = 'idx_photos_s3_key'",
            String::class.java,
        )
        assertTrue(definition!!.contains("(s3_key)"))
    }

    @Test
    fun `V63 backfills delivered receipts before creating the stale claim index`() {
        val schema = "receipt_v63_${UUID.randomUUID().toString().replace("-", "")}"
        dataSource.connection.use { connection ->
            connection.createStatement().use { statement ->
                statement.execute("CREATE SCHEMA $schema")
                statement.execute("SET search_path TO $schema")
                statement.execute("CREATE TABLE orders (id UUID PRIMARY KEY, email_sent_at TIMESTAMPTZ)")
                statement.execute(
                    "INSERT INTO orders (id, email_sent_at) VALUES " +
                        "('00000000-0000-0000-0000-000000000001', '2026-09-01T12:00:00Z'), " +
                        "('00000000-0000-0000-0000-000000000002', NULL)",
                )
            }

            try {
                ScriptUtils.executeSqlScript(
                    connection,
                    ClassPathResource("db/migration/V63__receipt_delivery_marker.sql"),
                )

                connection.prepareStatement(
                    "SELECT email_sent_at, receipt_delivered_at FROM orders WHERE id = ?",
                ).use { statement ->
                    statement.setObject(1, UUID.fromString("00000000-0000-0000-0000-000000000001"))
                    statement.executeQuery().use { result ->
                        result.next()
                        assertEquals(result.getObject("email_sent_at"), result.getObject("receipt_delivered_at"))
                    }
                    statement.setObject(1, UUID.fromString("00000000-0000-0000-0000-000000000002"))
                    statement.executeQuery().use { result ->
                        result.next()
                        assertNull(result.getObject("receipt_delivered_at"))
                    }
                }

                val indexDefinition = connection.prepareStatement(
                    "SELECT indexdef FROM pg_indexes WHERE schemaname = ? AND indexname = 'idx_orders_stale_receipt_claims'",
                ).use { statement ->
                    statement.setString(1, schema)
                    statement.executeQuery().use { result ->
                        result.next()
                        result.getString("indexdef")
                    }
                }
                assertTrue(indexDefinition.contains("(email_sent_at)"), indexDefinition)
                assertTrue(indexDefinition.contains("email_sent_at IS NOT NULL"), indexDefinition)
                assertTrue(indexDefinition.contains("receipt_delivered_at IS NULL"), indexDefinition)
            } finally {
                connection.createStatement().use {
                    it.execute("DROP SCHEMA $schema CASCADE")
                    it.execute("SET search_path TO public")
                }
            }
        }
    }

    @Test
    fun `V64 adds a separate photos-ready claim without changing delivered rows`() {
        val schema = "photo_alert_v64_${UUID.randomUUID().toString().replace("-", "")}"
        dataSource.connection.use { connection ->
            connection.createStatement().use { statement ->
                statement.execute("CREATE SCHEMA $schema")
                statement.execute("SET search_path TO $schema")
                statement.execute("CREATE TABLE event_photo_alerts (id UUID PRIMARY KEY, notified_at TIMESTAMPTZ)")
                statement.execute(
                    "INSERT INTO event_photo_alerts (id, notified_at) VALUES " +
                        "('00000000-0000-0000-0000-000000000001', '2026-09-01T12:00:00Z'), " +
                        "('00000000-0000-0000-0000-000000000002', NULL)",
                )
            }

            try {
                ScriptUtils.executeSqlScript(
                    connection,
                    ClassPathResource("db/migration/V64__event_photo_alert_notification_claim.sql"),
                )

                connection.prepareStatement(
                    "SELECT notified_at, notification_claimed_at FROM event_photo_alerts WHERE id = ?",
                ).use { statement ->
                    statement.setObject(1, UUID.fromString("00000000-0000-0000-0000-000000000001"))
                    statement.executeQuery().use { result ->
                        result.next()
                        assertEquals(
                            OffsetDateTime.parse("2026-09-01T12:00:00Z"),
                            result.getObject("notified_at", OffsetDateTime::class.java),
                        )
                        assertNull(result.getObject("notification_claimed_at"))
                    }
                }
            } finally {
                connection.createStatement().use {
                    it.execute("DROP SCHEMA $schema CASCADE")
                    it.execute("SET search_path TO public")
                }
            }
        }
    }

    @Test
    fun `photo upload intent schema is present`() {
        val columns = jdbcTemplate.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'photo_upload_intents'",
            String::class.java,
        )
        assertTrue(
            columns.containsAll(
                listOf(
                    "photo_id",
                    "photographer_id",
                    "event_id",
                    "request_id",
                    "content_hash",
                    "content_type",
                    "size_bytes",
                    "staging_key",
                    "original_key",
                    "state",
                    "lease_token",
                    "lease_until",
                    "result_photo_id",
                    "error_code",
                    "put_expires_at",
                    "expires_at",
                    "cleanup_after",
                    "cleanup_checked_at",
                    "cleaned_at",
                ),
            ),
        )

        val requestIndex = jdbcTemplate.queryForObject(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'uq_photo_upload_request'",
            String::class.java,
        ).orEmpty()
        assertTrue(requestIndex.contains("(photographer_id, request_id)"), requestIndex)
        assertTrue(requestIndex.contains("request_id IS NOT NULL"), requestIndex)
    }

    @Test
    fun `V65 upgrades V64 without changing existing photos or their hash constraint`() {
        val schema = "upload_intent_v65_${UUID.randomUUID().toString().replace("-", "")}"
        val ownerId = UUID.randomUUID()
        val photoId = UUID.randomUUID()
        val eventId = UUID.fromString("11111111-1111-1111-1111-111111111111")

        try {
            flywayFor(schema, "64").migrate()
            dataSource.connection.use { connection ->
                connection.createStatement().use { statement ->
                    statement.execute(
                        "INSERT INTO $schema.users (id, email, password_hash, name, role) VALUES " +
                            "('$ownerId', 'v65-upgrade@test.local', 'stub', 'V65 Upgrade', 'PHOTOGRAPHER')",
                    )
                    statement.execute(
                        "INSERT INTO $schema.photos " +
                            "(id, event_id, photographer_id, s3_key, status, price_php, content_hash) VALUES " +
                            "('$photoId', '$eventId', '$ownerId', 'legacy/original.jpg', 'LIVE', 100, '${"a".repeat(64)}')",
                    )
                }
            }
            val originalIndex = indexDefinition(schema, "uq_photos_photographer_content_hash")

            flywayFor(schema, "65").migrate()

            dataSource.connection.use { connection ->
                connection.prepareStatement(
                    "SELECT s3_key, content_hash FROM $schema.photos WHERE id = ?::uuid",
                ).use { statement ->
                    statement.setString(1, photoId.toString())
                    statement.executeQuery().use { result ->
                        assertTrue(result.next())
                        assertEquals("legacy/original.jpg", result.getString("s3_key"))
                        assertEquals("a".repeat(64), result.getString("content_hash"))
                    }
                }
            }
            assertEquals(originalIndex, indexDefinition(schema, "uq_photos_photographer_content_hash"))
            assertEquals(
                1,
                jdbcTemplate.queryForObject(
                    "SELECT count(*) FROM information_schema.tables " +
                        "WHERE table_schema = ? AND table_name = 'photo_upload_intents'",
                    Int::class.java,
                    schema,
                ),
            )
        } finally {
            jdbcTemplate.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    // Upgrade rehearsal: staging and production were on V89 when V90 was written.
    @Test
    fun `V90 upgrades V89 with two nullable refund link columns on disputes`() {
        val schema = "refund_link_v90_${UUID.randomUUID().toString().replace("-", "")}"
        try {
            flywayFor(schema, "89").migrate()
            flywayFor(schema, "90").migrate()

            val columns = jdbcTemplate.queryForList(
                "SELECT column_name, data_type, character_maximum_length, is_nullable " +
                    "FROM information_schema.columns WHERE table_schema = ? AND table_name = 'disputes' " +
                    "AND column_name IN ('refund_link_url', 'refund_link_sent_at') ORDER BY column_name",
                schema,
            ).map { listOf(it["column_name"], it["data_type"], it["character_maximum_length"], it["is_nullable"]) }
            assertEquals(
                listOf(
                    listOf("refund_link_sent_at", "timestamp with time zone", null, "YES"),
                    listOf("refund_link_url", "character varying", 500, "YES"),
                ),
                columns,
            )
        } finally {
            jdbcTemplate.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    // Upgrade rehearsal: staging was on V91 and production on V90 when V92 was written.
    @Test
    fun `V92 upgrades V91 with events and leaves every row unstamped`() {
        val schema = "face_expiry_v92_${UUID.randomUUID().toString().replace("-", "")}"
        val ids = mutableListOf<UUID>()
        try {
            flywayFor(schema, "91").migrate()
            val v91Columns = jdbcTemplate.queryForList(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = ? AND table_name = 'events' " +
                    "ORDER BY ordinal_position",
                String::class.java, schema,
            )
            assertFalse("faces_deleted_at" in v91Columns, "faces_deleted_at must not exist at V91")
            assertEquals("91", latestVersion(schema))

            // One old event and one deleted event, copied in from rows the V92 entity wrote.
            val old = events.save(Event(slug = "v92-old-${UUID.randomUUID().toString().take(8)}", name = "Old run",
                date = LocalDate.of(2025, 11, 1), location = "Cebu City", status = EventStatus.COMPLETED))
            val gone = events.save(Event(slug = "v92-del-${UUID.randomUUID().toString().take(8)}", name = "Deleted run",
                date = LocalDate.of(2026, 10, 30), location = "Cebu City", status = EventStatus.ACTIVE)
                .apply { deletedAt = OffsetDateTime.now() })
            ids += listOf(old.id, gone.id)
            val cols = v91Columns.joinToString(", ") { "\"$it\"" }
            assertEquals(2, jdbcTemplate.update(
                "INSERT INTO $schema.events ($cols) SELECT $cols FROM public.events WHERE id IN (?, ?)", old.id, gone.id,
            ))

            flywayFor(schema, "92").migrate()

            assertEquals("92", latestVersion(schema))
            val column = jdbcTemplate.queryForMap(
                "SELECT data_type, is_nullable, column_default FROM information_schema.columns " +
                    "WHERE table_schema = ? AND table_name = 'events' AND column_name = 'faces_deleted_at'",
                schema,
            )
            assertEquals("timestamp with time zone", column["data_type"])
            assertEquals("YES", column["is_nullable"])
            assertNull(column["column_default"])
            assertEquals(2, jdbcTemplate.queryForObject(
                "SELECT count(*) FROM $schema.events WHERE id IN (?, ?) AND faces_deleted_at IS NULL",
                Int::class.java, old.id, gone.id,
            ))
            // Nothing else moved: every V91 column of the copied rows is unchanged.
            assertEquals(0, jdbcTemplate.queryForObject(
                "SELECT count(*) FROM (SELECT $cols FROM public.events WHERE id IN (?, ?) " +
                    "EXCEPT SELECT $cols FROM $schema.events) d",
                Int::class.java, old.id, gone.id,
            ))
            assertEquals(true, jdbcTemplate.queryForObject(
                "SELECT bool_and(success) FROM $schema.flyway_schema_history WHERE version = '92'", Boolean::class.java,
            ))
        } finally {
            jdbcTemplate.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
            ids.forEach { jdbcTemplate.update("DELETE FROM public.events WHERE id = ?", it) }
        }
    }

    // Upgrade rehearsal: V93 seller details may already hold rows (staging, and production after the
    // release that carries V93) when V94 arrives.
    @Test
    fun `V94 upgrades V93 turning submitted seller details into manual review`() {
        val schema = "seller_v94_${UUID.randomUUID().toString().replace("-", "")}"
        try {
            flywayFor(schema, "93").migrate()
            fun seller(status: String): UUID {
                val id = jdbcTemplate.queryForObject(
                    "INSERT INTO $schema.users (email, password_hash, name, role) VALUES (?, 'x', 'V94 seller', 'PHOTOGRAPHER') RETURNING id",
                    UUID::class.java, "v94-${UUID.randomUUID()}@test.local",
                )!!
                jdbcTemplate.update(
                    "INSERT INTO $schema.photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, " +
                        "address, phone, status, rejection_reason, revision, submitted_at) " +
                        "VALUES (?, 'Juan Dela Cruz', 'PASSPORT', 'k/front', 'k/selfie', 'Cebu', '+63 917 123 4567', ?, ?, 1, now())",
                    id, status, if (status == "REJECTED") "Blurry ID" else null,
                )
                return id
            }
            val submitted = seller("SUBMITTED")
            val approved = seller("APPROVED")
            val rejected = seller("REJECTED")

            flywayFor(schema, "94").migrate()

            assertEquals("94", latestVersion(schema))
            fun row(id: UUID) = jdbcTemplate.queryForMap(
                "SELECT status, reason_codes::text AS reasons, repeat_matches::text AS matches, phone_key " +
                    "FROM $schema.photographer_seller_details WHERE user_id = ?", id,
            )
            assertEquals("MANUAL_REVIEW", row(submitted)["status"])
            assertEquals("APPROVED", row(approved)["status"])
            assertEquals("REJECTED", row(rejected)["status"])
            for (id in listOf(submitted, approved, rejected)) {
                assertEquals("[]", row(id)["reasons"])
                assertEquals("[]", row(id)["matches"])
                assertEquals("9171234567", row(id)["phone_key"])
            }
            // The old value is refused now; the new ones are accepted.
            assertFailsWith<DataIntegrityViolationException> {
                jdbcTemplate.update("UPDATE $schema.photographer_seller_details SET status = 'SUBMITTED' WHERE user_id = ?", submitted)
            }
            jdbcTemplate.update("UPDATE $schema.photographer_seller_details SET status = 'AUTO_APPROVED' WHERE user_id = ?", submitted)

            // The liveness table is shared with the runner photo-removal plan (RD-0.1).
            val liveness = jdbcTemplate.queryForList(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = ? AND table_name = 'liveness_sessions'",
                String::class.java, schema,
            ).toSet()
            assertEquals(
                setOf("session_id", "user_id", "purpose", "event_id", "consent_version", "created_at", "consumed_at"),
                liveness,
            )
        } finally {
            jdbcTemplate.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    // Upgrade rehearsal: staging is on V94 with events when V96 (AA-R1; V95 stays unused) arrives.
    @Test
    fun `V96 keeps existing events untyped and refuses race columns on untyped rows`() {
        val schema = "race_v96_${UUID.randomUUID().toString().replace("-", "")}"
        val ids = mutableListOf<UUID>()
        try {
            flywayFor(schema, "94").migrate()
            val v94Columns = jdbcTemplate.queryForList(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = ? AND table_name = 'events' " +
                    "ORDER BY ordinal_position",
                String::class.java, schema,
            )
            assertFalse("event_type" in v94Columns, "event_type must not exist at V94")
            val owner = jdbcTemplate.queryForObject(
                "INSERT INTO $schema.users (email, password_hash, name, role) VALUES (?, 'x', 'V96 owner', 'PHOTOGRAPHER') RETURNING id",
                UUID::class.java, "v96-${UUID.randomUUID()}@test.local",
            )!!
            // An admin event and a photographer-owned event, copied in from rows the V96 entity wrote.
            val admin = events.save(Event(slug = "v96-admin-${UUID.randomUUID().toString().take(8)}", name = "Cebu Marathon",
                date = LocalDate.of(2026, 1, 11), location = "Cebu City", status = EventStatus.COMPLETED))
            val owned = events.save(Event(slug = "v96-own-${UUID.randomUUID().toString().take(8)}", name = "Wedding shoot",
                date = LocalDate.of(2026, 10, 30), location = "Cebu City", status = EventStatus.ACTIVE))
            ids += listOf(admin.id, owned.id)
            val cols = v94Columns.joinToString(", ") { "\"$it\"" }
            assertEquals(2, jdbcTemplate.update(
                "INSERT INTO $schema.events ($cols) SELECT $cols FROM public.events WHERE id IN (?, ?)", admin.id, owned.id,
            ))
            jdbcTemplate.update("UPDATE $schema.events SET created_by = ? WHERE id = ?", owner, owned.id)

            flywayFor(schema, "96").migrate()

            assertEquals("96", latestVersion(schema))
            assertEquals(2, jdbcTemplate.queryForObject(
                "SELECT count(*) FROM $schema.events WHERE id IN (?, ?) AND event_type IS NULL AND submitted_by IS NULL " +
                    "AND name_key IS NULL AND possible_duplicate_of IS NULL AND admin_approved_at IS NULL " +
                    "AND hidden_at IS NULL AND proof_checked_at IS NULL AND check_reasons IS NULL",
                Int::class.java, admin.id, owned.id,
            ))
            // ck_events_race_only: an untyped row takes no race column.
            for (set in listOf("hidden_at = now()", "submitted_by = '$owner'", "name_key = 'cebu marathon'",
                "admin_approved_at = now()", "proof_checked_at = now()", "check_reasons = '[]'",
                "possible_duplicate_of = '${owned.id}'")) {
                assertFailsWith<DataIntegrityViolationException>(set) {
                    jdbcTemplate.update("UPDATE $schema.events SET $set WHERE id = ?", admin.id)
                }
            }
            // ck_events_race_shared: a photographer-owned event cannot become a race; an admin event can.
            assertFailsWith<DataIntegrityViolationException> {
                jdbcTemplate.update("UPDATE $schema.events SET event_type = 'ROAD_RACE' WHERE id = ?", owned.id)
            }
            assertFailsWith<DataIntegrityViolationException> {
                jdbcTemplate.update("UPDATE $schema.events SET event_type = 'TRIATHLON' WHERE id = ?", admin.id)
            }
            assertEquals(1, jdbcTemplate.update(
                "UPDATE $schema.events SET event_type = 'ROAD_RACE', hidden_at = now(), check_reasons = '[\"DAILY_CAP\"]' WHERE id = ?",
                admin.id,
            ))
            jdbcTemplate.update("INSERT INTO $schema.event_name_aliases (event_id, name_key, added_by) VALUES (?, 'cebu marathon', ?)",
                admin.id, owner)
            jdbcTemplate.update("DELETE FROM $schema.events WHERE id = ?", admin.id)
            assertEquals(0, jdbcTemplate.queryForObject(
                "SELECT count(*) FROM $schema.event_name_aliases", Int::class.java,
            ), "aliases go with their event")
        } finally {
            jdbcTemplate.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
            ids.forEach { jdbcTemplate.update("DELETE FROM public.events WHERE id = ?", it) }
        }
    }

    // Upgrade rehearsal: staging holds support-made removal cases when V99 (RD-1) arrives. Starts at V97
    // (AA-R10's index took V102, not V98).
    @Test
    fun `V99 upgrades earlier versions with existing removal cases defaulting to SUPPORT`() {
        val schema = "removal_v99_${UUID.randomUUID().toString().replace("-", "")}"
        try {
            flywayFor(schema, "97").migrate()
            val case = UUID.randomUUID()
            jdbcTemplate.update(
                "INSERT INTO $schema.photo_removal_requests (id, event_id_snapshot, event_name_snapshot, requester_contact, " +
                    "support_reference, requester_reason, public_to_owner_reason, status, created_at, updated_at, creator_id, " +
                    "creation_command_id) VALUES (?, ?, 'Cebu Marathon', 'support@example.test', 'SUP-1', 'Privacy', " +
                    "'Privacy removal request', 'OPEN', now(), now(), ?, ?)",
                case, UUID.randomUUID(), UUID.randomUUID(), UUID.randomUUID(),
            )

            flywayFor(schema, "99").migrate()

            assertEquals("99", latestVersion(schema))
            assertEquals(1, jdbcTemplate.queryForObject(
                "SELECT count(*) FROM $schema.photo_removal_requests WHERE id = ? AND source = 'SUPPORT' " +
                    "AND risk_flags = '[]'::jsonb AND reason_code IS NULL AND consent_version IS NULL AND consent_at IS NULL " +
                    "AND liveness_session_id IS NULL AND liveness_band IS NULL AND liveness_confidence IS NULL AND close_reason IS NULL",
                Int::class.java, case,
            ))
            assertEquals(listOf("YES"), jdbcTemplate.queryForList(
                "SELECT is_nullable FROM information_schema.columns WHERE table_schema = ? AND table_name = 'photos' " +
                    "AND column_name = 'face_count' AND column_default IS NULL",
                String::class.java, schema,
            ))
            assertFailsWith<DataIntegrityViolationException> {
                jdbcTemplate.update("UPDATE $schema.photo_removal_requests SET source = 'EMAIL' WHERE id = ?", case)
            }
            assertFailsWith<DataIntegrityViolationException> {
                jdbcTemplate.update("UPDATE $schema.photo_removal_requests SET risk_flags = '{}'::jsonb WHERE id = ?", case)
            }
        } finally {
            jdbcTemplate.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    // AA-R10: one live self-serve race per date and name key, the race lock's last line of defence.
    @Test
    fun `two self-serve races on one date and key are refused`() {
        val key = raceKey()
        race(key, photographer())
        val error = assertFailsWith<DataIntegrityViolationException> { race(key, photographer()) }
        assertTrue(RaceEventService.RACE_UNIQUE_INDEX in error.mostSpecificCause.message.orEmpty())
    }

    @Test
    fun `an admin race with the same key is allowed`() {
        val key = raceKey()
        race(key, photographer())
        race(key, submittedBy = null)
        race(key, submittedBy = null)
    }

    @Test
    fun `a different-race decision with the same key is allowed`() {
        val key = raceKey()
        val first = race(key, photographer())
        race(key, photographer(), duplicateOf = first.id)
    }

    @Test
    fun `a rejected race does not block a new race with the same name and date`() {
        val key = raceKey()
        race(key, photographer(), review = EventReviewStatus.REJECTED)
        race(key, photographer())
    }

    // Upgrade rehearsal: staging is on V100 with race rows when V102 arrives (V98 was passed by V99 and V100;
    // V101 is IX-W3c's and runs in between once it merges).
    @Test
    fun `V102 upgrades V100 with existing races and indexes only self-serve ones`() {
        val schema = "race_v102_${UUID.randomUUID().toString().replace("-", "")}"
        try {
            flywayFor(schema, "100").migrate()
            val key = raceKey()
            val selfServe = race(key, photographer())
            val admin = race(key, submittedBy = null)
            val cols = jdbcTemplate.queryForList(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = ? AND table_name = 'events'",
                String::class.java, schema,
            ).joinToString(", ") { "\"$it\"" }
            // submitted_by points at public.users, so the copy drops it into the schema's own users first.
            jdbcTemplate.update("INSERT INTO $schema.users SELECT * FROM public.users WHERE id = ?", selfServe.submittedBy)
            assertEquals(2, jdbcTemplate.update(
                "INSERT INTO $schema.events ($cols) SELECT $cols FROM public.events WHERE id IN (?, ?)", selfServe.id, admin.id,
            ))

            flywayFor(schema, "102").migrate()

            assertEquals("102", latestVersion(schema))
            assertTrue("WHERE" in indexDefinition(schema, RaceEventService.RACE_UNIQUE_INDEX))
        } finally {
            jdbcTemplate.execute("DROP SCHEMA IF EXISTS $schema CASCADE")
        }
    }

    private fun raceKey() = "unique run ${UUID.randomUUID()}"

    private fun photographer(): UUID = jdbcTemplate.queryForObject(
        "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'x', 'Race owner', 'PHOTOGRAPHER') RETURNING id",
        UUID::class.java, "r10-${UUID.randomUUID()}@test.local",
    )!!

    private fun race(
        key: String,
        submittedBy: UUID?,
        review: EventReviewStatus = EventReviewStatus.APPROVED,
        duplicateOf: UUID? = null,
    ): Event = events.saveAndFlush(
        Event(slug = "r10-${UUID.randomUUID()}", name = key, date = LocalDate.of(1905, 6, 4), location = "Cebu City",
            status = EventStatus.ACTIVE, eventType = RaceType.ROAD_RACE, reviewStatus = review)
            .apply { this.submittedBy = submittedBy; nameKey = key; possibleDuplicateOf = duplicateOf },
    )

    private fun latestVersion(schema: String): String = jdbcTemplate.queryForObject(
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

    private fun indexDefinition(schema: String, index: String): String = jdbcTemplate.queryForObject(
        "SELECT indexdef FROM pg_indexes WHERE schemaname = ? AND indexname = ?",
        String::class.java,
        schema,
        index,
    ).orEmpty()

    // The columns V29 and V30 add — the two this session introduced, and the
    // ones most likely to drift from their entities.
    @Test
    fun `the lockout and verification columns exist on users`() {
        val columns = jdbcTemplate.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'users'",
            String::class.java,
        )
        assertTrue(columns.containsAll(listOf("failed_login_attempts", "locked_until", "email_verified_at")))
    }

    @Test
    fun `the verification token table exists`() {
        val count = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'email_verification_tokens'",
            Int::class.java,
        )
        assertEquals(1, count)
    }

    // V45/V47/V49: one coupon per covered (event, photographer), with usage
    // reservation on orders.
    @Test
    fun `photographer coupon schema is present`() {
        val couponColumns = jdbcTemplate.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'photographer_coupons'",
            String::class.java,
        )
        assertTrue(
            couponColumns.containsAll(
                listOf("id", "event_id", "photographer_id", "code", "percent_off", "active", "expires_at", "usage_limit"),
            ),
        )

        fun columns(table: String) = jdbcTemplate.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_name = '$table'",
            String::class.java,
        )
        assertTrue(columns("orders").containsAll(listOf("coupon_code", "coupon_id")))
        assertTrue(columns("order_items").containsAll(listOf("discount_php", "coupon_id")))
        assertTrue("discount_php" in columns("transactions"))
        // V50: coupons are attributed per item so two photographers' coupons
        // can share one event order; usage counts distinct orders through items.
        val itemCouponFk = jdbcTemplate.queryForObject(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'fk_order_items_coupon'",
            String::class.java,
        ).orEmpty()
        assertTrue(itemCouponFk.contains("REFERENCES photographer_coupons(id) ON DELETE SET NULL"), itemCouponFk)

        // V49 replaced the events(id, created_by) FK with one on coverage.
        val coverageConstraint = jdbcTemplate.queryForObject(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'fk_photographer_coupons_coverage'",
            String::class.java,
        ).orEmpty()
        assertTrue(coverageConstraint.contains("REFERENCES event_photographer(event_id, photographer_id)"), coverageConstraint)
        val ownerConstraints = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM pg_constraint WHERE conname = 'fk_photographer_coupons_owned_event'",
            Int::class.java,
        )
        assertEquals(0, ownerConstraints)
        val perPhotographerIndex = jdbcTemplate.queryForObject(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'uq_photographer_coupons_event_photographer'",
            String::class.java,
        ).orEmpty()
        assertTrue(perPhotographerIndex.contains("(event_id, photographer_id)"), perPhotographerIndex)
    }

    // V46: photographer-owned events + the four event review message kinds.
    @Test
    fun `photographer-owned event schema is present`() {
        val eventColumns = jdbcTemplate.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'events'",
            String::class.java,
        )
        assertTrue(
            eventColumns.containsAll(
                listOf("created_by", "visibility", "pricing_mode", "watermark_policy", "review_status", "pending_change", "review_note"),
            ),
        )
        val kindCheck = jdbcTemplate.queryForObject(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'photographer_messages_kind_check'",
            String::class.java,
        )
        assertTrue(kindCheck!!.contains("event_change_rejected"), kindCheck)
    }

    @Test
    fun `checkout hardening schema is present`() {
        val orderColumns = jdbcTemplate.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'orders'",
            String::class.java,
        )
        assertTrue("legacy_share_token_hash" in orderColumns)
        assertTrue("share_token" !in orderColumns)

        val paymentColumns = jdbcTemplate.queryForList(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'payments'",
            String::class.java,
        )
        assertTrue("provider_payment_id" in paymentColumns)
        assertTrue("expires_at" in paymentColumns)

        val paymentMethodCheck = jdbcTemplate.queryForObject(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'orders_payment_method_check'",
            String::class.java,
        ).orEmpty()
        assertTrue(paymentMethodCheck.contains("qrph"), paymentMethodCheck)

        val indexes = jdbcTemplate.queryForList(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'orders'",
            String::class.java,
        )
        assertTrue(
            indexes.containsAll(
                listOf(
                    "uq_orders_user_idempotency_key_event",
                    "uq_orders_guest_idempotency_key_event",
                ),
            ),
        )
        assertTrue("uq_orders_idempotency_key_event" !in indexes)

        val statusCheck = jdbcTemplate.queryForObject(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'orders_status_check'",
            String::class.java,
        ).orEmpty()
        assertTrue("EXPIRED" in statusCheck)
    }

    // V41 widens the status CHECK. `DROP CONSTRAINT IF EXISTS` would silently
    // no-op on a misnamed constraint and leave the old one rejecting
    // 'escalated' at runtime - so assert both the widened definition and that
    // exactly one status check survived.
    @Test
    fun `flags accept the escalated status`() {
        val statusCheck = jdbcTemplate.queryForObject(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = 'flags_status_check'",
            String::class.java,
        ).orEmpty()
        assertTrue("escalated" in statusCheck, "flags_status_check does not allow 'escalated': $statusCheck")

        val statusChecks = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM pg_constraint WHERE conrelid = 'flags'::regclass AND contype = 'c' " +
                "AND pg_get_constraintdef(oid) LIKE '%status%'",
            Int::class.java,
        )
        assertEquals(1, statusChecks, "V41 left more than one status CHECK on flags")
    }

    // Redundant with ddl-auto=validate, which already ran at context load — but
    // it names the mapping explicitly, so a future entity/migration drift on
    // User reports here instead of only in a startup stack trace.
    @Test
    fun `the User entity mapping round-trips against the real schema`() {
        val mapped = entityManager
            .createQuery("SELECT count(u) FROM User u", java.lang.Long::class.java)
            .singleResult
        // BootstrapAdminRunner has run against this container.
        assertTrue(mapped.toLong() >= 1)
    }

    /**
     * The bootstrap admin lands verified on a fresh install.
     *
     * Only observable here. `BootstrapAdminRunner` no-ops the moment any ADMIN
     * exists, so a long-lived developer database — where the admin was created
     * long before V30 — can never exercise this branch and will show the admin
     * as unverified forever. A virgin container is the only place the stamp
     * actually runs. It matters because the address is operator-provisioned
     * from env and the default `admin@quickpitik.local` is not a deliverable
     * inbox, so nothing would ever verify it.
     */
    @Test
    fun `the bootstrap admin is created already verified`() {
        // Scoped to the bootstrap row: other classes share this database.
        val verifiedBootstrapAdmins = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM users WHERE email = ? AND role = 'ADMIN' AND email_verified_at IS NOT NULL",
            Int::class.java,
            bootstrapEmail,
        )
        assertEquals(1, verifiedBootstrapAdmins, "the bootstrap admin should be stamped verified on creation")
    }
}
