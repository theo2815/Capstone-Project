package com.quickpitik.support

import jakarta.persistence.EntityManager
import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.photographer.HandlePatchRequest
import com.quickpitik.exception.ApiException
import com.quickpitik.service.photographer.PhotographerSettingsService
import org.flywaydb.core.Flyway
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.beans.factory.annotation.Value
import org.springframework.core.io.ClassPathResource
import org.springframework.core.io.support.PathMatchingResourcePatternResolver
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.jdbc.datasource.init.ScriptUtils
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith
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
