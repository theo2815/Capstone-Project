package com.quickpitik.support

import jakarta.persistence.EntityManager
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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

    @Test
    fun `every migration applied cleanly`() {
        val failed = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM flyway_schema_history WHERE success = false",
            Int::class.java,
        )
        assertEquals(0, failed, "a migration is recorded as failed in flyway_schema_history")

        val applied = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM flyway_schema_history WHERE success = true",
            Int::class.java,
        ) ?: 0
        // V1..V39 today. A floor rather than an equality so adding V40 doesn't
        // fail this test for the wrong reason.
        assertTrue(applied >= 39, "expected at least 39 applied migrations, found $applied")
    }

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
        val unverifiedAdmins = jdbcTemplate.queryForObject(
            "SELECT count(*) FROM users WHERE role = 'ADMIN' AND email_verified_at IS NULL",
            Int::class.java,
        )
        assertEquals(0, unverifiedAdmins, "the bootstrap admin should be stamped verified on creation")
    }
}
