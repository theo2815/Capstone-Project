package com.quickpitik.support

import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.testcontainers.containers.Container
import org.testcontainers.utility.MountableFile
import java.nio.file.Path
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/** Runs `scripts/delete-runner-account.sql` with psql inside the test database container. */
class AccountDeletionScriptIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate

    @BeforeEach
    fun copyScript() {
        postgres.copyFileToContainer(
            MountableFile.forHostPath(Path.of("scripts/delete-runner-account.sql").toAbsolutePath()),
            "/tmp/delete-runner-account.sql",
        )
    }

    private fun run(userId: Any?, confirm: String? = null): Container.ExecResult {
        val args = mutableListOf("psql", "-X", "-U", "quickpitik", "-d", "quickpitik_test")
        if (userId != null) args += listOf("-v", "user_id=$userId")
        if (confirm != null) args += listOf("-v", "confirm=$confirm")
        args += listOf("-f", "/tmp/delete-runner-account.sql")
        return postgres.execInContainer(*args.toTypedArray())
    }

    private fun id(sql: String, vararg params: Any): UUID = jdbc.queryForObject("$sql RETURNING id", UUID::class.java, *params)!!

    private fun count(sql: String, vararg params: Any): Int = jdbc.queryForObject(sql, Int::class.java, *params)!!

    private fun user(role: String) = id(
        "INSERT INTO users (email, password_hash, name, role) VALUES (?, 'unused', 'Del Test', ?)",
        "del-${UUID.randomUUID()}@test.local", role,
    )

    private class Seed(val runner: UUID, val photographer: UUID, val order: UUID, val selfieKeys: List<String>, val avatarKey: String)

    /** A runner with every kind of row the account can own, plus one purchase from a photographer. */
    private fun seedRunner(): Seed {
        val runner = user("RUNNER")
        val photographer = user("PHOTOGRAPHER")
        val avatarKey = "avatars/$runner/${UUID.randomUUID()}.jpg"
        jdbc.update("UPDATE users SET avatar_s3_key = ? WHERE id = ?", avatarKey, runner)
        val event = id(
            "INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Del Run', current_date, 'Cebu', 'ACTIVE')",
            "del-${UUID.randomUUID()}",
        )
        val photo = id("INSERT INTO photos (event_id, photographer_id, s3_key, price_php) VALUES (?, ?, ?, 125)", event, photographer, "del/${UUID.randomUUID()}")
        val selfieKeys = List(2) { "selfies/$runner/${UUID.randomUUID()}.jpg" }
        val selfie = selfieKeys.map { id("INSERT INTO user_selfies (user_id, s3_key) VALUES (?, ?)", runner, it) }.first()
        jdbc.update("INSERT INTO event_photo_alerts (event_id, user_id, selfie_id) VALUES (?, ?, ?)", event, runner, selfie)
        jdbc.update("INSERT INTO saved_events (user_id, event_id) VALUES (?, ?)", runner, event)
        jdbc.update("INSERT INTO cart_items (user_id, photo_id, event_id, price_php_at_add) VALUES (?, ?, ?, 125)", runner, photo, event)
        jdbc.update("INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (?, ?, now() + interval '1 day')", runner, "del-${UUID.randomUUID()}")
        jdbc.update(
            "INSERT INTO email_change_tokens (user_id, new_email, token_hash, expires_at) VALUES (?, 'new@test.local', ?, now() + interval '1 day')",
            runner, "del-${UUID.randomUUID()}",
        )
        val order = id(
            "INSERT INTO orders (user_id, event_id, recipient_email, payment_method, status, total_php) VALUES (?, ?, 'buyer@test.local', 'qrph', 'PAID', 125)",
            runner, event,
        )
        jdbc.update("INSERT INTO order_items (order_id, photo_id, price_php_at_purchase) VALUES (?, ?, 125)", order, photo)
        jdbc.update("INSERT INTO payments (order_id, provider, amount_php, status) VALUES (?, 'paymongo', 125, 'SUCCEEDED')", order)
        jdbc.update(
            "INSERT INTO transactions (paid_at, photographer_id, event_id, photo_id, order_id, amount_kept_php, buyer_id, buyer_display_name) " +
                "VALUES (now(), ?, ?, ?, ?, 93.75, ?, 'Del T.')",
            photographer, event, photo, order, runner,
        )
        jdbc.update(
            "INSERT INTO disputes (order_id, photo_id, reason, runner_id, refund_requested_by, note) VALUES (?, ?, 'wrong_runner', ?, ?, 'not me')",
            order, photo, runner, runner,
        )
        jdbc.update("INSERT INTO flags (target_kind, target_id, reason, reporter_id) VALUES ('photo', ?, 'other', ?)", photo, runner)
        return Seed(runner, photographer, order, selfieKeys, avatarKey)
    }

    private fun ownedRows(runner: UUID) = count(
        "SELECT (SELECT count(*) FROM users WHERE id = ?) + (SELECT count(*) FROM user_selfies WHERE user_id = ?) + " +
            "(SELECT count(*) FROM refresh_tokens WHERE user_id = ?) + (SELECT count(*) FROM orders WHERE user_id = ?) + " +
            "(SELECT count(*) FROM transactions WHERE buyer_id = ?) + (SELECT count(*) FROM disputes WHERE runner_id = ?)",
        runner, runner, runner, runner, runner, runner,
    )

    @Test
    fun `dry run lists the R2 keys and changes nothing`() {
        val seed = seedRunner()
        val before = ownedRows(seed.runner)

        val dry = run(seed.runner)

        assertEquals(0, dry.exitCode, dry.stderr)
        (seed.selfieKeys + seed.avatarKey).forEach { assertTrue(it in dry.stdout, "missing R2 key $it in:\n${dry.stdout}") }
        assertTrue("DRY RUN" in dry.stdout, dry.stdout)
        assertEquals(before, ownedRows(seed.runner))
        assertEquals(7, before) // the seed really is there: user, 2 selfies, token, order, ledger row, dispute
    }

    @Test
    fun `confirmed run deletes the runner and keeps money rows with the buyer link cleared`() {
        val seed = seedRunner()
        val r = seed.runner

        val result = run(r, confirm = "yes")

        assertEquals(0, result.exitCode, result.stderr)
        assertEquals(0, count("SELECT count(*) FROM users WHERE id = ?", r))
        for (table in listOf("user_selfies", "event_photo_alerts", "saved_events", "cart_items", "refresh_tokens", "email_change_tokens")) {
            assertEquals(0, count("SELECT count(*) FROM $table WHERE user_id = ?", r), table)
        }
        assertEquals("|buyer@test.local", jdbc.queryForObject("SELECT coalesce(user_id::text, '') || '|' || recipient_email FROM orders WHERE id = ?", String::class.java, seed.order))
        assertEquals(1, count("SELECT count(*) FROM order_items WHERE order_id = ?", seed.order))
        assertEquals(1, count("SELECT count(*) FROM payments WHERE order_id = ?", seed.order))
        assertEquals(1, count("SELECT count(*) FROM transactions WHERE order_id = ? AND buyer_id IS NULL AND buyer_display_name = 'Del T.'", seed.order))
        assertEquals(1, count("SELECT count(*) FROM disputes WHERE order_id = ? AND runner_id IS NULL AND refund_requested_by IS NULL", seed.order))
        assertEquals(1, count("SELECT count(*) FROM flags f JOIN disputes d ON d.photo_id = f.target_id WHERE d.order_id = ? AND f.reporter_id IS NULL", seed.order))
        assertEquals(1, count("SELECT count(*) FROM users WHERE id = ?", seed.photographer))
    }

    @Test
    fun `admins photographers and unknown ids are refused with no change`() {
        val admin = user("ADMIN")
        val photographer = seedRunner().photographer
        for (target in listOf(admin, photographer, UUID.randomUUID())) {
            val result = run(target, confirm = "yes")
            assertNotEquals(0, result.exitCode, "expected a refusal for $target:\n${result.stdout}")
            assertTrue("Refused" in result.stderr, result.stderr)
        }
        assertEquals(2, count("SELECT count(*) FROM users WHERE id IN (?, ?)", admin, photographer))
        assertEquals(1, count("SELECT count(*) FROM transactions WHERE photographer_id = ?", photographer))
    }

    @Test
    fun `a runner who still owns photographer sales is refused`() {
        // A role changed by hand would let the delete cascade away a sales ledger.
        val seed = seedRunner()
        jdbc.update("UPDATE users SET role = 'RUNNER' WHERE id = ?", seed.photographer)

        val result = run(seed.photographer, confirm = "yes")

        assertNotEquals(0, result.exitCode, result.stdout)
        assertTrue("Refused" in result.stderr, result.stderr)
        assertEquals(1, count("SELECT count(*) FROM transactions WHERE photographer_id = ?", seed.photographer))
    }

    @Test
    fun `a runner who still has a photographer payout account is refused`() {
        val former = user("RUNNER")
        jdbc.update(
            "INSERT INTO photographer_payout_accounts (id, user_id, method, account_number, account_name) VALUES (?, ?, 'GCASH', '09170000000', 'Del T')",
            UUID.randomUUID(), former,
        )

        val result = run(former, confirm = "yes")

        assertNotEquals(0, result.exitCode, result.stdout)
        assertEquals(1, count("SELECT count(*) FROM photographer_payout_accounts WHERE user_id = ?", former))
    }

    @Test
    fun `a confirm value other than yes changes nothing`() {
        val seed = seedRunner()
        run(seed.runner, confirm = "maybe")
        assertEquals(7, ownedRows(seed.runner))
    }

    @Test
    fun `missing user_id stops before anything runs`() {
        assertNotEquals(0, run(null, confirm = "yes").exitCode)
    }
}
