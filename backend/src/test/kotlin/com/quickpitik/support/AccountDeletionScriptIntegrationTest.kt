package com.quickpitik.support

import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.service.photographer.PublicPhotographerService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.support.TransactionTemplate
import org.testcontainers.containers.Container
import org.testcontainers.utility.MountableFile
import java.nio.file.Path
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Runs `scripts/delete-runner-account.sql` and `scripts/anonymise-photographer-account.sql`
 * with psql inside the test database container.
 */
class AccountDeletionScriptIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var jdbc: JdbcTemplate
    @Autowired private lateinit var publicPhotographers: PublicPhotographerService
    @Autowired private lateinit var relevance: BatchRelevanceRepository
    @Autowired private lateinit var tx: TransactionTemplate

    @BeforeEach
    fun copyScripts() {
        for (script in listOf(RUNNER_SCRIPT, PHOTOGRAPHER_SCRIPT)) {
            postgres.copyFileToContainer(MountableFile.forHostPath(Path.of("scripts/$script").toAbsolutePath()), "/tmp/$script")
        }
    }

    private fun run(userId: Any?, confirm: String? = null, script: String = RUNNER_SCRIPT): Container.ExecResult {
        val args = mutableListOf("psql", "-X", "-U", "quickpitik", "-d", "quickpitik_test")
        if (userId != null) args += listOf("-v", "user_id=$userId")
        if (confirm != null) args += listOf("-v", "confirm=$confirm")
        args += listOf("-f", "/tmp/$script")
        return postgres.execInContainer(*args.toTypedArray())
    }

    private fun anonymise(userId: Any?, confirm: String? = null) = run(userId, confirm, PHOTOGRAPHER_SCRIPT)

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

    // ─── DEL-1b: photographer accounts are anonymised and suspended, never deleted ───

    private class Studio(
        val photographer: UUID, val handle: String, val soldPhoto: UUID, val event: UUID, val order: UUID,
        val r2Keys: List<String>, val snapshotQrKey: String,
    )

    /**
     * An approved photographer with one sold photo that is fully paid out, one unsold photo
     * already erased, and every kind of personal row the account can own.
     */
    private fun seedStudio(): Studio {
        val p = user("PHOTOGRAPHER")
        val buyer = user("RUNNER")
        val handle = "del-${UUID.randomUUID().toString().take(8)}"
        val tag = UUID.randomUUID()
        val cover = "photographers/$p/cover/$tag.jpg"
        val watermark = "photographers/$p/watermark/$tag.png"
        val avatar = "avatars/$p/$tag.jpg"
        val liveQr = "photographers/$p/payouts/a1/$tag.png"
        val snapshotQr = "photographers/$p/payouts/a0/$tag.png"
        jdbc.update("UPDATE users SET avatar_s3_key = ?, avatar_url = 'https://x/a.jpg', google_sub = ? WHERE id = ?", avatar, "g-$tag", p)
        jdbc.update(
            "INSERT INTO photographer_settings (user_id, handle, brand_name, bio, city, cover_s3_key, watermark_s3_key, watermark_label, verification_status) " +
                "VALUES (?, ?, 'Del Studio', 'I shoot races', 'Cebu City', ?, ?, '@del', 'APPROVED')",
            p, handle, cover, watermark,
        )
        val event = id(
            "INSERT INTO events (slug, name, date, location, status) VALUES (?, 'Del Run', current_date, 'Cebu', 'ACTIVE')",
            "del-${UUID.randomUUID()}",
        )
        jdbc.update("UPDATE events SET photo_count = 1 WHERE id = ?", event)
        jdbc.update("INSERT INTO event_photographer (event_id, photographer_id, photo_count) VALUES (?, ?, 1)", event, p)
        val sold = id("INSERT INTO photos (event_id, photographer_id, s3_key, price_php) VALUES (?, ?, ?, 125)", event, p, "del/${UUID.randomUUID()}")
        jdbc.update(
            "INSERT INTO photos (event_id, photographer_id, s3_key, price_php, deleted_at, deletion_started_at, purged_at) " +
                "VALUES (?, ?, '', 125, now(), now(), now())",
            event, p,
        )
        val order = id(
            "INSERT INTO orders (user_id, event_id, recipient_email, payment_method, status, total_php, paid_at) VALUES (?, ?, 'buyer@test.local', 'qrph', 'PAID', 125, now())",
            buyer, event,
        )
        jdbc.update("INSERT INTO order_items (order_id, photo_id, price_php_at_purchase) VALUES (?, ?, 125)", order, sold)
        jdbc.update("INSERT INTO download_grants (order_id, photo_id, granted_until) VALUES (?, ?, now() + interval '30 days')", order, sold)
        jdbc.update(
            "INSERT INTO transactions (paid_at, photographer_id, event_id, photo_id, order_id, amount_kept_php, buyer_id, buyer_display_name) " +
                "VALUES (now() - interval '30 days', ?, ?, ?, ?, 93.75, ?, 'Del T.')",
            p, event, sold, order, buyer,
        )
        jdbc.update(
            "INSERT INTO payout_cycles (id, photographer_id, week_of, amount_php, status, method, account_snapshot_name, account_snapshot_qr_key) " +
                "VALUES (?, ?, current_date, 93.75, 'paid', 'gcash', 'Del Real Name', ?)",
            "del-${UUID.randomUUID()}", p, snapshotQr,
        )
        jdbc.update(
            "INSERT INTO photographer_payout_accounts (id, user_id, method, account_number, account_name, qr_s3_key) VALUES (?, ?, 'GCASH', '09170000000', 'Del Real Name', ?)",
            UUID.randomUUID(), p, liveQr,
        )
        jdbc.update(
            "INSERT INTO photographer_payout_accounts (id, user_id, method, account_number, account_name, qr_s3_key, is_primary) VALUES (?, ?, 'MAYA', '09180000000', 'Del Real Name', ?, false)",
            UUID.randomUUID(), p, snapshotQr,
        )
        jdbc.update("INSERT INTO payout_account_audit (user_id, action) VALUES (?, 'create')", p)
        jdbc.update("INSERT INTO social_links (id, user_id, platform, url) VALUES (?, ?, 'INSTAGRAM', 'https://instagram.com/del')", UUID.randomUUID(), p)
        jdbc.update("INSERT INTO photographer_coupons (photographer_id, code, percent_off, event_id) VALUES (?, ?, 10, ?)", p, "D${tag.toString().take(8)}", event)
        jdbc.update("INSERT INTO photographer_messages (photographer_id, kind, body) VALUES (?, 'admin_message', 'hello')", p)
        jdbc.update("INSERT INTO refresh_tokens (user_id, token_hash, expires_at) VALUES (?, ?, now() + interval '1 day')", p, "del-${UUID.randomUUID()}")
        jdbc.update("INSERT INTO desktop_auth_codes (user_id, code_hash, code_challenge, expires_at) VALUES (?, ?, 'c', now() + interval '1 minute')", p, "del-${UUID.randomUUID()}")
        val staging = "staging/photo-uploads/$p/$event/$tag/original"
        val leftoverOriginal = "events/$event/photos/$tag/original.jpg"
        jdbc.update(
            "INSERT INTO photo_upload_intents (photo_id, photographer_id, event_id, content_hash, content_type, size_bytes, staging_key, original_key, state, " +
                "created_at, put_expires_at, expires_at, updated_at) VALUES (?, ?, ?, ?, 'image/jpeg', 10, ?, ?, 'EXPIRED', now(), now(), now(), now())",
            UUID.randomUUID(), p, event, "a".repeat(64), staging, leftoverOriginal,
        )
        return Studio(p, handle, sold, event, order, listOf(cover, watermark, avatar, liveQr, "staging/photo-uploads/$p/", leftoverOriginal), snapshotQr)
    }

    private fun email(userId: UUID): String = jdbc.queryForObject("SELECT email FROM users WHERE id = ?", String::class.java, userId)!!

    @Test
    fun `photographer dry run lists the R2 files to delete, keeps the payout snapshot QR, and changes nothing`() {
        val s = seedStudio()
        val before = email(s.photographer)

        val dry = anonymise(s.photographer)

        assertEquals(0, dry.exitCode, dry.stderr)
        s.r2Keys.forEach { assertTrue(it in dry.stdout, "missing R2 key $it in:\n${dry.stdout}") }
        val toDelete = dry.stdout.substringAfter("R2 files to delete").substringBefore("Kept in R2")
        assertTrue(s.snapshotQrKey !in toDelete, dry.stdout)
        assertTrue(s.snapshotQrKey in dry.stdout.substringAfter("Kept in R2"), dry.stdout)
        assertTrue("DRY RUN" in dry.stdout, dry.stdout)
        assertEquals(before, email(s.photographer))
        assertEquals(1, count("SELECT count(*) FROM social_links WHERE user_id = ?", s.photographer))
    }

    private val deletedHandle = { p: UUID -> "deleted-" + p.toString().replace("-", "") }

    /** What another photographer's account looks like, to prove the run touched only one. */
    private fun fingerprint(p: UUID): String = jdbc.queryForObject(
        "SELECT u.email || u.name || u.password_hash || coalesce(u.suspended_at::text, '-') || s.handle || s.bio || " +
            "(SELECT count(*) FROM social_links WHERE user_id = u.id) || (SELECT count(*) FROM photographer_payout_accounts WHERE user_id = u.id) || " +
            "(SELECT count(*) FROM photographer_coupons WHERE photographer_id = u.id AND active) || " +
            "(SELECT count(*) FROM photos WHERE photographer_id = u.id AND deleted_at IS NULL) || " +
            "(SELECT count(*) FROM refresh_tokens WHERE user_id = u.id) " +
            "FROM users u JOIN photographer_settings s ON s.user_id = u.id WHERE u.id = ?",
        String::class.java, p,
    )!!

    @Test
    fun `confirmed run anonymises and suspends the photographer and keeps every money row`() {
        val s = seedStudio()
        val p = s.photographer
        val bystander = seedStudio().photographer
        val bystanderBefore = fingerprint(bystander)

        val result = anonymise(p, confirm = "yes")

        assertEquals(0, result.exitCode, result.stderr)
        val row = jdbc.queryForMap("SELECT * FROM users WHERE id = ?", p)
        assertEquals("Deleted photographer", row["name"])
        assertEquals("deleted-$p@deleted.invalid", row["email"])
        assertTrue((row["password_hash"] as String).startsWith("\$2"), "password must be an unusable bcrypt hash")
        assertEquals(null, row["google_sub"])
        assertEquals(null, row["avatar_url"])
        assertEquals(null, row["avatar_s3_key"])
        assertNotEquals(null, row["suspended_at"])
        assertEquals("account deleted on request", row["suspension_reason"])
        for ((table, column) in listOf(
            "refresh_tokens" to "user_id", "desktop_auth_codes" to "user_id", "social_links" to "user_id",
            "photographer_payout_accounts" to "user_id", "payout_account_audit" to "user_id",
            "photographer_messages" to "photographer_id", "photo_upload_intents" to "photographer_id",
        )) {
            assertEquals(0, count("SELECT count(*) FROM $table WHERE $column = ?", p), table)
        }
        val settings = jdbc.queryForMap("SELECT * FROM photographer_settings WHERE user_id = ?", p)
        assertEquals(deletedHandle(p), settings["handle"])
        assertEquals("", settings["bio"])
        listOf("brand_name", "city", "cover_s3_key", "watermark_s3_key", "watermark_label").forEach { assertEquals(null, settings[it], it) }
        // Money and sold photos stay: the ledger, the payout with its snapshot, the buyer's photo.
        assertEquals(1, count("SELECT count(*) FROM transactions WHERE photographer_id = ?", p))
        assertEquals(1, count("SELECT count(*) FROM payout_cycles WHERE photographer_id = ? AND account_snapshot_qr_key = ?", p, s.snapshotQrKey))
        assertEquals(1, count("SELECT count(*) FROM photos WHERE id = ? AND photographer_id = ?", s.soldPhoto, p))
        // ...but binned, so nobody can buy it any more; the buyer keeps their download.
        assertEquals(1, count("SELECT count(*) FROM photos WHERE id = ? AND deleted_at IS NOT NULL", s.soldPhoto))
        assertEquals(1, count("SELECT count(*) FROM download_grants WHERE photo_id = ?", s.soldPhoto))
        assertEquals(0, count("SELECT photo_count FROM events WHERE id = ?", s.event))
        assertEquals(0, count("SELECT photo_count FROM event_photographer WHERE event_id = ? AND photographer_id = ?", s.event, p))
        // Coupons are switched off, not deleted: past orders keep the link to the coupon they used.
        assertEquals(0, count("SELECT count(*) FROM photographer_coupons WHERE photographer_id = ? AND active", p))
        assertEquals(1, count("SELECT count(*) FROM photographer_coupons WHERE photographer_id = ?", p))
        // The public profile is gone under the old handle and the new one.
        assertThrows<NotFoundException> { publicPhotographers.getProfile(s.handle) }
        assertThrows<NotFoundException> { publicPhotographers.getProfile(deletedHandle(p)) }
        assertEquals(bystanderBefore, fingerprint(bystander))
    }

    @Test
    fun `photographer with unsold photos still in the app is refused`() {
        val s = seedStudio()
        val event = jdbc.queryForObject("SELECT event_id FROM photos WHERE id = ?", UUID::class.java, s.soldPhoto)
        // Trashed but not deleted forever: its files and faces are still stored.
        jdbc.update("INSERT INTO photos (event_id, photographer_id, s3_key, price_php, deleted_at) VALUES (?, ?, 'del/x', 125, now())", event, s.photographer)
        assertRefusedUnchanged(s.photographer)
    }

    @Test
    fun `photographer with a photo in an unpaid order is refused`() {
        // A payment still in flight would add a sale to the closed account.
        val s = seedStudio()
        val order = id(
            "INSERT INTO orders (event_id, recipient_email, payment_method, status, total_php) VALUES (?, 'late@test.local', 'qrph', 'PENDING', 125)",
            s.event,
        )
        jdbc.update("INSERT INTO order_items (order_id, photo_id, price_php_at_purchase) VALUES (?, ?, 125)", order, s.soldPhoto)
        assertRefusedUnchanged(s.photographer)
    }

    @Test
    fun `photographer with an open payout is refused`() {
        val s = seedStudio()
        jdbc.update(
            "INSERT INTO payout_cycles (id, photographer_id, week_of, amount_php, status, method) VALUES (?, ?, current_date, 0, 'pending', 'gcash')",
            "del-${UUID.randomUUID()}", s.photographer,
        )
        assertRefusedUnchanged(s.photographer)
    }

    @Test
    fun `photographer who is still owed money is refused`() {
        // Any unpaid money counts, even inside the clearing window: D20 says pay everything out first.
        val s = seedStudio()
        jdbc.update("UPDATE payout_cycles SET amount_php = 50 WHERE photographer_id = ?", s.photographer)
        assertRefusedUnchanged(s.photographer)
    }

    // PE-2.1 (R2-05): the admin closes the relevance flag (resolve or dismiss) before the account goes.
    @Test
    fun `an open relevance flag refuses the deletion`() {
        val s = seedStudio()
        val flag = id("INSERT INTO flags (target_kind, target_id, reason, status) VALUES ('user', ?, 'batch_relevance_low', 'escalated')", s.photographer)
        assertRefusedUnchanged(s.photographer)
        assertTrue("close the relevance flag first" in anonymise(s.photographer, confirm = "yes").stderr)
        jdbc.update("UPDATE flags SET status = 'dismissed', resolved_at = now() WHERE id = ?", flag)
        assertEquals(0, anonymise(s.photographer, confirm = "yes").exitCode)
    }

    @Test
    fun `runners admins and unknown ids are refused by the photographer script`() {
        for (target in listOf(user("RUNNER"), user("ADMIN"), UUID.randomUUID())) {
            val result = anonymise(target, confirm = "yes")
            assertNotEquals(0, result.exitCode, "expected a refusal for $target:\n${result.stdout}")
            assertTrue("Refused" in result.stderr, result.stderr)
        }
    }

    // ─── SC-1: seller details are legal records, kept by both scripts ───

    private fun sellerDetails(userId: UUID) = jdbc.update(
        "INSERT INTO photographer_seller_details (user_id, legal_name, id_type, id_front_key, selfie_key, address, phone, phone_key, status, revision, submitted_at) " +
            "VALUES (?, 'Del Real Name', 'PASSPORT', ?, ?, 'Cebu', '09171234567', '9171234567', 'MANUAL_REVIEW', 1, now())",
        userId, "seller-details/$userId/${UUID.randomUUID()}.jpg", "seller-details/$userId/${UUID.randomUUID()}.jpg",
    )

    @Test
    fun `a runner with a seller-details row is refused`() {
        val former = user("RUNNER")
        sellerDetails(former)

        val result = run(former, confirm = "yes")

        assertNotEquals(0, result.exitCode, result.stdout)
        assertTrue("Refused" in result.stderr && "seller details" in result.stderr, result.stderr)
        assertEquals(1, count("SELECT count(*) FROM users WHERE id = ?", former))
    }

    @Test
    fun `the photographer dry run lists the record and the prefix under Kept, and the confirmed run keeps the row`() {
        val s = seedStudio()
        sellerDetails(s.photographer)

        val dry = anonymise(s.photographer)

        assertEquals(0, dry.exitCode, dry.stderr)
        val keptRows = dry.stdout.substringAfter("== Kept (rows)").substringBefore("== R2 files to delete")
        assertTrue("photographer_seller_details" in keptRows, dry.stdout)
        val toDelete = dry.stdout.substringAfter("R2 files to delete").substringBefore("Kept in R2")
        assertTrue("seller-details/" !in toDelete, dry.stdout)
        assertTrue("seller-details/${s.photographer}/" in dry.stdout.substringAfter("Kept in R2"), dry.stdout)

        val result = anonymise(s.photographer, confirm = "yes")

        assertEquals(0, result.exitCode, result.stderr)
        assertEquals(1, count("SELECT count(*) FROM photographer_seller_details WHERE user_id = ? AND legal_name = 'Del Real Name'", s.photographer))
    }

    @Test
    fun `the anonymise run deletes the liveness sessions of the account`() {
        val s = seedStudio()
        sellerDetails(s.photographer)
        val other = user("PHOTOGRAPHER")
        fun session(owner: UUID) = jdbc.update(
            "INSERT INTO liveness_sessions (session_id, user_id, purpose, consent_version, created_at) " +
                "VALUES (?, ?, 'PHOTOGRAPHER_IDENTITY', '1.2', now())",
            "s-${UUID.randomUUID()}", owner,
        )
        session(s.photographer)
        session(s.photographer)
        session(other)

        val dry = anonymise(s.photographer)
        assertEquals(0, dry.exitCode, dry.stderr)
        assertEquals(2, count("SELECT count(*) FROM liveness_sessions WHERE user_id = ?", s.photographer), "a dry run changes nothing")

        val result = anonymise(s.photographer, confirm = "yes")

        assertEquals(0, result.exitCode, result.stderr)
        assertEquals(0, count("SELECT count(*) FROM liveness_sessions WHERE user_id = ?", s.photographer))
        assertEquals(1, count("SELECT count(*) FROM liveness_sessions WHERE user_id = ?", other))
    }

    // PE-1.2 (P2): the relevance rows go, and the sweep's due query skips the suspended account so the
    // kept event_photographer row does not make them again. Uploads sit in 2001 (shared container).
    @Test
    fun `anonymising a photographer removes their relevance rows and they are not made again`() {
        val s = seedStudio()
        val upload = OffsetDateTime.parse("2001-05-06T07:08:09.123456Z")
        jdbc.update("UPDATE event_photographer SET last_upload_at = ? WHERE event_id = ? AND photographer_id = ?", upload, s.event, s.photographer)
        fun isDue() = relevance.findDueBatches(upload, upload, LocalDate.of(1990, 1, 1), LocalDate.of(1990, 1, 1), 10_000, false)
            .any { it.eventId == s.event && it.photographerId == s.photographer }
        assertTrue(isDue())
        assertEquals(1, tx.execute { relevance.ensureBatch(s.event, s.photographer) })
        jdbc.update(
            "INSERT INTO batch_relevance_photos (photo_id, batch_id, looked_up_at) " +
                "SELECT ?, id, now() FROM batch_relevance WHERE event_id = ? AND photographer_id = ?",
            s.soldPhoto, s.event, s.photographer,
        )

        val result = anonymise(s.photographer, confirm = "yes")

        assertEquals(0, result.exitCode, result.stderr)
        assertEquals(0, count("SELECT count(*) FROM batch_relevance WHERE photographer_id = ?", s.photographer))
        assertEquals(0, count("SELECT count(*) FROM batch_relevance_photos WHERE photo_id = ?", s.soldPhoto))
        assertFalse(isDue())
        assertEquals(0, tx.execute { relevance.ensureBatch(s.event, s.photographer) })
    }

    private fun assertRefusedUnchanged(photographer: UUID) {
        val before = email(photographer)
        val result = anonymise(photographer, confirm = "yes")
        assertNotEquals(0, result.exitCode, result.stdout)
        assertTrue("Refused" in result.stderr, result.stderr)
        assertEquals(before, email(photographer))
        assertEquals(1, count("SELECT count(*) FROM social_links WHERE user_id = ?", photographer))
    }

    private companion object {
        const val RUNNER_SCRIPT = "delete-runner-account.sql"
        const val PHOTOGRAPHER_SCRIPT = "anonymise-photographer-account.sql"
    }
}
