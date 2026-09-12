package com.quickpitik.support

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.CorsConfig
import com.quickpitik.dto.admin.MarkPayoutPaidRequest
import com.quickpitik.dto.orders.*
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.admin.*
import com.quickpitik.service.disbursement.PayoutDisbursementWebhookService
import com.quickpitik.service.earnings.PayoutRequestService
import com.quickpitik.support.PostgresIntegrationTest
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.*

class PayoutHardeningIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var cycles: PayoutCycleRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var admin: AdminPayoutService
    @Autowired lateinit var requests: PayoutRequestService
    @Autowired lateinit var txManager: PlatformTransactionManager
    @Autowired lateinit var mfa: AdminMfaService
    @Autowired lateinit var mfaRows: AdminMfaRepository
    @Autowired lateinit var cipher: MfaSecretCipher
    @Autowired lateinit var totp: TotpService
    @Autowired lateinit var webhooks: PayoutDisbursementWebhookService
    @Autowired lateinit var accounts: PayoutAccountRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var transactions: TransactionRepository
    @Autowired lateinit var recoveryCodes: AdminMfaRecoveryCodeRepository
    @Autowired lateinit var dataSource: javax.sql.DataSource
    @Autowired lateinit var jdbc: org.springframework.jdbc.core.JdbcTemplate
    @Autowired lateinit var auth: com.quickpitik.service.auth.AuthService
    @Autowired lateinit var refreshTokens: com.quickpitik.service.auth.RefreshTokenService
    @Autowired lateinit var passwords: org.springframework.security.crypto.password.PasswordEncoder
    @Autowired lateinit var payoutAccounts: com.quickpitik.service.photographer.PayoutAccountService
    @Autowired lateinit var storage: com.quickpitik.service.storage.StorageService

    private fun user(role: Role) = users.saveAndFlush(User(
        email = "review-${UUID.randomUUID()}@test.local", passwordHash = "test-only-unused",
        name = "Review fixture", role = role,
    ))

    private fun cycle(status: PayoutCycleStatus) = cycles.saveAndFlush(PayoutCycle(
        id = "PAY-REVIEW-${UUID.randomUUID()}", photographerId = user(Role.PHOTOGRAPHER).id,
        weekOf = LocalDate.now(), amountPhp = BigDecimal("1000"), statusWire = status.wire,
    ))

    @Test fun `CORS accepts the header required by step-up`() {
        val source = CorsConfig("https://quickpitik.com").corsConfigurationSource()
        val request = org.springframework.mock.web.MockHttpServletRequest("OPTIONS", "/api/v1/me/photographer/payouts")
        val config = requireNotNull(source.getCorsConfiguration(request))
        assertTrue(config.checkHeaders(listOf("content-type", "authorization", "x-step-up"))!!.contains("x-step-up"))
    }

    @Test fun `documented deployment flags bind through application yaml`() {
        val environment = org.springframework.mock.env.MockEnvironment()
        environment.propertySources.addFirst(org.springframework.core.env.SystemEnvironmentPropertySource(
            "review-env", mapOf("PAYOUT_STEP_UP_ENABLED" to "true", "PAYOUT_DISBURSEMENT_PROVIDER" to "paymongo"),
        ))
        org.springframework.boot.env.YamlPropertySourceLoader()
            .load("application", org.springframework.core.io.ClassPathResource("application.yml"))
            .forEach { environment.propertySources.addLast(it) }
        val properties = org.springframework.boot.context.properties.bind.Binder.get(environment)
            .bindOrCreate("app.platform", com.quickpitik.config.PlatformProperties::class.java)
        assertTrue(properties.payoutStepUpEnabled)
        assertEquals("paymongo", properties.disbursementProvider)
        environment.setProperty("app.platform.payout-step-up-enabled", "true")
        assertTrue(org.springframework.boot.context.properties.bind.Binder.get(environment)
            .bindOrCreate("app.platform", com.quickpitik.config.PlatformProperties::class.java).payoutStepUpEnabled)
    }

    @Test fun `migration preserves legacy approvals as a payable explicit state`() {
        val c = cycle(PayoutCycleStatus.PENDING)
        c.settledAt = OffsetDateTime.now().minusDays(1)
        cycles.saveAndFlush(c)
        val migration = org.springframework.core.io.ClassPathResource("db/migration/V61__payout_review_integrity.sql")
            .inputStream.bufferedReader().use { it.readText() }.substringBefore("ALTER TABLE")
        dataSource.connection.use { connection ->
            org.springframework.jdbc.datasource.init.ScriptUtils.executeSqlScript(connection,
                org.springframework.core.io.ByteArrayResource(migration.toByteArray()))
        }
        val row = admin.list(null, c.id.lowercase(), PaginationParams.of(null, null)).items.single()
        assertEquals("approved", row.status)
        assertEquals("paid", admin.markPaid(user(Role.ADMIN).id, c.id, MarkPayoutPaidRequest("TEST")).status)
    }

    @Test fun `approved and processing payouts stay in the earnings pending summary`() {
        val c = cycle(PayoutCycleStatus.APPROVED)
        assertEquals(0, cycles.sumOpenAmount(c.photographerId).compareTo(c.amountPhp))
        c.status = PayoutCycleStatus.PROCESSING
        cycles.saveAndFlush(c)
        assertEquals(0, cycles.sumOpenAmount(c.photographerId).compareTo(c.amountPhp))
        assertEquals(c.id, cycles.findNextUpcoming(c.photographerId, org.springframework.data.domain.PageRequest.of(0, 1)).single().id)
        assertEquals(0, cycles.sumReservedByPhotographer(c.photographerId).compareTo(c.amountPhp))
    }

    @Test fun `a failure waits for a competing payment then rejects the paid state`() {
        val c = cycle(PayoutCycleStatus.PROCESSING)
        val adminId = user(Role.ADMIN).id
        val pool = Executors.newSingleThreadExecutor()
        try {
            dataSource.connection.use { lock ->
                lock.autoCommit = false
                lock.prepareStatement("SELECT id FROM payout_cycles WHERE id = ? FOR UPDATE").use {
                    it.setString(1, c.id)
                    it.executeQuery().close()
                }
                val failure = pool.submit<Result<String>> { runCatching { admin.markFailed(adminId, c.id, "Failed").status } }
                awaitBlockedQueries(1)
                lock.prepareStatement("UPDATE payout_cycles SET status = 'paid', settled_at = now() WHERE id = ?").use {
                    it.setString(1, c.id)
                    it.executeUpdate()
                }
                lock.commit()
                assertIs<ApiException>(failure.get(20, TimeUnit.SECONDS).exceptionOrNull())
            }
            assertEquals(PayoutCycleStatus.PAID, cycles.findById(c.id).get().status)
            assertEquals(0, cycles.sumPaidByPhotographer(c.photographerId).compareTo(c.amountPhp))
        } finally {
            pool.shutdownNow()
        }
    }

    @Test fun `a persisted TOTP can only be consumed once`() {
        val a = user(Role.ADMIN)
        val secret = totp.randomSecret()
        mfaRows.saveAndFlush(AdminMfa(userId = a.id, secretEnc = cipher.encrypt(secret), enabled = true))
        val code = totp.generate(secret, System.currentTimeMillis() / 1000)
        assertTrue(mfa.verifyCode(a.id, code))
        assertFalse(mfa.verifyCode(a.id, code))
    }

    private fun awaitBlockedQueries(count: Int) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
        while (jdbc.queryForObject("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() " +
                "AND wait_event_type = 'Lock'", Int::class.java)!! < count) {
            check(System.nanoTime() < deadline) { "Expected $count blocked database operations" }
            Thread.sleep(20)
        }
    }

    // One writer waits on the child table; its competitor must wait on the user
    // lock instead of reading the same unused recovery code or available balance.
    private fun <T> concurrentWrites(table: String, action: () -> T): List<Result<T>> {
        require(table in setOf("payout_cycles", "admin_mfa_recovery_codes"))
        val pool = Executors.newFixedThreadPool(2)
        try {
            dataSource.connection.use { lock ->
                lock.autoCommit = false
                lock.createStatement().use { it.execute("LOCK TABLE $table IN SHARE MODE") }
                val results = List(2) { pool.submit<Result<T>> { runCatching(action) } }
                try {
                    awaitBlockedQueries(2)
                } finally {
                    lock.commit()
                }
                return results.map { it.get(20, TimeUnit.SECONDS) }
            }
        } finally {
            pool.shutdownNow()
        }
    }

    @Test fun `concurrent requests reserve once and the loser gets a clean conflict`() {
        val p = user(Role.PHOTOGRAPHER)
        accounts.saveAndFlush(PayoutAccount(userId = p.id, method = PayoutMethod.GCASH,
            accountNumber = "09171234567", accountName = "Review", isPrimary = true, qrS3Key = "review/qr.png"))
        val e = events.saveAndFlush(Event(slug = "review-${UUID.randomUUID()}", name = "Review",
            date = LocalDate.now(), location = "Cebu", status = EventStatus.ACTIVE))
        val photo = photos.saveAndFlush(Photo(eventId = e.id, photographerId = p.id,
            s3Key = "review/${UUID.randomUUID()}", pricePhp = BigDecimal("1000")))
        val order = orders.saveAndFlush(Order(eventId = e.id, recipientEmail = "review@test.local",
            paymentMethodWire = "gcash", status = OrderStatus.PAID, totalPhp = BigDecimal("1000")))
        transactions.saveAndFlush(Transaction(paidAt = OffsetDateTime.now().minusDays(10),
            photographerId = p.id, eventId = e.id, photoId = photo.id, orderId = order.id,
            amountKeptPhp = BigDecimal("1000")))
        val results = concurrentWrites("payout_cycles") { requests.request(p.id) }
        assertEquals(1, results.count { it.isSuccess })
        val conflict = assertIs<ApiException>(results.single { it.isFailure }.exceptionOrNull())
        assertEquals(org.springframework.http.HttpStatus.CONFLICT, conflict.status)
        assertEquals(1L, cycles.countByPhotographerId(p.id))
    }

    @Test fun `simultaneous recovery attempts consume the code only once`() {
        val a = user(Role.ADMIN)
        mfaRows.saveAndFlush(AdminMfa(userId = a.id, secretEnc = cipher.encrypt(totp.randomSecret()), enabled = true))
        val code = "1122334455667788"
        val hash = java.security.MessageDigest.getInstance("SHA-256").digest(code.toByteArray())
            .joinToString("") { "%02x".format(it) }
        recoveryCodes.saveAndFlush(AdminMfaRecoveryCode(userId = a.id, codeHash = hash))
        val results = concurrentWrites("admin_mfa_recovery_codes") { mfa.verifyCode(a.id, code) }
        assertEquals(listOf(false, true), results.map { it.getOrThrow() }.sorted())
        assertFalse(mfa.verifyCode(a.id, code))
    }

    @Test fun `frozen destination never uses the replacement live QR`() {
        val c = cycle(PayoutCycleStatus.APPROVED)
        c.accountSnapshotMethod = "gcash"
        c.accountSnapshotNumber = "09170000001"
        c.accountSnapshotName = "Original recipient"
        c.accountSnapshotQrKey = "review/original-recipient-qr.png"
        cycles.saveAndFlush(c)
        accounts.saveAndFlush(PayoutAccount(userId = c.photographerId, method = PayoutMethod.GCASH,
            accountNumber = "09170000002", accountName = "Replacement recipient",
            qrS3Key = "review/replacement-recipient-qr.png", isPrimary = true))
        val account = admin.list(null, c.id.lowercase(), PaginationParams.of(null, null)).items.single().payoutAccount
        assertEquals("09170000001", account.accountNumber)
        assertContains(account.qr!!.dataUrl, "original-recipient-qr.png")
    }

    @Test fun `replacing and deleting a live account preserves the payout QR`() {
        val c = cycle(PayoutCycleStatus.APPROVED)
        val key = "test-qr/${UUID.randomUUID()}.png"
        val bytes = java.io.ByteArrayOutputStream().also {
            javax.imageio.ImageIO.write(java.awt.image.BufferedImage(2, 2, java.awt.image.BufferedImage.TYPE_INT_RGB), "png", it)
        }.toByteArray()
        storage.put(key, bytes, "image/png")
        val account = accounts.saveAndFlush(PayoutAccount(userId = c.photographerId, method = PayoutMethod.GCASH,
            accountNumber = "09171234567", accountName = "Original", isPrimary = true, qrS3Key = key))
        c.accountSnapshotMethod = "gcash"
        c.accountSnapshotNumber = account.accountNumber
        c.accountSnapshotName = account.accountName
        c.accountSnapshotQrKey = key
        cycles.saveAndFlush(c)
        var replacement: String? = null
        try {
            payoutAccounts.uploadQr(c.photographerId, account.id, bytes, "image/png")
            replacement = accounts.findById(account.id).get().qrS3Key
            assertNotEquals(key, replacement)
            assertTrue(storage.exists(key))
            TransactionTemplate(txManager).execute { tx ->
                payoutAccounts.patch(c.photographerId, account.id,
                    com.quickpitik.dto.photographer.PatchPayoutRequest(accountNumber = "09170000002"))
                tx.setRollbackOnly()
            }
            assertEquals(replacement, accounts.findById(account.id).get().qrS3Key)
            assertTrue(storage.exists(replacement!!), "A rolled-back edit must not delete the saved QR")
            payoutAccounts.patch(c.photographerId, account.id,
                com.quickpitik.dto.photographer.PatchPayoutRequest(accountNumber = "09170000002"))
            assertNull(accounts.findById(account.id).get().qrS3Key)
            assertEquals(false, requests.getBalance(c.photographerId).primaryAccount?.hasQr)
            payoutAccounts.delete(c.photographerId, account.id)
            assertTrue(storage.exists(key))
            val frozen = admin.list(null, c.id.lowercase(), PaginationParams.of(null, null)).items.single().payoutAccount
            assertEquals("09171234567", frozen.accountNumber)
            assertNotNull(frozen.qr)
        } finally {
            storage.delete(key)
            replacement?.let { storage.delete(it) }
        }
    }

    @Test fun `Google first factor returns an MFA challenge instead of session tokens`() {
        val a = user(Role.ADMIN)
        a.googleSub = "review-${UUID.randomUUID()}"
        users.saveAndFlush(a)
        mfaRows.saveAndFlush(AdminMfa(userId = a.id, secretEnc = cipher.encrypt(totp.randomSecret()), enabled = true))
        val decoder = org.mockito.Mockito.mock(org.springframework.security.oauth2.jwt.JwtDecoder::class.java)
        val token = org.springframework.security.oauth2.jwt.Jwt.withTokenValue("test-only")
            .header("alg", "RS256").subject(a.googleSub).claim("email", a.email).claim("email_verified", true).build()
        org.mockito.Mockito.`when`(decoder.decode("test-only")).thenReturn(token)
        val google = com.quickpitik.service.auth.GoogleAuthService(users, passwords, refreshTokens, auth, decoder,
            com.quickpitik.config.GoogleAuthProperties(clientId = "review"))
        val result = google.login(com.quickpitik.dto.auth.GoogleLoginRequest("test-only"))
        assertTrue(result.mfaRequired)
        assertTrue(result.accessToken.isBlank())
        assertTrue(result.refreshToken.isBlank())
        assertNotNull(result.mfaToken)
    }

    @Test fun `provider transfers reject manual overrides and retain their reservation`() {
        val c = cycle(PayoutCycleStatus.PROCESSING)
        c.provider = "paymongo"
        c.providerTransferId = "tr-review-${UUID.randomUUID()}"
        cycles.saveAndFlush(c)
        val a = user(Role.ADMIN).id
        assertFailsWith<ApiException> { admin.hold(a, c.id, "Hold") }
        assertFailsWith<ApiException> { admin.markPaid(a, c.id, MarkPayoutPaidRequest("manual")) }
        assertFailsWith<ApiException> { admin.markFailed(a, c.id, "Failed") }
        assertFailsWith<ApiException> { requests.withdraw(c.photographerId, c.id) }
        assertEquals(0, cycles.sumReservedByPhotographer(c.photographerId).compareTo(c.amountPhp))
        assertEquals("paid", webhooks.handle(event("evt-${UUID.randomUUID()}",
            "transfer.outward.successful", c.providerTransferId!!))["status"])
        assertEquals(PayoutCycleStatus.PAID, cycles.findById(c.id).get().status)
    }

    private fun event(id: String, type: String, transferId: String) = PaymongoTransferWebhookEvent(
        PaymongoTransferWebhookOuter(id, PaymongoTransferWebhookAttributes(
            type, PaymongoTransferNode(transferId, null),
        )),
    )

    @Test fun `out-of-order return is durable and supersedes later success`() {
        val c = cycle(PayoutCycleStatus.PROCESSING)
        c.provider = "paymongo"
        c.providerTransferId = "tr-review-${UUID.randomUUID()}"
        cycles.saveAndFlush(c)
        val returned = event("evt-review-${UUID.randomUUID()}", "transfer.outward.returned", c.providerTransferId!!)
        assertEquals("returned", webhooks.handle(returned)["status"])
        webhooks.handle(event("evt-review-${UUID.randomUUID()}", "transfer.outward.successful", c.providerTransferId!!))
        assertEquals(true, webhooks.handle(returned)["deduped"])
        assertEquals(PayoutCycleStatus.RETURNED, cycles.findById(c.id).get().status)
    }
}
