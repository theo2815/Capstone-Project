package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.ResponseEnvelopeAdvice
import com.quickpitik.controller.AdminUsersController
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.repository.*
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminUserService
import com.quickpitik.service.photographer.PhotographerSettingsService
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataAccessException
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import java.sql.SQLException
import java.util.UUID
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.sql.DataSource
import kotlin.test.*

class AdminUserActionConcurrencyIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var settings: PhotographerSettingsRepository
    @Autowired lateinit var socials: SocialLinkRepository
    @Autowired lateinit var payouts: PayoutAccountRepository
    @Autowired lateinit var admin: AdminUserService
    @Autowired lateinit var photographer: PhotographerSettingsService
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var dataSource: DataSource
    @Autowired lateinit var txManager: PlatformTransactionManager
    @Autowired lateinit var mapper: ObjectMapper
    @Autowired lateinit var refreshTokens: RefreshTokenRepository

    private fun user(role: Role) = users.saveAndFlush(User(
        email = "decision-${UUID.randomUUID()}@test.local", passwordHash = "test-only-unused",
        name = "Decision fixture", role = role, avatarUrl = "https://example.test/avatar.jpg",
    ))

    private fun pending(): UUID {
        val id = user(Role.PHOTOGRAPHER).id
        settings.saveAndFlush(PhotographerSettings(
            userId = id, handle = "p-${id.toString().take(28)}", brandName = "Decision fixture",
            regionCode = "07", provinceCode = "0722", coverS3Key = "test/cover.jpg",
            watermarkS3Key = "test/watermark.png", verificationStatus = VerificationStatus.PENDING,
        ))
        socials.saveAndFlush(SocialLink(userId = id, platform = SocialPlatform.INSTAGRAM, url = "https://example.test/profile"))
        payouts.saveAndFlush(PayoutAccount(userId = id, method = PayoutMethod.GCASH,
            accountName = "Decision fixture", accountNumber = "09171234567", qrS3Key = "test/qr.png"))
        return id
    }

    private fun effects(id: UUID, expected: Int) {
        assertEquals(expected, jdbc.queryForObject("SELECT count(*) FROM admin_decision_log WHERE target_user_id = ?", Int::class.java, id))
        assertEquals(expected, jdbc.queryForObject("SELECT count(*) FROM photographer_messages WHERE photographer_id = ?", Int::class.java, id))
        assertEquals(expected, jdbc.queryForObject("SELECT count(*) FROM photographer_messages m JOIN admin_decision_log d ON d.id = m.source_decision_id WHERE m.photographer_id = ? AND d.target_user_id = ?", Int::class.java, id, id))
    }

    private fun revision(id: UUID): Long = jdbc.queryForObject(
        "SELECT verification_revision FROM photographer_settings WHERE user_id = ?",
        Long::class.java, id,
    )!!

    private fun request(actor: UUID, id: UUID, action: String, body: String? = null): org.springframework.mock.web.MockHttpServletResponse {
        val principal = AuthPrincipal(actor, "admin@test.local", Role.ADMIN)
        SecurityContextHolder.getContext().authentication =
            UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
        try {
            val mvc = MockMvcBuilders.standaloneSetup(AdminUsersController(admin))
                .setControllerAdvice(GlobalExceptionHandler(), ResponseEnvelopeAdvice())
                .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver()).build()
            val req = post("/api/v1/admin/users/$id/$action").contentType(MediaType.APPLICATION_JSON)
            if (body != null) req.content(body)
            return mvc.perform(req).andReturn().response
        } finally { SecurityContextHolder.clearContext() }
    }

    private fun conflict(result: Result<*>) {
        val ex = assertIs<ApiException>(result.exceptionOrNull())
        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
    }

    private fun awaitBlocked(count: Int) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
        while (jdbc.queryForObject("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'", Int::class.java)!! < count) {
            check(System.nanoTime() < deadline) { "Expected $count blocked PostgreSQL operations" }
            Thread.sleep(20)
        }
    }

    // Keep both writes in flight: the winner waits on the table and its competitor on the user.
    private fun race(first: () -> Unit, second: () -> Unit, table: String = "photographer_settings"): List<Result<Unit>> {
        require(table in setOf("photographer_settings", "users"))
        val pool = Executors.newFixedThreadPool(2)
        val barrier = CyclicBarrier(2)
        try {
            dataSource.connection.use { lock ->
                lock.autoCommit = false
                lock.createStatement().use { it.execute("LOCK TABLE $table IN SHARE MODE") }
                val futures = listOf(first, second).map { action -> pool.submit<Result<Unit>> {
                    runCatching {
                        barrier.await(10, TimeUnit.SECONDS)
                        TransactionTemplate(txManager).executeWithoutResult {
                            jdbc.execute("SET LOCAL lock_timeout = '15s'")
                            action()
                        }
                    }
                } }
                try { awaitBlocked(2) } finally { lock.commit() }
                return futures.map { it.get(20, TimeUnit.SECONDS) }
            }
        } finally { pool.shutdownNow() }
    }

    @Test
    fun `two concurrent approvals accept once with one audit and inbox row`() {
        val id = pending()
        val a = user(Role.ADMIN).id
        val b = user(Role.ADMIN).id
        val results = race({ admin.approve(a, id) }, { admin.approve(b, id) })
        assertEquals(1, results.count { it.isSuccess })
        conflict(results.single { it.isFailure })
        assertEquals(VerificationStatus.APPROVED, settings.findById(id).get().verificationStatus)
        effects(id, 1)
    }

    @Test
    fun `approval competing with rejection accepts one decision and notification`() {
        val id = pending()
        val a = user(Role.ADMIN).id
        val b = user(Role.ADMIN).id
        val results = race({ admin.approve(a, id) }, { admin.reject(b, id, "Correct the profile") })
        assertEquals(1, results.count { it.isSuccess })
        conflict(results.single { it.isFailure })
        val state = settings.findById(id).get().verificationStatus
        assertEquals(if (results[0].isSuccess) VerificationStatus.APPROVED else VerificationStatus.INCOMPLETE, state)
        effects(id, 1)
    }

    @ParameterizedTest
    @ValueSource(strings = ["approve", "reject", "reset-verification"])
    fun `a stale supplied revision refuses every decision before side effects`(action: String) {
        val id = pending()
        val response = request(user(Role.ADMIN).id, id, action,
            """{"expectedRevision":999,"reason":"Correct the evidence"}""")
        assertEquals(409, response.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, mapper.readTree(response.contentAsString)["errors"][0]["code"].asText())
        assertEquals(VerificationStatus.PENDING, settings.findById(id).get().verificationStatus)
        effects(id, 0)
    }

    @ParameterizedTest
    @ValueSource(strings = ["approve", "reject", "reset-verification"])
    fun `an accepted decision returns the next revision and legacy input still works`(action: String) {
        val id = pending()
        val body = if (action == "approve") null else """{"reason":"Correct the evidence"}"""
        val response = request(user(Role.ADMIN).id, id, action, body)
        assertEquals(200, response.status)
        assertEquals(1L, mapper.readTree(response.contentAsString)["data"].path("verificationRevision").asLong(-1))
        assertEquals(1L, revision(id))
        effects(id, 1)
    }

    @ParameterizedTest
    @ValueSource(strings = ["approve", "reject", "reset-verification"])
    fun `negative revision inputs are rejected without effects`(action: String) {
        val id = pending()
        assertEquals(400, request(user(Role.ADMIN).id, id, action,
            """{"expectedRevision":-1,"reason":"Correct the evidence"}""").status)
        effects(id, 0)
    }

    @Test
    fun `admin row and detail expose the current review generation`() {
        val id = pending()
        assertEquals(0L, mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(admin.detail(id)).path("verificationRevision").asLong(-1))
        val row = admin.list("PHOTOGRAPHER", null, users.findById(id).get().email,
            com.quickpitik.common.PaginationParams.of(0, 10)).items.single()
        assertEquals(0L, mapper.valueToTree<com.fasterxml.jackson.databind.JsonNode>(row).path("verificationRevision").asLong(-1))
    }

    @Test
    fun `withdraw resubmit rejects the delayed old approval and accepts the current generation`() {
        val id = pending()
        val actor = user(Role.ADMIN).id
        photographer.withdrawVerification(id)
        photographer.submitVerification(id)
        val stale = request(actor, id, "approve", """{"expectedRevision":0}""")
        assertEquals(409, stale.status)
        assertEquals(VerificationStatus.PENDING, settings.findById(id).get().verificationStatus)
        assertEquals(2L, revision(id))
        effects(id, 0)
        assertEquals(200, request(actor, id, "approve", """{"expectedRevision":2}""").status)
        assertEquals(3L, revision(id))
        effects(id, 1)
        photographer.submitVerification(id)
        assertEquals(VerificationStatus.PENDING, settings.findById(id).get().verificationStatus)
        assertEquals(4L, revision(id))
        effects(id, 1)
        assertEquals(200, request(actor, id, "approve", """{"expectedRevision":4}""").status)
        assertEquals(5L, revision(id))
        effects(id, 2)
    }

    @Test
    fun `withdrawal competing with approval leaves the profile incomplete with exact effects`() {
        val id = pending()
        val actor = user(Role.ADMIN).id
        val results = race({ admin.approve(actor, id) }, { photographer.withdrawVerification(id) })
        assertTrue(results[1].isSuccess)
        if (results[0].isFailure) conflict(results[0])
        assertEquals(VerificationStatus.INCOMPLETE, settings.findById(id).get().verificationStatus)
        effects(id, if (results[0].isSuccess) 1 else 0)
        assertEquals(if (results[0].isSuccess) 2L else 1L, revision(id))
    }

    @ParameterizedTest
    @ValueSource(strings = ["submit", "withdraw", "initialize"])
    fun `photographer writers wait on the parent even before settings exist`(action: String) {
        val id = if (action == "initialize") user(Role.PHOTOGRAPHER).id else pending()
        val pool = Executors.newSingleThreadExecutor()
        try {
            dataSource.connection.use { lock ->
                lock.autoCommit = false
                lock.prepareStatement("SELECT id FROM users WHERE id = ? FOR NO KEY UPDATE").use {
                    it.setObject(1, id)
                    it.executeQuery().close()
                }
                val future = pool.submit<Result<Unit>> { runCatching {
                    when (action) {
                        "submit" -> photographer.submitVerification(id)
                        "withdraw" -> photographer.withdrawVerification(id)
                        else -> photographer.getOrCreate(id)
                    }
                    Unit
                } }
                try { awaitBlocked(1) } finally { lock.commit() }
                future.get(20, TimeUnit.SECONDS).getOrThrow()
            }
        } finally { pool.shutdownNow() }
        assertTrue(settings.existsById(id))
    }

    @Test
    fun `missing settings initialize once without reclassifying the photographer`() {
        val id = user(Role.PHOTOGRAPHER).id
        val results = race({ photographer.getOrCreate(id) }, { photographer.withdrawVerification(id) })
        assertEquals(2, results.count { it.isSuccess })
        assertEquals(VerificationStatus.INCOMPLETE, settings.findById(id).get().verificationStatus)
        assertEquals(0L, revision(id))
        effects(id, 0)
    }

    @Test
    fun `migration starts settings at revision zero`() {
        val id = user(Role.PHOTOGRAPHER).id
        jdbc.update("INSERT INTO photographer_settings(user_id) VALUES (?)", id)
        assertEquals(0L, revision(id))
    }

    @Test
    fun `the database enforces a nonnegative revision`() {
        val id = pending()
        val ex = assertFailsWith<DataAccessException> {
            jdbc.update("UPDATE photographer_settings SET verification_revision = -1 WHERE user_id = ?", id)
        }
        val sql = generateSequence<Throwable>(ex) { it.cause }.filterIsInstance<SQLException>().first()
        assertEquals("23514", sql.sqlState)
    }

    @Test
    fun `rollback discards the decision and inbox with the state change`() {
        val id = pending()
        val actor = user(Role.ADMIN).id
        assertFailsWith<IllegalStateException> {
            TransactionTemplate(txManager).executeWithoutResult {
                admin.approve(actor, id)
                settings.flush()
                throw IllegalStateException("Roll back the accepted command")
            }
        }
        assertEquals(VerificationStatus.PENDING, settings.findById(id).get().verificationStatus)
        effects(id, 0)
    }

    @ParameterizedTest
    @ValueSource(strings = ["reject", "reset-verification", "suspend"])
    fun `invalid reasons retain HTTP validation without audit or inbox rows`(action: String) {
        val id = pending()
        val actor = user(Role.ADMIN).id
        for (reason in listOf(" ", "x".repeat(501))) {
            assertEquals(400, request(actor, id, action, """{"reason":"$reason"}""").status)
        }
        effects(id, 0)
    }

    @ParameterizedTest
    @ValueSource(strings = ["suspend", "unsuspend"])
    fun `concurrent account transitions accept once without duplicate effects`(action: String) {
        val id = pending()
        val revokedAt = java.time.OffsetDateTime.parse("2026-01-01T00:00:00Z")
        if (action == "unsuspend") {
            users.saveAndFlush(users.findById(id).get().apply {
                suspendedAt = revokedAt
                suspensionReason = "Original suspension"
            })
        }
        val token = refreshTokens.saveAndFlush(RefreshToken(userId = id, tokenHash = UUID.randomUUID().toString(),
            expiresAt = java.time.OffsetDateTime.now().plusDays(1),
            revokedAt = if (action == "unsuspend") revokedAt else null))
        val a = user(Role.ADMIN).id
        val b = user(Role.ADMIN).id
        fun act(actor: UUID) {
            if (action == "suspend") admin.suspend(actor, id, "Suspended by $actor") else admin.unsuspend(actor, id)
        }
        val results = race({ act(a) }, { act(b) }, table = "users")
        assertEquals(1, results.count { it.isSuccess })
        conflict(results.single { it.isFailure })
        effects(id, 1)
        val persistedToken = refreshTokens.findById(token.id).get()
        val persistedUser = users.findById(id).get()
        if (action == "suspend") {
            assertNotNull(persistedToken.revokedAt)
            assertNotNull(persistedUser.suspendedAt)
            assertEquals("Suspended by ${if (results[0].isSuccess) a else b}", persistedUser.suspensionReason)
        } else {
            assertEquals(revokedAt, persistedToken.revokedAt)
            assertNull(persistedUser.suspendedAt)
            assertNull(persistedUser.suspensionReason)
        }
    }

    @Test
    fun `missing settings cannot be approved and failed initialization is rolled back`() {
        val id = user(Role.PHOTOGRAPHER).id
        assertEquals(422, request(user(Role.ADMIN).id, id, "approve").status)
        assertFalse(settings.existsById(id))
        effects(id, 0)
    }

    @Test
    fun `suspension rollback keeps refresh tokens active and writes no effects`() {
        val id = pending()
        val actor = user(Role.ADMIN).id
        val token = refreshTokens.saveAndFlush(RefreshToken(userId = id, tokenHash = UUID.randomUUID().toString(),
            expiresAt = java.time.OffsetDateTime.now().plusDays(1)))
        assertFailsWith<IllegalStateException> {
            TransactionTemplate(txManager).executeWithoutResult {
                admin.suspend(actor, id, "Rollback")
                throw IllegalStateException("Roll back suspension")
            }
        }
        assertNull(users.findById(id).get().suspendedAt)
        assertNull(refreshTokens.findById(token.id).get().revokedAt)
        effects(id, 0)
    }
}
