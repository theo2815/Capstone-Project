package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.common.ResponseEnvelopeAdvice
import com.quickpitik.controller.AdminUsersController
import com.quickpitik.entity.*
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminUserService
import com.quickpitik.service.admin.AdminActionCommandService
import com.quickpitik.service.admin.AdminActionResult
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.repository.AdminActionCommandRepository
import com.quickpitik.service.runner.RunnerMessagesService
import org.springframework.dao.DataAccessException
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import javax.sql.DataSource
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import org.junit.jupiter.params.provider.ValueSource
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.http.MediaType
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import java.util.UUID
import java.time.Clock
import java.time.Instant
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.concurrent.CountDownLatch
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.*

class AdminActionIdempotencyIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var admin: AdminUserService
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var mapper: ObjectMapper
    @Autowired lateinit var commands: AdminActionCommandService
    @Autowired lateinit var receipts: AdminActionCommandRepository
    @Autowired lateinit var log: AdminDecisionLogService
    @Autowired lateinit var inbox: RunnerMessagesService
    @Autowired lateinit var manager: PlatformTransactionManager
    @Autowired lateinit var dataSource: DataSource

    private fun user(role: Role) = users.saveAndFlush(User(email = "message-${UUID.randomUUID()}@test.local",
        passwordHash = "unused", name = "Message fixture", role = role)).id

    private fun send(actor: UUID, target: UUID, key: String? = null, subject: String = "Hello", body: String = "Message"):
        org.springframework.mock.web.MockHttpServletResponse {
        val principal = AuthPrincipal(actor, "fixture@test.local", users.findById(actor).orElseThrow().role)
        SecurityContextHolder.getContext().authentication = UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
        try {
            val mvc = MockMvcBuilders.standaloneSetup(AdminUsersController(admin))
                .setControllerAdvice(GlobalExceptionHandler(), ResponseEnvelopeAdvice())
                .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver()).build()
            val request = post("/api/v1/admin/users/$target/message").contentType(MediaType.APPLICATION_JSON)
                .content(mapper.writeValueAsBytes(mapOf("subject" to subject, "body" to body)))
            if (key != null) request.header("Idempotency-Key", key)
            return mvc.perform(request).andReturn().response
        } finally { SecurityContextHolder.clearContext() }
    }

    private fun count(table: String, column: String, id: UUID): Int =
        jdbc.queryForObject("SELECT count(*) FROM $table WHERE $column = ?", Int::class.java, id)!!

    private fun effects(target: UUID, role: Role, expected: Int) {
        assertEquals(expected, count("admin_decision_log", "target_user_id", target))
        assertEquals(if (role == Role.PHOTOGRAPHER) expected else 0, count("photographer_messages", "photographer_id", target))
        assertEquals(if (role == Role.RUNNER) expected else 0, count("runner_messages", "runner_id", target))
    }

    @ParameterizedTest @EnumSource(value = Role::class, names = ["PHOTOGRAPHER", "RUNNER"])
    fun `lost response replay returns original decision and message in the correct inbox`(role: Role) {
        val actor = user(Role.ADMIN); val target = user(role); val key = UUID.randomUUID().toString()
        val first = send(actor, target, key, " Hello ", " Message ")
        assertEquals(200, first.status)
        val second = send(actor, target, key)
        assertEquals(200, second.status)
        val original = mapper.readTree(first.contentAsString)["data"]
        val replay = mapper.readTree(second.contentAsString)["data"]
        assertEquals(original["id"], replay["id"])
        assertEquals(original["sourceDecisionId"], replay["sourceDecisionId"])
        assertEquals(original["createdAt"], replay["createdAt"])
        assertEquals("Hello", replay["title"].asText())
        effects(target, role, 1)
        val deliberate = send(actor, target, UUID.randomUUID().toString())
        assertEquals(200, deliberate.status)
        assertNotEquals(original["id"], mapper.readTree(deliberate.contentAsString)["data"]["id"])
        effects(target, role, 2)
    }

    @ParameterizedTest @ValueSource(strings = ["subject", "body", "recipient", "actor"])
    fun `same key rejects each changed identity field`(field: String) {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER); val other = user(Role.RUNNER)
        val key = UUID.randomUUID().toString()
        assertEquals(200, send(actor, target, key).status)
        val response = when (field) {
            "subject" -> send(actor, target, key, subject = "Other")
            "body" -> send(actor, target, key, body = "Other")
            "recipient" -> send(actor, other, key)
            else -> send(user(Role.ADMIN), target, key)
        }
        assertEquals(409, response.status)
        assertEquals("CONFLICT", mapper.readTree(response.contentAsString)["errors"][0]["code"].asText())
        effects(target, Role.RUNNER, 1); effects(other, Role.RUNNER, 0)
    }

    @Test fun `malformed key and invalid message do not claim receipts`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        assertEquals(400, send(actor, target, "invalid").status)
        val key = UUID.randomUUID().toString()
        assertEquals(400, send(actor, target, key, subject = " ").status)
        assertEquals(0, count("admin_action_commands", "command_id", UUID.fromString(key)))
        effects(target, Role.RUNNER, 0)
        assertEquals(200, send(actor, target, key).status)
    }

    @Test fun `authorization is checked before successful replay`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER); val key = UUID.randomUUID().toString()
        assertEquals(200, send(actor, target, key).status)
        jdbc.update("UPDATE users SET suspended_at = now() WHERE id = ?", actor)
        assertEquals(403, send(actor, target, key).status)
        assertEquals(403, send(user(Role.RUNNER), target, key).status)
        effects(target, Role.RUNNER, 1)
    }

    @Test fun `legacy unkeyed sends remain separate intentional messages`() {
        val actor = user(Role.ADMIN); val target = user(Role.PHOTOGRAPHER)
        assertEquals(200, send(actor, target).status); assertEquals(200, send(actor, target).status)
        effects(target, Role.PHOTOGRAPHER, 2)
    }

    @ParameterizedTest @EnumSource(value = Role::class, names = ["PHOTOGRAPHER", "RUNNER"])
    fun `simultaneous requests commit one message and replay its IDs`(role: Role) {
        val actor = user(Role.ADMIN); val target = user(role); val key = UUID.randomUUID().toString()
        val barrier = CyclicBarrier(2); val pool = Executors.newFixedThreadPool(2)
        try {
            val futures = (1..2).map { pool.submit<org.springframework.mock.web.MockHttpServletResponse> {
                barrier.await(10, TimeUnit.SECONDS); send(actor, target, key)
            } }
            val responses = futures.map { it.get(25, TimeUnit.SECONDS) }
            responses.forEach { assertEquals(200, it.status) }
            assertEquals(mapper.readTree(responses[0].contentAsString)["data"]["id"],
                mapper.readTree(responses[1].contentAsString)["data"]["id"])
            effects(target, role, 1)
        } finally { pool.shutdownNow() }
    }

    private fun action(actor: UUID, target: UUID): AdminActionResult {
        val decision = log.logUserDecision(actor, target, "messaged", "Hello")
        val message = inbox.pushMessage(target, RunnerMessageKind.ADMIN_MESSAGE, "Message", actor, decision.id, title = "Hello")
        return AdminActionResult(target, decision.id, message.id)
    }

    @Test fun `uniqueness loser rereads original result in a fresh transaction`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        val key = IdempotencyKey(UUID.randomUUID().toString()); val hash = "a".repeat(64)
        val transactions = CopyOnWriteArrayList<Long>(); val executions = AtomicInteger()
        val pool = Executors.newFixedThreadPool(2)
        try {
            dataSource.connection.use { blocker ->
                blocker.autoCommit = false
                blocker.createStatement().use { it.execute("LOCK TABLE admin_action_commands IN SHARE MODE") }
                val futures = (1..2).map { pool.submit<AdminActionResult> {
                    commands.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash,
                        { transactions.add(jdbc.queryForObject("SELECT txid_current()", Long::class.java)!!) },
                        { executions.incrementAndGet(); action(actor, target) })
                } }
                try {
                    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
                    while (jdbc.queryForObject("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'", Int::class.java)!! < 2) {
                        check(System.nanoTime() < deadline) { "Both claims must reach the real PostgreSQL insert" }; Thread.sleep(20)
                    }
                } finally { blocker.commit() }
                val results = futures.map { it.get(25, TimeUnit.SECONDS) }
                assertEquals(results[0], results[1]); assertEquals(1, executions.get())
                assertEquals(3, transactions.size); assertEquals(3, transactions.toSet().size)
                effects(target, Role.RUNNER, 1)
            }
        } finally { pool.shutdownNow() }
    }

    @Test fun `failed message transaction removes receipt decision and inbox before same-key retry`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        val key = IdempotencyKey(UUID.randomUUID().toString()); val hash = "a".repeat(64)
        assertFailsWith<IllegalStateException> {
            commands.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash, {}, {
                action(actor, target); error("simulated write failure")
            })
        }
        TransactionTemplate(manager).executeWithoutResult {
            assertEquals(0, count("admin_action_commands", "command_id", UUID.fromString(key.value)))
            effects(target, Role.RUNNER, 0)
        }
        val result = commands.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash, {}, { action(actor, target) })
        assertNotNull(result.messageId); effects(target, Role.RUNNER, 1)
    }

    @Test fun `completion failure rolls back already-written message and decision`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        val key = IdempotencyKey(UUID.randomUUID().toString())
        assertFailsWith<DataAccessException> { commands.execute(key, actor, AdminActionOperation.USER_MESSAGE,
            target, "a".repeat(64), {}, { action(actor, target); AdminActionResult(null, null, null) }) }
        TransactionTemplate(manager).executeWithoutResult {
            assertEquals(0, count("admin_action_commands", "command_id", UUID.fromString(key.value)))
            effects(target, Role.RUNNER, 0)
        }
    }

    @Test fun `database protects completed identity results state and retention`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER); val key = UUID.randomUUID().toString()
        assertEquals(200, send(actor, target, key).status)
        val id = UUID.fromString(key)
        for (assignment in listOf("actor_id = gen_random_uuid()", "operation = 'EVENT_CREATE'", "target_id = gen_random_uuid()",
            "input_hash = repeat('b',64)", "result_message_id = gen_random_uuid()", "state = 'PENDING'",
            "claim_owner = gen_random_uuid()", "intended_storage_key = 'new-key'", "completed_at = NULL")) {
            assertFailsWith<DataAccessException> { jdbc.update("UPDATE admin_action_commands SET $assignment WHERE command_id = ?", id) }
        }
        assertFailsWith<DataAccessException> { jdbc.update("DELETE FROM admin_action_commands WHERE command_id = ?", id) }
        assertEquals(200, send(actor, target, key).status); effects(target, Role.RUNNER, 1)
    }

    @Test fun `receipt states provide typed outcomes and never execute a new message`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        for ((state, code) in listOf("PENDING" to "IDEMPOTENT_REPLAY_IN_PROGRESS", "UNKNOWN" to "ADMIN_COMMAND_RECOVERY_REQUIRED", "FAILED" to "CONFLICT")) {
            val key = UUID.randomUUID()
            receipts.saveAndFlush(AdminActionCommand(commandId = key, actorId = actor, operation = AdminActionOperation.USER_MESSAGE,
                targetId = target, inputHash = commands.inputHash(listOf("Hello", "Message")),
                state = AdminActionCommandState.valueOf(state), claimedAt = java.time.Instant.now(), updatedAt = java.time.Instant.now()))
            val response = send(actor, target, key.toString())
            assertEquals(409, response.status); assertEquals(code, mapper.readTree(response.contentAsString)["errors"][0]["code"].asText())
        }
        effects(target, Role.RUNNER, 0)
    }

    @Test fun `winner committing after lookup but before claim still replays`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        val key = IdempotencyKey(UUID.randomUUID().toString()); val hash = "a".repeat(64)
        val missed = CountDownLatch(1); val resume = CountDownLatch(1)
        val calls = AtomicInteger(); val loserActions = AtomicInteger()
        val delayedClock = object : Clock() {
            override fun getZone(): ZoneId = ZoneOffset.UTC
            override fun withZone(zone: ZoneId): Clock = this
            override fun instant(): Instant {
                if (calls.incrementAndGet() == 1) {
                    missed.countDown(); check(resume.await(15, TimeUnit.SECONDS))
                }
                return Instant.now()
            }
        }
        val delayed = AdminActionCommandService(receipts, manager, mapper, delayedClock)
        val pool = Executors.newSingleThreadExecutor()
        try {
            val loser = pool.submit<AdminActionResult> { delayed.execute(key, actor, AdminActionOperation.USER_MESSAGE,
                target, hash, {}, { loserActions.incrementAndGet(); action(actor, target) }) }
            assertTrue(missed.await(10, TimeUnit.SECONDS), "Loser must observe no receipt before winner commits")
            val winner = try { commands.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash, {}, { action(actor, target) }) }
                finally { resume.countDown() }
            assertEquals(winner, loser.get(20, TimeUnit.SECONDS))
            assertEquals(0, loserActions.get()); effects(target, Role.RUNNER, 1)
        } finally { resume.countDown(); pool.shutdownNow() }
    }
}
