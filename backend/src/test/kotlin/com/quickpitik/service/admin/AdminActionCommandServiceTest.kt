package com.quickpitik.service.admin

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.AdminActionCommandRepository
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import org.junit.jupiter.params.provider.ValueSource
import org.mockito.Mockito
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import java.time.Clock
import java.time.Instant
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.*

class AdminActionCommandServiceTest {
    private val repository = Mockito.mock(AdminActionCommandRepository::class.java)
    private val manager = Mockito.mock(PlatformTransactionManager::class.java)
    private val now = Instant.parse("2026-10-08T00:00:00Z")
    private val service = AdminActionCommandService(repository, manager, ObjectMapper(), Clock.fixed(now, ZoneOffset.UTC))
    private val actor = UUID.randomUUID()
    private val target = UUID.randomUUID()
    private val key = IdempotencyKey(UUID.randomUUID().toString())
    private val hash = "a".repeat(64)
    private val result = AdminActionResult(target, UUID.randomUUID(), UUID.randomUUID())

    init { Mockito.`when`(manager.getTransaction(Mockito.any())).thenAnswer { SimpleTransactionStatus() } }
    private fun receipt(state: AdminActionCommandState = AdminActionCommandState.SUCCEEDED) = AdminActionCommand(
        commandId = UUID.fromString(key.value), actorId = actor, operation = AdminActionOperation.USER_MESSAGE,
        targetId = target, inputHash = hash, claimedAt = now, updatedAt = now, state = state,
        resultResourceId = result.resourceId, resultDecisionId = result.decisionId, resultMessageId = result.messageId,
        completedAt = if (state == AdminActionCommandState.SUCCEEDED) now else null,
    )

    @Test fun `completed receipt replays original IDs without running action`() {
        Mockito.`when`(repository.findById(UUID.fromString(key.value))).thenReturn(Optional.of(receipt()))
        var authorized = false
        assertEquals(result, service.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash,
            { authorized = true }, { error("duplicate send") }))
        assertTrue(authorized)
    }

    @ParameterizedTest @ValueSource(strings = ["actor", "operation", "target", "payload"])
    fun `each identity mismatch conflicts`(field: String) {
        Mockito.`when`(repository.findById(UUID.fromString(key.value))).thenReturn(Optional.of(receipt()))
        val ex = assertFailsWith<ApiException> { service.execute(key,
            if (field == "actor") UUID.randomUUID() else actor,
            if (field == "operation") AdminActionOperation.EVENT_CREATE else AdminActionOperation.USER_MESSAGE,
            if (field == "target") UUID.randomUUID() else target,
            if (field == "payload") "b".repeat(64) else hash, {}, { error("duplicate action") }) }
        assertEquals("CONFLICT", ex.code)
    }

    @ParameterizedTest @EnumSource(value = AdminActionCommandState::class, names = ["PENDING", "UNKNOWN", "FAILED"])
    fun `unsettled receipts never run another action`(state: AdminActionCommandState) {
        Mockito.`when`(repository.findById(UUID.fromString(key.value))).thenReturn(Optional.of(receipt(state)))
        val ex = assertFailsWith<ApiException> { service.execute(key, actor, AdminActionOperation.USER_MESSAGE,
            target, hash, {}, { error("duplicate action") }) }
        assertEquals(when(state) {
            AdminActionCommandState.PENDING -> "IDEMPOTENT_REPLAY_IN_PROGRESS"
            AdminActionCommandState.UNKNOWN -> "ADMIN_COMMAND_RECOVERY_REQUIRED"
            else -> "CONFLICT"
        }, ex.code)
    }

    @Test fun `authorization fails before reading a receipt`() {
        assertFailsWith<IllegalStateException> { service.execute(key, actor, AdminActionOperation.USER_MESSAGE,
            target, hash, { error("forbidden") }, { error("send") }) }
        Mockito.verifyNoInteractions(repository)
    }

    @Test fun `new command saves only stable IDs and completes at the injected time`() {
        Mockito.`when`(repository.findById(UUID.fromString(key.value))).thenReturn(Optional.empty())
        val saved = mutableListOf<AdminActionCommand>()
        Mockito.`when`(repository.saveAndFlush(Mockito.any(AdminActionCommand::class.java))).thenAnswer {
            val receipt = it.getArgument<AdminActionCommand>(0); saved.add(receipt); receipt
        }
        assertEquals(result, service.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash, {}, { result }))
        assertEquals(AdminActionCommandState.SUCCEEDED, saved.last().state)
        assertEquals(now, saved.last().completedAt)
        assertEquals(result.messageId, saved.last().resultMessageId)
    }

    @Test fun `canonical field encoding cannot confuse field boundaries or separators`() {
        assertNotEquals(service.inputHash(listOf("ab", "c")), service.inputHash(listOf("a", "bc")))
        assertNotEquals(service.inputHash(listOf("a\nb", "c")), service.inputHash(listOf("a", "b\nc")))
        assertEquals(64, service.inputHash(listOf("Hello", "Message")).length)
    }
}
