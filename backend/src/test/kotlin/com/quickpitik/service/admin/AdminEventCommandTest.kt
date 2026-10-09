package com.quickpitik.service.admin

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.config.AdminActionProperties
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.AdminActionCommandRepository
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.*

class AdminEventCommandTest {
    private val receipts = Mockito.mock(AdminActionCommandRepository::class.java)
    private val now = Instant.parse("2026-10-08T12:00:00Z")
    private val key = IdempotencyKey(UUID.randomUUID().toString())
    private val actor = UUID.randomUUID()
    private val id = UUID.randomUUID()
    private val service = AdminActionCommandService(receipts, testTransactionTemplate().transactionManager!!,
        ObjectMapper(), Clock.fixed(now, ZoneOffset.UTC))

    private fun run(action: (UUID, String?) -> AdminActionResult = { _, _ -> error("must not execute") }) =
        service.executeEvent(key, actor, AdminActionOperation.EVENT_UPDATE, id, "a".repeat(64), {}, {},
            { null }, { _, _ -> error("no cover") }, { error("no cover") }, action)

    private val coverKey = "events/$id/cover/${key.value}.jpg"
    private val deleted = mutableListOf<String>()
    private fun claimedWithCover(): () -> AdminActionCommand? {
        var claimed: AdminActionCommand? = null
        Mockito.`when`(receipts.insertEventClaim(anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg()))
            .thenAnswer {
                claimed = AdminActionCommand(commandId=UUID.fromString(key.value),actorId=actor,
                    operation=AdminActionOperation.EVENT_UPDATE,targetId=id,inputHash="a".repeat(64),
                    claimOwner=it.getArgument(5),leaseExpiresAt=now.plusSeconds(300),intendedEventId=id,
                    intendedStorageKey=coverKey,processedCoverDigest="b".repeat(64),claimedAt=now,updatedAt=now)
                1
            }
        Mockito.`when`(receipts.findForUpdate(UUID.fromString(key.value))).thenAnswer { claimed }
        return { claimed }
    }
    private fun runWithCover(write: (String, Boolean) -> Unit = { _, _ -> }, action: (UUID, String?) -> AdminActionResult) =
        service.executeEvent(key, actor, AdminActionOperation.EVENT_UPDATE, id, "a".repeat(64), {}, {},
            { "b".repeat(64) }, write, { deleted += it }, action)

    @Test fun `typed failure after the cover is written fails the receipt and deletes that cover`() {
        val claimed = claimedWithCover()
        val error = assertFailsWith<ApiException> { runWithCover { _, _ ->
            throw ApiException(org.springframework.http.HttpStatus.CONFLICT, "INVALID_STATE_TRANSITION", "Event has changed") } }
        assertEquals("INVALID_STATE_TRANSITION", error.code)
        assertEquals(AdminActionCommandState.FAILED, claimed()!!.state)
        assertEquals(listOf(coverKey), deleted)
    }

    @Test fun `unknown failure after the cover is written stays UNKNOWN and keeps the cover`() {
        val claimed = claimedWithCover()
        val error = assertFailsWith<ApiException> { runWithCover { _, _ -> throw IllegalStateException("connection reset") } }
        assertEquals("ADMIN_COMMAND_RECOVERY_REQUIRED", error.code)
        assertEquals(AdminActionCommandState.UNKNOWN, claimed()!!.state)
        assertTrue(deleted.isEmpty())
    }

    @Test fun `typed failure of the cover write itself stays UNKNOWN and deletes nothing`() {
        val claimed = claimedWithCover()
        assertFailsWith<ApiException> { runWithCover(write = { _, _ ->
            throw ApiException(org.springframework.http.HttpStatus.CONFLICT, "ADMIN_COMMAND_RECOVERY_REQUIRED", "Cover object does not match") }) {
            _, _ -> error("must not execute") } }
        assertEquals(AdminActionCommandState.UNKNOWN, claimed()!!.state)
        assertTrue(deleted.isEmpty())
    }

    @Test fun `failed final event validation preserves its domain error and unresolved receipt`() {
        var claimed: AdminActionCommand? = null
        Mockito.`when`(receipts.insertEventClaim(anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),anyArg(),Mockito.isNull(),Mockito.isNull()))
            .thenAnswer {
                claimed = AdminActionCommand(commandId=UUID.fromString(key.value),actorId=actor,
                    operation=AdminActionOperation.EVENT_UPDATE,targetId=id,inputHash="a".repeat(64),
                    claimOwner=it.getArgument(5),leaseExpiresAt=now.plusSeconds(300),intendedEventId=id,
                    claimedAt=now,updatedAt=now)
                1
            }
        Mockito.`when`(receipts.findForUpdate(UUID.fromString(key.value))).thenAnswer { claimed }
        val error = assertFailsWith<NotFoundException> { run { _, _ -> throw NotFoundException("Deleted", "EVENT_NOT_FOUND") } }
        assertEquals("EVENT_NOT_FOUND",error.code)
        assertEquals(AdminActionCommandState.UNKNOWN,claimed!!.state)
        assertNull(claimed!!.resultResourceId)
    }

    @Test fun `legacy unresolved event receipt without lease requires administrative reconciliation`() {
        val receipt=AdminActionCommand(commandId=UUID.fromString(key.value),actorId=actor,
            operation=AdminActionOperation.EVENT_UPDATE,targetId=id,inputHash="a".repeat(64),
            state=AdminActionCommandState.UNKNOWN,claimedAt=now,updatedAt=now)
        Mockito.`when`(receipts.findForUpdate(receipt.commandId)).thenReturn(receipt)
        val error=assertFailsWith<ApiException> { run() }
        assertEquals("ADMIN_COMMAND_RECOVERY_REQUIRED",error.code)
    }

    @Test fun `completed event receipt skips preparation and now old revision checks`() {
        val receipt=AdminActionCommand(commandId=UUID.fromString(key.value),actorId=actor,
            operation=AdminActionOperation.EVENT_UPDATE,targetId=id,inputHash="a".repeat(64),
            state=AdminActionCommandState.SUCCEEDED,claimedAt=now,updatedAt=now,resultResourceId=id,
            resultDecisionId=UUID.randomUUID(),completedAt=now)
        Mockito.`when`(receipts.findForUpdate(receipt.commandId)).thenReturn(receipt)
        val result=service.executeEvent(key,actor,AdminActionOperation.EVENT_UPDATE,id,"a".repeat(64),{},
            { error("old revision") },{ error("image decoding") },{ _, _ -> error("storage") },{ error("storage") },{ _, _ -> error("duplicate edit") })
        assertEquals(id,result.resourceId); assertEquals(receipt.resultDecisionId,result.decisionId)
    }

    @Test fun `invalid timeout cannot enable command execution`() {
        assertFailsWith<IllegalArgumentException> { AdminActionProperties(Duration.ZERO) }
        assertFailsWith<IllegalArgumentException> { AdminActionProperties(Duration.ofSeconds(-1)) }
    }

    private fun <T> anyArg(): T = Mockito.any()
}
