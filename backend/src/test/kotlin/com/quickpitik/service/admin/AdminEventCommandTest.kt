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
            { null }, { _, _ -> error("no cover") }, action)

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
            { error("old revision") },{ error("image decoding") },{ _, _ -> error("storage") },{ _, _ -> error("duplicate edit") })
        assertEquals(id,result.resourceId); assertEquals(receipt.resultDecisionId,result.decisionId)
    }

    @Test fun `invalid timeout cannot enable command execution`() {
        assertFailsWith<IllegalArgumentException> { AdminActionProperties(Duration.ZERO) }
        assertFailsWith<IllegalArgumentException> { AdminActionProperties(Duration.ofSeconds(-1)) }
    }

    private fun <T> anyArg(): T = Mockito.any()
}
