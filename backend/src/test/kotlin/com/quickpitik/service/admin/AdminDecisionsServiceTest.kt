package com.quickpitik.service.admin

import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.Pageable
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNull

class AdminDecisionsServiceTest {

    private val log = Mockito.mock(AdminDecisionLogRepository::class.java)
    private val users = Mockito.mock(UserRepository::class.java)
    private val service = AdminDecisionsService(log, users)

    private val adminId = UUID.randomUUID()
    private val goneId = UUID.randomUUID()
    private val userId = UUID.randomUUID()
    private val requestId = UUID.randomUUID()
    private var pageSeen: Pageable? = null

    private fun <T> anyArg(): T = Mockito.any()

    private fun stub(rows: List<AdminDecisionLog>) {
        Mockito.`when`(log.findAllByOrderByDecidedAtDescIdAsc(anyArg())).thenAnswer {
            pageSeen = it.getArgument(0)
            rows
        }
    }

    @Test
    fun `recent keeps the stored order and names each admin with one lookup`() {
        val now = OffsetDateTime.now()
        stub(listOf(
            AdminDecisionLog(adminId = adminId, targetUserId = userId, decision = "suspended", reason = "abuse", decidedAt = now),
            AdminDecisionLog(adminId = goneId, targetPayoutId = "PAY-1", decision = "PAYOUT_MARKED_PAID"),
            AdminDecisionLog(adminId = adminId, targetDisputeId = requestId, decision = "refund_resolved"),
            AdminDecisionLog(adminId = adminId, targetEventId = requestId, decision = "event_hidden"),
        ))
        Mockito.`when`(users.findAllById(setOf(adminId, goneId, userId))).thenReturn(listOf(
            User(id = adminId, email = "boss@test.local", passwordHash = "x", name = "Boss", role = Role.ADMIN),
            User(id = userId, email = "pat@test.local", passwordHash = "x", name = "Pat Photo", role = Role.PHOTOGRAPHER),
        ))

        val out = service.recent(20)

        assertEquals(listOf("suspended", "PAYOUT_MARKED_PAID", "refund_resolved", "event_hidden"), out.map { it.decision })
        assertEquals(listOf("Boss", null, "Boss", "Boss"), out.map { it.adminName })
        assertEquals(listOf("Pat Photo", null, null, null), out.map { it.targetName })
        assertEquals(goneId, out[1].adminId)
        assertEquals("abuse", out[0].reason)
        assertEquals(now, out[0].decidedAt)
        assertEquals(listOf("user" to userId.toString(), "payout" to "PAY-1", "dispute" to requestId.toString(),
            "event" to requestId.toString()), out.map { it.targetType to it.targetId })
        Mockito.verify(users, Mockito.times(1)).findAllById(anyArg<Iterable<UUID>>())
    }

    @Test
    fun `recent names a user target by e-mail when the name is blank and leaves a missing user unnamed`() {
        val blank = UUID.randomUUID()
        stub(listOf(
            AdminDecisionLog(adminId = adminId, targetUserId = blank, decision = "approved"),
            AdminDecisionLog(adminId = adminId, targetUserId = userId, decision = "approved"),
        ))
        Mockito.`when`(users.findAllById(setOf(adminId, blank, userId))).thenReturn(listOf(
            User(id = blank, email = "quiet@test.local", passwordHash = "x", name = " ", role = Role.PHOTOGRAPHER),
        ))

        assertEquals(listOf("quiet@test.local", null), service.recent(20).map { it.targetName })
        Mockito.verify(users, Mockito.times(1)).findAllById(anyArg<Iterable<UUID>>())
    }

    @Test
    fun `recent reads a photo removal target from the request id in meta`() {
        stub(listOf(AdminDecisionLog(adminId = adminId, decision = "photo_removal_approved",
            meta = mapOf("requestId" to requestId.toString(), "itemId" to UUID.randomUUID().toString()))))
        Mockito.`when`(users.findAllById(setOf(adminId))).thenReturn(emptyList())

        val row = service.recent(20).single()

        assertEquals("removal", row.targetType)
        assertEquals(requestId.toString(), row.targetId)
    }

    @Test
    fun `recent leaves the target empty when a row points at nothing`() {
        stub(listOf(AdminDecisionLog(adminId = adminId, decision = "mystery")))
        Mockito.`when`(users.findAllById(setOf(adminId))).thenReturn(emptyList())

        val row = service.recent(20).single()

        assertNull(row.targetType)
        assertNull(row.targetId)
    }

    @Test
    fun `recent defaults to 20 and clamps the limit between 1 and 100`() {
        stub(emptyList())
        for ((asked, used) in listOf<Pair<Int?, Int>>(null to 20, 0 to 1, -5 to 1, 7 to 7, 100 to 100, 5000 to 100)) {
            service.recent(asked)
            assertEquals(used, pageSeen!!.pageSize, "limit $asked")
            assertEquals(0, pageSeen!!.pageNumber)
        }
    }

    @Test
    fun `recent skips the admin lookup when there are no decisions`() {
        stub(emptyList())

        assertEquals(emptyList<com.quickpitik.dto.admin.AdminDecisionDto>(), service.recent(20))
        Mockito.verifyNoInteractions(users)
    }
}
