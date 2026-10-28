package com.quickpitik.support

import com.quickpitik.entity.AdminDecisionLog
import com.quickpitik.entity.REFUND_SENT_TO_MANUAL_REVIEW
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.AdminDecisionLogRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.time.LocalDate
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.UUID
import kotlin.test.assertEquals

// Pins the admin KPI decision counts: refund hand-off rows are steps of one
// refund decision, not decisions of their own (vault backend/tasks.md, KPI double count).
class AdminDecisionLogKpiIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var decisions: AdminDecisionLogRepository

    @Autowired
    private lateinit var users: UserRepository

    // A week no other test writes to, because the container is shared across classes.
    private val day = OffsetDateTime.of(2001, 1, 3, 12, 0, 0, 0, ZoneOffset.UTC)
    private val from = day.minusDays(2)
    private val to = day.plusDays(5)

    private fun log(adminId: UUID, decision: String) = decisions.save(
        AdminDecisionLog(adminId = adminId, decision = decision, decidedAt = day),
    )

    @Test
    fun `refund hand-off rows do not count as admin decisions`() {
        val admin = users.save(
            User(email = "kpi-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "KPI Test Admin", role = Role.ADMIN),
        )
        // One approved refund: hand-off, link sent, then resolved. Counts as one decision.
        log(admin.id, REFUND_SENT_TO_MANUAL_REVIEW)
        log(admin.id, "refund_link_sent")
        log(admin.id, "resolved")
        // An auto-cancelled refund is not a new admin decision either.
        log(admin.id, "refund_cancelled")
        // An unrelated decision still counts.
        log(admin.id, "denied")

        assertEquals(2, decisions.countLogicalDecisionsBetween(from, to))
        val byDay = decisions.decisionCountsByDayBetween(from, to)
            .associate { it[0].toString() to (it[1] as Number).toLong() }
        assertEquals(mapOf(LocalDate.of(2001, 1, 3).toString() to 2L), byDay)
    }
}
