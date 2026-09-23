package com.quickpitik.support

import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutReport
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PayoutReportRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.earnings.PayoutReportService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

// GET /me/photographer/payouts/reports pages like its sibling endpoints: the
// photographer/cycle/status predicates hold on both the page and the count.
class PayoutReportPaginationIntegrationTest : PostgresIntegrationTest() {

    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var cycles: PayoutCycleRepository
    @Autowired lateinit var reports: PayoutReportRepository
    @Autowired lateinit var service: PayoutReportService

    private fun photographer() = users.saveAndFlush(User(
        email = "reports-${UUID.randomUUID()}@test.local", passwordHash = "test-only-unused",
        name = "Report fixture", role = Role.PHOTOGRAPHER,
    ))

    // Terminal status: a photographer may hold only one open cycle.
    private fun cycle(owner: User) = cycles.saveAndFlush(PayoutCycle(
        id = "PAY-REPORTS-${UUID.randomUUID()}", photographerId = owner.id,
        weekOf = LocalDate.now(), amountPhp = BigDecimal("1000"), statusWire = "paid",
    ))

    private fun report(owner: User, cycle: PayoutCycle, minutesAgo: Long, status: String = "open") =
        reports.saveAndFlush(PayoutReport(
            payoutId = cycle.id, photographerId = owner.id, reasonWire = "not_received",
            statusWire = status, openedAt = OffsetDateTime.now().minusMinutes(minutesAgo),
        ))

    @Test
    fun `reports are paged newest first with a total scoped to the photographer and filters`() {
        val me = photographer()
        val first = cycle(me)
        val second = cycle(me)
        report(me, first, minutesAgo = 30, status = "resolved")
        val middle = report(me, second, minutesAgo = 20)
        report(me, first, minutesAgo = 10)
        val other = photographer()
        report(other, cycle(other), minutesAgo = 5)

        val page = service.list(me.id, null, null, PaginationParams.of(1, 1))
        assertEquals(3L, page.total)
        assertEquals(listOf(middle.id.toString()), page.items.map { it.id })

        assertEquals(2L, service.list(me.id, null, "open", PaginationParams.of(0, 50)).total)
        assertEquals(2, service.list(me.id, first.id, null, PaginationParams.of(0, 50)).items.size)
    }
}
