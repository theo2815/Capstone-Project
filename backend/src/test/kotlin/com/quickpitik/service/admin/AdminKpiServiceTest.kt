package com.quickpitik.service.admin

import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.repository.*
import com.quickpitik.service.removal.PhotoRemovalQueryService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.Page
import org.springframework.data.domain.PageImpl
import org.springframework.data.domain.Pageable
import org.springframework.jdbc.core.JdbcTemplate
import java.sql.Date
import java.time.LocalDate
import kotlin.test.assertEquals

// AX-8: each queue count comes from the query its list uses, so the badge equals the list's total.
class AdminKpiServiceTest {
    private val cycles = Mockito.mock(PayoutCycleRepository::class.java)
    private val disputes = Mockito.mock(DisputeRepository::class.java)
    private val flags = Mockito.mock(FlagRepository::class.java)
    private val sellerDetails = Mockito.mock(SellerDetailsRepository::class.java)
    private val reports = Mockito.mock(PayoutReportRepository::class.java)
    private val cases = Mockito.mock(RelevanceCaseRepository::class.java)
    private val appeals = Mockito.mock(RelevanceAppealRepository::class.java)
    private val removals = Mockito.mock(PhotoRemovalQueryService::class.java)
    private val disputeService = Mockito.mock(AdminDisputeService::class.java)
    private val photos = Mockito.mock(PhotoRepository::class.java)
    private val users = Mockito.mock(UserRepository::class.java)
    private val service = AdminKpiService(users, cycles, disputes, flags,
        Mockito.mock(EventRepository::class.java), Mockito.mock(AdminDecisionLogRepository::class.java),
        Mockito.mock(JdbcTemplate::class.java), disputeService, sellerDetails, reports, cases, appeals, removals, photos)

    private fun <T> anyArg(): T = Mockito.any()
    private fun total(n: Long): Page<Nothing> = PageImpl(emptyList(), Pageable.ofSize(1), n)

    @Test
    fun `each queue count is its list's total and the old fields stay as they were`() {
        Mockito.`when`(disputeService.queueCounts()).thenReturn(DisputeQueueCounts(needsAdmin = 2, overdue = 1))
        Mockito.`when`(cycles.countByStatusWire(PayoutCycleStatus.PENDING.wire)).thenReturn(4)
        Mockito.`when`(cycles.countByStatusWire(PayoutCycleStatus.SCHEDULED.wire)).thenReturn(3)
        Mockito.`when`(flags.countByStatusWire("open")).thenReturn(9)
        // A count read with the wrong filter gets 999, so the assertions below catch it.
        Mockito.`when`(sellerDetails.findForAdmin(anyArg(), anyArg())).thenAnswer {
            total(if (it.getArgument<SellerDetailsStatus?>(0) == SellerDetailsStatus.MANUAL_REVIEW) 5 else 999)
        }
        Mockito.`when`(reports.pageForAdmin(anyArg(), anyArg())).thenAnswer { total(if (it.getArgument<String?>(0) == "open") 6 else 999) }
        Mockito.`when`(cases.pageCurrentCases(anyArg(), Mockito.anyBoolean(), anyArg(), anyArg())).thenAnswer {
            if (it.getArgument<Boolean>(1)) total(999)
            else total(when (it.getArgument<String>(0)) { "hidden" -> 7; "cant_tell" -> 8; else -> 999 })
        }
        Mockito.`when`(appeals.pageAppeals(anyArg(), anyArg())).thenAnswer { total(if (it.getArgument<String>(0) == "OPEN") 10 else 999) }
        Mockito.`when`(cases.countNeedingPerson()).thenReturn(11)
        Mockito.`when`(removals.countNeedingAction()).thenReturn(12)
        // AX-46 / AX-58: the Pending and Approved tabs' own query, so suspended photographers are not counted.
        Mockito.`when`(users.searchForAdmin(anyArg(), anyArg(), anyArg(), anyArg())).thenAnswer {
            val tab = it.getArgument<Role?>(0) == Role.PHOTOGRAPHER && it.getArgument<String>(1) == ""
            total(when (it.getArgument<String>(2)) { "pending" -> if (tab) 13 else 999; "approved" -> if (tab) 14 else 999; else -> 999 })
        }

        val kpis = service.kpis()

        assertEquals(13L, kpis.pendingVerifications)
        assertEquals(14L, kpis.approvedPhotographers)

        assertEquals(listOf(5L, 12L, 6L, 7L, 8L, 10L, 11L, 4L), listOf(kpis.pendingSellerDetails, kpis.removalsNeedingAction,
            kpis.openPayoutReports, kpis.relevanceHidden, kpis.relevanceCantTell, kpis.relevanceAppealsOpen,
            kpis.relevanceNeedsPerson, kpis.payoutsAwaitingReview))
        assertEquals(listOf(7L, 9L, 2L, 1L), listOf(kpis.pendingPayouts, kpis.openFlags, kpis.disputesNeedingAdmin, kpis.disputesOverdue))
    }

    // AX-11: the Overview trend's uploads series, one count per day, 0 on a day with none.
    @Test
    fun `trend gives every day an uploads count, 0 when nothing was uploaded`() {
        val today = LocalDate.now(PH_ZONE)
        // Days must be cut in Manila: any other zone gets no rows, so the assertion catches it.
        Mockito.`when`(photos.uploadCountsByDayBetween(anyArg(), anyArg(), anyArg())).thenAnswer {
            if (it.getArgument<String>(2) == PH_ZONE.id) listOf(arrayOf<Any>(Date.valueOf(today), 5L)) else emptyList()
        }

        val trend = service.trend(3)

        assertEquals(listOf(0L, 0L, 5L), trend.map { it.uploads })
    }

    // AX-47: the decisions, disputes and payouts series are cut in Manila as well.
    @Test
    fun `trend passes the Manila zone to the decisions, disputes and payouts queries`() {
        val today = LocalDate.now(PH_ZONE)
        // Rows come back only when the query got the Manila zone, so a missing zone shows as 0.
        val decisionLog = Mockito.mock(AdminDecisionLogRepository::class.java)
        Mockito.`when`(decisionLog.decisionCountsByDayBetween(anyArg(), anyArg(), anyArg())).thenAnswer {
            if (it.getArgument<String>(2) == PH_ZONE.id) listOf(arrayOf<Any>(Date.valueOf(today), 4L)) else emptyList()
        }
        val jdbc = Mockito.mock(JdbcTemplate::class.java) { inv ->
            if (inv.method.name == "queryForList" && PH_ZONE.id in inv.arguments.drop(1)) {
                val cnt = if ((inv.arguments[0] as String).contains("FROM disputes")) 2L else 3L
                listOf(mapOf("day" to Date.valueOf(today), "cnt" to cnt))
            } else Mockito.RETURNS_DEFAULTS.answer(inv)
        }
        val service = AdminKpiService(users, cycles, disputes, flags, Mockito.mock(EventRepository::class.java),
            decisionLog, jdbc, disputeService, sellerDetails, reports, cases, appeals, removals, photos)

        val last = service.trend(2).last()

        assertEquals(listOf(4L, 2L, 3L), listOf(last.decisions, last.disputes, last.payouts))
    }
}
