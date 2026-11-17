package com.quickpitik.support

import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.entity.User
import com.quickpitik.repository.FlagRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.util.UUID
import kotlin.test.assertEquals

// AA-R9: the report threshold counts distinct accounts with a live report.
class ReportQueriesIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var flags: FlagRepository
    @Autowired private lateinit var users: UserRepository

    @Test
    fun `distinct reporter count ignores dismissed rows`() {
        val target = UUID.randomUUID()
        val (a, b, c, d) = List(4) { user() }
        report(target, a)
        report(target, a) // a double row from one account counts once
        report(target, b, FlagStatus.ESCALATED)
        report(target, c, FlagStatus.DISMISSED)
        report(target, d, FlagStatus.RESOLVED)
        report(target, null) // a system finding has no reporter
        report(UUID.randomUUID(), c) // another target
        report(target, c, reason = SystemFlagReason.POSSIBLE_DUPLICATE.wire)
        report(target, c, kind = "user") // same id, other kind

        assertEquals(2, flags.countDistinctReporters("event", target))
        assertEquals(1, flags.countDistinctReporters("user", target))
        assertEquals(0, flags.countDistinctReporters("event", UUID.randomUUID()))
    }

    private fun user(): User = users.save(
        User(email = "rq-${UUID.randomUUID()}@test.local", passwordHash = "stub", name = "RQ", role = Role.RUNNER),
    )

    private fun report(
        target: UUID,
        reporter: User?,
        status: FlagStatus = FlagStatus.OPEN,
        reason: String = SystemFlagReason.USER_REPORT.wire,
        kind: String = "event",
    ) {
        flags.saveAndFlush(
            Flag(targetKindWire = kind, targetId = target, reporterId = reporter?.id, reason = reason, note = "n")
                .also { it.status = status },
        )
    }
}
