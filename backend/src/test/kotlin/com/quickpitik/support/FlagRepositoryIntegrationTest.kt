package com.quickpitik.support

import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.repository.FlagRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.util.UUID
import kotlin.test.assertEquals

// Pins SystemFlagService's duplicate lookup against real Postgres: only open and
// escalated rows of one (target kind, target, reason) come back.
class FlagRepositoryIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var flags: FlagRepository

    private fun save(kind: FlagTargetKind, targetId: UUID, reason: SystemFlagReason, status: FlagStatus): Flag =
        flags.save(Flag(targetKindWire = kind.wire, targetId = targetId, reason = reason.wire, statusWire = status.wire))

    @Test
    fun `the duplicate lookup returns only open and escalated rows of that target and reason`() {
        val target = UUID.randomUUID()
        val reason = SystemFlagReason.POSSIBLE_DUPLICATE
        val open = save(FlagTargetKind.EVENT, target, reason, FlagStatus.OPEN)
        val escalated = save(FlagTargetKind.EVENT, target, reason, FlagStatus.ESCALATED)
        for (closed in listOf(FlagStatus.RESOLVED, FlagStatus.DISMISSED, FlagStatus.HIDDEN)) {
            save(FlagTargetKind.EVENT, target, reason, closed)
        }
        save(FlagTargetKind.EVENT, target, SystemFlagReason.RACE_CHECK_MISS, FlagStatus.OPEN) // other reason
        save(FlagTargetKind.EVENT, UUID.randomUUID(), reason, FlagStatus.OPEN) // other target
        save(FlagTargetKind.USER, target, reason, FlagStatus.OPEN) // other kind

        val found = flags.findByTargetKindWireAndTargetIdAndReasonAndStatusWireIn(
            FlagTargetKind.EVENT.wire, target, reason.wire, listOf(FlagStatus.OPEN.wire, FlagStatus.ESCALATED.wire),
        )

        assertEquals(setOf(open.id, escalated.id), found.map { it.id }.toSet())
    }
}
