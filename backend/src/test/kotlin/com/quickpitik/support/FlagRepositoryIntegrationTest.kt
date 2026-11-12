package com.quickpitik.support

import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.repository.FlagRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

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

    // PE-2.2: the payout-hold reads.
    private val relevance = SystemFlagReason.BATCH_RELEVANCE_LOW

    private fun held(kind: FlagTargetKind, targetId: UUID, reason: SystemFlagReason, status: FlagStatus): Flag =
        flags.save(
            Flag(
                targetKindWire = kind.wire, targetId = targetId, reason = reason.wire, statusWire = status.wire,
                payoutHoldAt = OffsetDateTime.now(),
            ),
        )

    @Test
    fun `an open hold is found only on an open or escalated relevance flag that carries the hold stamp`() {
        val openHeld = UUID.randomUUID().also { held(FlagTargetKind.USER, it, relevance, FlagStatus.OPEN) }
        val escalatedHeld = UUID.randomUUID().also { held(FlagTargetKind.USER, it, relevance, FlagStatus.ESCALATED) }
        val notHeld = UUID.randomUUID().also { save(FlagTargetKind.USER, it, relevance, FlagStatus.OPEN) }
        val closed = UUID.randomUUID().also { id ->
            for (s in listOf(FlagStatus.RESOLVED, FlagStatus.DISMISSED, FlagStatus.HIDDEN)) held(FlagTargetKind.USER, id, relevance, s)
        }
        val otherReason = UUID.randomUUID().also { held(FlagTargetKind.USER, it, SystemFlagReason.PAYOUT_NAME_CHANGED, FlagStatus.OPEN) }
        val otherKind = UUID.randomUUID().also { held(FlagTargetKind.EVENT, it, relevance, FlagStatus.OPEN) }

        assertTrue(flags.existsOpenPayoutHold(openHeld, relevance.wire))
        assertTrue(flags.existsOpenPayoutHold(escalatedHeld, relevance.wire))
        for (id in listOf(notHeld, closed, otherReason, otherKind, UUID.randomUUID())) {
            assertFalse(flags.existsOpenPayoutHold(id, relevance.wire))
        }
    }

    @Test
    fun `open relevance flags are read for many photographers in one query`() {
        val a = UUID.randomUUID()
        val b = UUID.randomUUID()
        val aFlag = save(FlagTargetKind.USER, a, relevance, FlagStatus.OPEN)
        val bFlag = held(FlagTargetKind.USER, b, relevance, FlagStatus.ESCALATED)
        save(FlagTargetKind.USER, a, relevance, FlagStatus.RESOLVED)
        save(FlagTargetKind.USER, b, SystemFlagReason.PAYOUT_NAME_CHANGED, FlagStatus.OPEN)
        save(FlagTargetKind.USER, UUID.randomUUID(), relevance, FlagStatus.OPEN) // not asked for

        val found = flags.findOpenByUserTargetsAndReason(listOf(a, b), relevance.wire)

        assertEquals(setOf(aFlag.id, bFlag.id), found.map { it.id }.toSet())
    }

    @Test
    fun `the relevance target is returned for a relevance user flag and nothing for any other flag`() {
        val photographer = UUID.randomUUID()
        val relevanceFlag = save(FlagTargetKind.USER, photographer, relevance, FlagStatus.OPEN)
        val otherReason = save(FlagTargetKind.USER, photographer, SystemFlagReason.PAYOUT_NAME_CHANGED, FlagStatus.OPEN)
        val photoFlag = save(FlagTargetKind.PHOTO, UUID.randomUUID(), relevance, FlagStatus.OPEN)

        assertEquals(photographer, flags.findRelevanceUserTarget(relevanceFlag.id, relevance.wire))
        assertNull(flags.findRelevanceUserTarget(otherReason.id, relevance.wire))
        assertNull(flags.findRelevanceUserTarget(photoFlag.id, relevance.wire))
        assertNull(flags.findRelevanceUserTarget(UUID.randomUUID(), relevance.wire))
    }
}
