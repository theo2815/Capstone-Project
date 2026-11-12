package com.quickpitik.service.admin

import com.quickpitik.entity.Flag
import com.quickpitik.entity.FlagStatus
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.repository.FlagRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SystemFlagServiceTest {

    private lateinit var flagRepository: FlagRepository
    private val targetId = UUID.randomUUID()
    private val openStatuses = listOf(FlagStatus.OPEN.wire, FlagStatus.ESCALATED.wire)

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        flagRepository = Mockito.mock(FlagRepository::class.java)
        Mockito.`when`(flagRepository.save(anyArg<Flag>())).thenAnswer { it.arguments[0] }
        stubOpen()
    }

    private fun service() = SystemFlagService(flagRepository)

    // The rows the duplicate lookup returns: only open or escalated flags of this target and reason.
    private fun stubOpen(vararg rows: Flag, reason: SystemFlagReason = SystemFlagReason.PROFILE_IMAGE_FLAGGED) {
        Mockito.`when`(
            flagRepository.findByTargetKindWireAndTargetIdAndReasonAndStatusWireIn(
                FlagTargetKind.USER.wire, targetId, reason.wire, openStatuses,
            ),
        ).thenReturn(rows.toList())
    }

    private fun existing(reporterId: UUID? = null, status: FlagStatus = FlagStatus.OPEN, note: String = "") = Flag(
        targetKindWire = FlagTargetKind.USER.wire,
        targetId = targetId,
        reporterId = reporterId,
        reason = SystemFlagReason.PROFILE_IMAGE_FLAGGED.wire,
        note = note,
        statusWire = status.wire,
    )

    private fun saved(): Flag {
        val captor = ArgumentCaptor.forClass(Flag::class.java)
        Mockito.verify(flagRepository).save(captor.capture())
        return captor.value
    }

    @Test
    fun `raise writes an open flag with no reporter`() {
        val wrote = service().raise(FlagTargetKind.USER, targetId, SystemFlagReason.PROFILE_IMAGE_FLAGGED, note = "labels: Nudity")

        assertTrue(wrote)
        val flag = saved()
        assertEquals("user", flag.targetKindWire)
        assertEquals(targetId, flag.targetId)
        assertEquals("profile_image_flagged", flag.reason)
        assertEquals(FlagStatus.OPEN, flag.status)
        assertNull(flag.reporterId)
        assertEquals("labels: Nudity", flag.note)
    }

    @Test
    fun `raise skips while the same finding is open or escalated and returns false`() {
        stubOpen(existing(status = FlagStatus.ESCALATED))

        assertFalse(service().raise(FlagTargetKind.USER, targetId, SystemFlagReason.PROFILE_IMAGE_FLAGGED))
        Mockito.verify(flagRepository, Mockito.never()).save(anyArg<Flag>())
    }

    @Test
    fun `raise writes again once the earlier flag is resolved or dismissed`() {
        // Resolved and dismissed rows are not returned by the open-or-escalated lookup.
        stubOpen()

        assertTrue(service().raise(FlagTargetKind.USER, targetId, SystemFlagReason.PROFILE_IMAGE_FLAGGED))
        saved()
    }

    @Test
    fun `two reporters make two rows and the same reporter twice makes one`() {
        val alice = UUID.randomUUID()
        val bob = UUID.randomUUID()
        stubOpen(existing(reporterId = alice), reason = SystemFlagReason.USER_REPORT)

        assertFalse(service().raise(FlagTargetKind.USER, targetId, SystemFlagReason.USER_REPORT, reporterId = alice))
        assertTrue(service().raise(FlagTargetKind.USER, targetId, SystemFlagReason.USER_REPORT, reporterId = bob))
        assertEquals(bob, saved().reporterId)
    }

    @Test
    fun `a user report does not hide behind an open system flag`() {
        stubOpen(existing(reporterId = null), reason = SystemFlagReason.USER_REPORT)

        assertTrue(service().raise(FlagTargetKind.USER, targetId, SystemFlagReason.USER_REPORT, reporterId = UUID.randomUUID()))
    }

    @Test
    fun `raise keeps the note to 2000 characters`() {
        service().raise(FlagTargetKind.USER, targetId, SystemFlagReason.PROFILE_IMAGE_FLAGGED, note = "x".repeat(2500))

        assertEquals(2000, saved().note.length)
    }

    @Test
    fun `appendNote adds a line to the open system flag and writes nothing when none is open`() {
        val open = existing(note = "first try")
        stubOpen(existing(reporterId = UUID.randomUUID()), open)

        assertTrue(service().appendNote(FlagTargetKind.USER, targetId, SystemFlagReason.PROFILE_IMAGE_FLAGGED, "second try"))
        assertEquals("first try\nsecond try", saved().note)

        Mockito.reset(flagRepository)
        stubOpen(existing(reporterId = UUID.randomUUID()))
        assertFalse(service().appendNote(FlagTargetKind.USER, targetId, SystemFlagReason.PROFILE_IMAGE_FLAGGED, "third try"))
        Mockito.verify(flagRepository, Mockito.never()).save(anyArg<Flag>())
    }

    @Test
    fun `appendNote keeps the last 2000 characters`() {
        val open = existing(note = "a".repeat(1990))
        stubOpen(open)

        service().appendNote(FlagTargetKind.USER, targetId, SystemFlagReason.PROFILE_IMAGE_FLAGGED, "b".repeat(20))

        val note = saved().note
        assertEquals(2000, note.length)
        assertTrue(note.endsWith("b".repeat(20)))
    }

    @Test
    fun `every reason fits the 40-character column`() {
        assertEquals(13, SystemFlagReason.entries.size) // RD-6 added runner_removal_held
        assertTrue(SystemFlagReason.entries.all { it.wire.length <= 40 && it.wire == it.name.lowercase() })
    }
}
