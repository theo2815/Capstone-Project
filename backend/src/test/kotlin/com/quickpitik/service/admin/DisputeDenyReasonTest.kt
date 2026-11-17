package com.quickpitik.service.admin

import com.quickpitik.service.orders.RefundBlocker
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DisputeDenyReasonTest {

    @Test
    fun `every reason except other has runner text of at most 250 characters`() {
        for (reason in DisputeDenyReason.entries.filter { it != DisputeDenyReason.OTHER }) {
            assertTrue(reason.runnerText.isNotBlank() && reason.runnerText.length <= 250, reason.code)
            assertTrue(!reason.noteRequired, reason.code)
        }
        assertTrue(DisputeDenyReason.OTHER.noteRequired)
    }

    @Test
    fun `codes are unique and bought_together and window_closed match the refund blockers`() {
        val codes = DisputeDenyReason.entries.map { it.code }
        assertEquals(codes.size, codes.toSet().size)
        assertEquals(9, codes.size)
        assertEquals(RefundBlocker.BOUGHT_TOGETHER.wire, DisputeDenyReason.BOUGHT_TOGETHER.code)
        assertEquals(RefundBlocker.WINDOW_CLOSED.wire, DisputeDenyReason.WINDOW_CLOSED.code)
        assertEquals(DisputeDenyReason.WRONG_PHOTO, DisputeDenyReason.fromCode("wrong_photo"))
        assertEquals(null, DisputeDenyReason.fromCode("nope"))
    }
}
