package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.width
import androidx.compose.ui.Modifier
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.data.remote.OrderDetailDto
import com.quickpitik.mobile.data.remote.OrderPhotoDetailDto
import com.quickpitik.mobile.data.remote.RunnerDisputeDto
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * The manual-refund labels are long. On a narrow phone they must wrap in the space
 * left over instead of squeezing "Photo N" (request dialog) or the time (timeline)
 * onto several lines. Each test checks the text keeps the width it has where nothing
 * squeezes it, which on a phone means it stays on one line.
 *
 * shortcut: Robolectric 4.11's default graphics measure text at ~1px per letter and never
 * wrap it, and it has no native graphics for Windows, so the widths below are scaled down
 * until the long label overflows the way it does on a phone; switch to
 * @GraphicsMode(NATIVE), real phone widths and line-height checks once the Robolectric
 * version here runs native graphics on Windows.
 */
@RunWith(RobolectricTestRunner::class)
class RefundLabelLayoutTest {
    @get:Rule val compose = createComposeRule()

    @Test @Config(qualifiers = "w240dp-h640dp")
    fun `a long refund label in the request dialog does not squeeze Photo N`() {
        val waiting = dispute("p1", "escalated", refundStatus = "manual_review")
        val order = OrderDetailDto(
            id = "o1", eventId = "e1", photoIds = listOf("p1", "p2"), total = 300.0, paymentMethod = "QRPH",
            paidAt = null, eventName = null, eventSlug = null, status = "FULFILLED",
            photos = listOf(OrderPhotoDetailDto("p1", null, "", 0), OrderPhotoDetailDto("p2", null, "", 0)),
            disputes = listOf(waiting),
        )
        compose.setContent {
            QuickPitikMobileTheme {
                RefundRequestDialog(order = order, submitting = false, onDismiss = {}, onSubmit = { _, _, _ -> })
            }
        }

        compose.onNodeWithText("Refund approved, waiting for PayMongo refund link").assertExists()
        // Photo 2 has no refund label, so nothing squeezes it.
        val photo1 = compose.onNodeWithText("Photo 1").fetchSemanticsNode().size.width
        val photo2 = compose.onNodeWithText("Photo 2").fetchSemanticsNode().size.width
        assertTrue("Photo 1 is $photo1 px wide, Photo 2 is $photo2 px", photo1 >= photo2)
    }

    @Test fun `a long refund label in the timeline does not squeeze the time`() {
        // Filed and cancelled at the same minute, so both lines show the same time text.
        val cancelled = dispute(
            "p1", "denied", refundStatus = "cancelled",
            openedAt = "2026-10-30T04:00:00Z", resolvedAt = "2026-10-30T04:00:00Z",
        )
        compose.setContent {
            QuickPitikMobileTheme {
                Box(Modifier.width(100.dp)) { RefundTimeline(listOf(cancelled)) }
            }
        }

        compose.onNodeWithText("Refund cancelled: the refund window closed").assertExists()
        // The "Filed" line has a short label, so nothing squeezes its time.
        val (filed, cancelledTime) = compose.onAllNodesWithText("2026-10-30 · 12:00")
            .fetchSemanticsNodes().map { it.size.width }
        assertEquals("the cancelled line's time is squeezed", filed, cancelledTime)
    }

    private fun dispute(
        photoId: String,
        status: String,
        refundStatus: String? = null,
        openedAt: String = "2026-10-26T02:00:00Z",
        resolvedAt: String? = null,
    ) = RunnerDisputeDto(
        id = "d-$photoId", photoId = photoId, reason = "other", note = "", status = status,
        openedAt = openedAt, resolvedAt = resolvedAt, refundStatus = refundStatus,
    )
}
