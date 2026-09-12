package com.quickpitik.mobile.ui.runner

import com.quickpitik.mobile.data.remote.OrderListItemDto
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The settled-order rule, read by three callers: the cart's owned-photo flags,
 * the checkout sheet's no-redirect route to the receipt, and the receipt poll's
 * terminal branch. A status wrongly read as settled clears a cart that was never
 * paid for; wrongly read as unsettled leaves the runner polling a paid order.
 */
class OwnedOrdersTest {

    private fun order(status: String?, vararg photoIds: String) =
        OrderListItemDto("o-$status", "ev-1", photoIds.toList(), 0.0, "paymongo", null, null, null, null, null, status)

    @Test fun `only PAID and FULFILLED are settled, in any case`() {
        assertTrue(isSettledOrderStatus("PAID"))
        assertTrue(isSettledOrderStatus("FULFILLED"))
        assertTrue(isSettledOrderStatus("paid"))
        assertTrue(isSettledOrderStatus("Fulfilled"))
        for (open in listOf("PENDING", "EXPIRED", "REFUNDED", "", "PAID_OUT", null)) {
            assertFalse(open.toString(), isSettledOrderStatus(open))
        }
    }

    @Test fun `settled photo ids come from settled orders only, and an unloaded list owns nothing`() {
        val orders = listOf(
            order("PAID", "p1", "p2"),
            order("FULFILLED", "p3"),
            order("PENDING", "p4"),
            order("REFUNDED", "p5"),
            order(null, "p6"),
        )
        assertEquals(setOf("p1", "p2", "p3"), settledPhotoIds(OrdersState.Success(orders)))
        // Loading / error means "we do not know yet" — never flag a duplicate on
        // a guess; the backend 409 is the net.
        assertEquals(emptySet<String>(), settledPhotoIds(OrdersState.Loading))
        assertEquals(emptySet<String>(), settledPhotoIds(OrdersState.Error("offline")))
        assertEquals(emptySet<String>(), settledPhotoIds(OrdersState.Success(emptyList())))
    }
}
