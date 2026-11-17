package com.quickpitik.mobile.ui.runner

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class PriceNoteTest {
    @Test fun `note only when price rose`() {
        assertEquals("Price went up from ₱125.00 to ₱150.00", priceRoseNote(price = 150.0, priceAtAdd = 125.0))
        assertNull(priceRoseNote(price = 125.0, priceAtAdd = 125.0))
        assertNull(priceRoseNote(price = 100.0, priceAtAdd = 125.0))
        assertNull(priceRoseNote(price = 150.0, priceAtAdd = null))
    }
}
