package com.quickpitik.service.reference

import com.quickpitik.config.RaceProperties
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class RaceTypesServiceTest {

    @Test
    fun `lists six shared codes plus OTHER and the switch`() {
        val off = RaceTypesService(RaceProperties()).list()
        assertFalse(off.selfServeEnabled)
        assertEquals(
            listOf("ROAD_RACE", "FUN_RUN", "TRAIL_RUN", "ULTRA", "OBSTACLE_RACE", "CYCLING", "OTHER"),
            off.types.map { it.code },
        )
        assertEquals(List(6) { true } + false, off.types.map { it.shared })
        assertTrue(off.types.all { it.label.isNotBlank() && it.hint.isNotBlank() })

        assertTrue(RaceTypesService(RaceProperties(selfServeEnabled = true)).list().selfServeEnabled)
    }
}
