package com.quickpitik.service.earnings

import com.quickpitik.common.ErrorCodes
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.PayoutStatusHistory
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.PayoutStatusHistoryRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.http.HttpStatus
import org.springframework.transaction.annotation.Propagation
import org.springframework.transaction.annotation.Transactional
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

// One table for the payout lifecycle. Before this every call site spelled its
// own allowed-from set, and two had drifted: hold() refused a HELD cycle while
// bulkHold() accepted one and wrote a HELD -> HELD row into the audit trail
// (audit 2026-09-11, items 1 and 3).
class PayoutCycleTransitionsTest {

    private lateinit var payoutCycleRepository: PayoutCycleRepository
    private lateinit var payoutStatusHistoryRepository: PayoutStatusHistoryRepository
    private lateinit var transitions: PayoutCycleTransitions

    private val adminId = UUID.randomUUID()

    @BeforeEach
    fun setUp() {
        payoutCycleRepository = Mockito.mock(PayoutCycleRepository::class.java)
        payoutStatusHistoryRepository = Mockito.mock(PayoutStatusHistoryRepository::class.java)
        transitions = PayoutCycleTransitions(
            payoutCycleRepository,
            PayoutStatusRecorder(payoutStatusHistoryRepository),
        )
    }

    private fun cycle(status: PayoutCycleStatus): PayoutCycle {
        val cycle = PayoutCycle(
            id = "pc_test_1",
            photographerId = UUID.randomUUID(),
            weekOf = LocalDate.of(2026, 9, 7),
            methodWire = "gcash",
        )
        cycle.status = status
        return cycle
    }

    @Test
    fun `an allowed transition moves the status, saves it and records one history row`() {
        val cycle = cycle(PayoutCycleStatus.APPROVED)

        transitions.apply(cycle, PayoutTransition.AdminMarkPaid, adminId)

        assertEquals(PayoutCycleStatus.PAID, cycle.status)
        Mockito.verify(payoutCycleRepository).save(cycle)
        val captor = ArgumentCaptor.forClass(PayoutStatusHistory::class.java)
        Mockito.verify(payoutStatusHistoryRepository).save(captor.capture())
        assertEquals("approved", captor.value.fromStatus)
        assertEquals("paid", captor.value.toStatus)
        assertEquals("admin", captor.value.actorType)
    }

    @Test
    fun `a disallowed transition is a 409 that changes nothing and writes nothing`() {
        val cycle = cycle(PayoutCycleStatus.PENDING)

        val ex = assertFailsWith<ApiException> {
            transitions.apply(cycle, PayoutTransition.AdminMarkPaid, adminId)
        }

        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertEquals(PayoutCycleStatus.PENDING, cycle.status)
        Mockito.verifyNoInteractions(payoutCycleRepository)
        Mockito.verifyNoInteractions(payoutStatusHistoryRepository)
    }

    @Test
    fun `re-applying the state a cycle already holds is a 409 that writes nothing`() {
        val cycle = cycle(PayoutCycleStatus.HELD)

        val ex = assertFailsWith<ApiException> {
            transitions.apply(cycle, PayoutTransition.AdminHold, adminId)
        }

        assertEquals(HttpStatus.CONFLICT, ex.status)
        assertEquals(ErrorCodes.INVALID_STATE_TRANSITION, ex.code)
        assertEquals(PayoutCycleStatus.HELD, cycle.status)
        Mockito.verifyNoInteractions(payoutCycleRepository)
        Mockito.verifyNoInteractions(payoutStatusHistoryRepository)
    }

    @Test
    fun `admin and the provider do not share authority over the same target state`() {
        // Why the table is keyed by named transition and not by target status: a
        // target-keyed table would let the webhook pay a cycle an admin has only
        // approved, and let an admin return one the provider is still processing.
        assertTrue(PayoutCycleStatus.APPROVED in PayoutTransition.AdminMarkPaid.allowedFrom)
        assertFalse(PayoutCycleStatus.APPROVED in PayoutTransition.ProviderPaid.allowedFrom)
        assertTrue(PayoutCycleStatus.PROCESSING in PayoutTransition.ProviderReturned.allowedFrom)
        assertFalse(PayoutCycleStatus.PROCESSING in PayoutTransition.AdminMarkReturned.allowedFrom)
    }

    @Test
    fun `the audit recorder refuses to run outside a caller transaction`() {
        val tx = PayoutStatusRecorder::class.java.getMethod(
            "record", String::class.java, PayoutCycleStatus::class.java, PayoutCycleStatus::class.java,
            String::class.java, UUID::class.java, String::class.java,
        ).getAnnotation(Transactional::class.java)
        assertEquals(Propagation.MANDATORY, tx?.propagation)
    }

    @Test
    fun `single hold and bulk hold cannot drift again because they are one transition`() {
        assertEquals(
            setOf(PayoutCycleStatus.PENDING, PayoutCycleStatus.APPROVED, PayoutCycleStatus.PROCESSING),
            PayoutTransition.AdminHold.allowedFrom,
        )
    }
}
