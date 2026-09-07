package com.quickpitik.support

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.auth.RegisterRequest
import com.quickpitik.entity.PayoutAccountAudit
import com.quickpitik.entity.PayoutCycle
import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.Role
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.PayoutAccountAuditRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.AuthService
import com.quickpitik.service.earnings.PayoutRequestService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataIntegrityViolationException
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The financial-integrity guarantees that only a real database can prove:
 *
 *  1. `uq_payout_open_per_photographer` (V53) makes a second OPEN payout cycle
 *     impossible — the DB, not app code, is what closes the double-request race.
 *     Paired with PayoutRequestServiceBalanceTest's "duplicate-open -> 409"
 *     unit test (which proves request() maps the violation), this covers the
 *     race end-to-end without seeding a full event->photo->order->transaction
 *     balance chain: the index rejects the duplicate regardless of how it arrives.
 *  2. The post-account-change cooldown (V54, Phase 3a) actually blocks request()
 *     against real rows.
 */
class PayoutLifecycleIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var authService: AuthService
    @Autowired private lateinit var userRepository: UserRepository
    @Autowired private lateinit var payoutCycleRepository: PayoutCycleRepository
    @Autowired private lateinit var payoutAccountAuditRepository: PayoutAccountAuditRepository
    @Autowired private lateinit var payoutRequestService: PayoutRequestService

    private fun newPhotographer(): UUID {
        val email = "payout-${UUID.randomUUID()}@test.local"
        authService.register(
            RegisterRequest(name = "P", email = email, password = "correct-horse-battery-staple-42", role = Role.PHOTOGRAPHER),
        )
        return requireNotNull(userRepository.findByEmail(email)).id
    }

    private fun cycle(pid: UUID, status: PayoutCycleStatus): PayoutCycle =
        PayoutCycle(
            id = "PAY-IT-${UUID.randomUUID().toString().take(12)}",
            photographerId = pid,
            weekOf = LocalDate.now(),
            methodWire = "gcash",
        ).apply {
            this.status = status
            amountPhp = BigDecimal("500")
        }

    @Test
    fun `only one open cycle per photographer is allowed by the DB`() {
        val pid = newPhotographer()
        payoutCycleRepository.saveAndFlush(cycle(pid, PayoutCycleStatus.PENDING))

        // A second OPEN cycle for the same photographer violates the partial
        // unique index — this is what makes the concurrent-request race impossible.
        assertFailsWith<DataIntegrityViolationException> {
            payoutCycleRepository.saveAndFlush(cycle(pid, PayoutCycleStatus.PENDING))
        }
    }

    @Test
    fun `a terminal cycle does not block a new open cycle`() {
        val pid = newPhotographer()
        payoutCycleRepository.saveAndFlush(cycle(pid, PayoutCycleStatus.PAID))
        payoutCycleRepository.saveAndFlush(cycle(pid, PayoutCycleStatus.CANCELLED))

        // Neither PAID nor CANCELLED is "open", so a fresh open cycle is fine.
        payoutCycleRepository.saveAndFlush(cycle(pid, PayoutCycleStatus.PENDING))
    }

    @Test
    fun `request is blocked during the payout-account cooldown`() {
        val pid = newPhotographer()
        payoutAccountAuditRepository.saveAndFlush(
            PayoutAccountAudit(userId = pid, action = "update", createdAt = OffsetDateTime.now()),
        )

        val ex = assertFailsWith<ApiException> { payoutRequestService.request(pid) }
        assertEquals(ErrorCodes.PAYOUT_ACCOUNT_COOLDOWN, ex.code)
    }

    @Test
    fun `an open cycle after a cancelled one is permitted (soft-cancel frees the slot)`() {
        val pid = newPhotographer()
        // Simulate request -> hold -> withdraw(cancel): the cancelled row stays
        // (audit), and a new open cycle is allowed.
        payoutCycleRepository.saveAndFlush(cycle(pid, PayoutCycleStatus.CANCELLED))
        val open = payoutCycleRepository.saveAndFlush(cycle(pid, PayoutCycleStatus.PENDING))
        assertTrue(payoutCycleRepository.findById(open.id).isPresent)
    }
}
