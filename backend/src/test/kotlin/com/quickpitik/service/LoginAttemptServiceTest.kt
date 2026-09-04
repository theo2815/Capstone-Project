package com.quickpitik.service

import com.quickpitik.config.AuthLockoutProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.Duration
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// The counter arithmetic and the lock window. What lives one layer up in
// AuthServiceTest is *when* these get called; what lives in
// AuthLockoutIntegrationTest is that the write survives a rollback.
class LoginAttemptServiceTest {

    private lateinit var userRepository: UserRepository
    private val userId: UUID = UUID.randomUUID()
    private val properties = AuthLockoutProperties(maxAttempts = 5, duration = Duration.ofMinutes(15))

    @BeforeEach
    fun setUp() {
        userRepository = Mockito.mock(UserRepository::class.java)
    }

    @Test
    fun `a failure short of the threshold only increments`() {
        val user = stubUser(attempts = 2)

        service().recordFailure(userId)

        assertEquals(3, user.failedLoginAttempts)
        assertNull(user.lockedUntil)
    }

    @Test
    fun `the fifth consecutive failure locks the account`() {
        val user = stubUser(attempts = 4)

        service().recordFailure(userId)

        assertNotNull(user.lockedUntil)
        // Fifteen minutes out, give or take the clock tick between the two calls.
        val window = Duration.between(OffsetDateTime.now(), user.lockedUntil)
        assert(window > Duration.ofMinutes(14)) { "lock window too short: $window" }
        assert(window <= Duration.ofMinutes(15)) { "lock window too long: $window" }
    }

    // If the counter stayed at the threshold, the first mistake after a lock
    // expired would immediately re-lock — a 15-minute lock would become a
    // permanent one for anyone who mistypes twice in a row.
    @Test
    fun `applying the lock resets the counter`() {
        val user = stubUser(attempts = 4)

        service().recordFailure(userId)

        assertEquals(0, user.failedLoginAttempts)
    }

    // NFR-S-14 window (V34): failures only accumulate within 15 minutes of the
    // previous one. Five typos spread over a month must never lock an account.
    @Test
    fun `a failure after a stale streak restarts the counter at 1`() {
        val user = stubUser(attempts = 4).apply {
            lastFailedLoginAt = OffsetDateTime.now().minusMinutes(16)
        }

        service().recordFailure(userId)

        assertEquals(1, user.failedLoginAttempts)
        assertNull(user.lockedUntil)
        // The streak clock restarts with this failure.
        assertNotNull(user.lastFailedLoginAt)
        assert(user.lastFailedLoginAt!! > OffsetDateTime.now().minusMinutes(1))
    }

    @Test
    fun `a failure within the window extends the streak and restamps it`() {
        val user = stubUser(attempts = 2).apply {
            lastFailedLoginAt = OffsetDateTime.now().minusMinutes(1)
        }

        service().recordFailure(userId)

        assertEquals(3, user.failedLoginAttempts)
        assert(user.lastFailedLoginAt!! > OffsetDateTime.now().minusSeconds(30))
    }

    @Test
    fun `success clears the streak timestamp too`() {
        val user = stubUser(attempts = 0).apply {
            lastFailedLoginAt = OffsetDateTime.now().minusMinutes(20)
        }

        service().recordSuccess(userId)

        assertNull(user.lastFailedLoginAt)
    }

    @Test
    fun `success clears both the counter and the lock`() {
        val user = stubUser(attempts = 3, lockedUntil = OffsetDateTime.now().plusMinutes(5))

        service().recordSuccess(userId)

        assertEquals(0, user.failedLoginAttempts)
        assertNull(user.lockedUntil)
    }

    // A clean login on a clean account shouldn't cost an UPDATE — every
    // sign-in in the system takes this path.
    @Test
    fun `success on an untouched account writes nothing`() {
        stubUser(attempts = 0)

        service().recordSuccess(userId)

        Mockito.verify(userRepository, Mockito.never()).save(anyArg())
    }

    @Test
    fun `lockRemaining reports the time left while the lock holds`() {
        val user = newUser().apply { lockedUntil = OffsetDateTime.now().plusMinutes(10) }

        val remaining = service().lockRemaining(user)

        assertNotNull(remaining)
        assert(remaining > Duration.ofMinutes(9)) { "expected ~10 minutes, got $remaining" }
    }

    // Expired locks are read as "unlocked" rather than swept: a stale timestamp
    // on a row nobody is authenticating against costs nothing, and the next
    // successful login clears it anyway.
    @Test
    fun `an elapsed lock reads as unlocked`() {
        val user = newUser().apply { lockedUntil = OffsetDateTime.now().minusSeconds(1) }

        assertNull(service().lockRemaining(user))
    }

    @Test
    fun `an account that never failed is not locked`() {
        assertNull(service().lockRemaining(newUser()))
    }

    // A user deleted between the failed login and this out-of-band write is a
    // real interleaving, not a hypothetical — it must not surface as a 500 on
    // top of the credential error the caller is already getting.
    @Test
    fun `a vanished user is ignored rather than thrown`() {
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.empty())

        service().recordFailure(userId)
        service().recordSuccess(userId)

        Mockito.verify(userRepository, Mockito.never()).save(anyArg())
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun service() = LoginAttemptService(userRepository, properties)

    private fun stubUser(attempts: Int, lockedUntil: OffsetDateTime? = null): User {
        val user = newUser().apply {
            failedLoginAttempts = attempts
            this.lockedUntil = lockedUntil
        }
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
        return user
    }

    private fun newUser(): User = User(
        id = userId,
        email = "runner@example.com",
        passwordHash = "\$2a\$12\$stub",
        name = "Test Runner",
        role = Role.RUNNER,
    )

    private fun <T> anyArg(): T = Mockito.any()
}
