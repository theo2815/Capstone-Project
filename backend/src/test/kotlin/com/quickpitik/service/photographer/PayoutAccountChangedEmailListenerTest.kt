package com.quickpitik.service.photographer

import com.quickpitik.entity.User
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.EmailService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.util.Optional
import java.util.UUID

// Phase 3b — the payout-account-change alert reaches the OWNER's registered email
// (a compromised session can't suppress it, unlike an in-app inbox message).
class PayoutAccountChangedEmailListenerTest {

    @Test
    fun `dispatches the owner alert email on a payout-account change`() {
        val userRepository = Mockito.mock(UserRepository::class.java)
        val emailService = Mockito.mock(EmailService::class.java)
        val userId = UUID.randomUUID()
        val user = Mockito.mock(User::class.java)
        Mockito.`when`(user.email).thenReturn("owner@example.com")
        Mockito.`when`(user.name).thenReturn("Juan Dela Cruz")
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))

        PayoutAccountChangedEmailListener(userRepository, emailService)
            .onPayoutAccountChanged(
                PayoutAccountChangedEvent(userId = userId, action = "create", method = "gcash", maskedAccount = "••••4567"),
            )

        Mockito.verify(emailService).sendPayoutAccountChangedAlert(
            "owner@example.com", "Juan Dela Cruz", "create", "gcash", "••••4567",
        )
    }

    @Test
    fun `no email when the user no longer exists`() {
        val userRepository = Mockito.mock(UserRepository::class.java)
        val emailService = Mockito.mock(EmailService::class.java)
        val userId = UUID.randomUUID()
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.empty())

        PayoutAccountChangedEmailListener(userRepository, emailService)
            .onPayoutAccountChanged(
                PayoutAccountChangedEvent(userId = userId, action = "delete", method = "gcash", maskedAccount = "••••4567"),
            )

        Mockito.verifyNoInteractions(emailService)
    }
}
