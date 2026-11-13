package com.quickpitik.service.admin

import com.quickpitik.entity.User
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.util.Optional
import java.util.UUID

// PE-2.2: the hold e-mail goes to the photographer's own address once the hold has committed.
class PayoutHoldEmailListenerTest {

    private val userRepository = Mockito.mock(UserRepository::class.java)
    private val emailService = Mockito.mock(EmailService::class.java)
    private val photographerId = UUID.randomUUID()

    @Test
    fun `the hold alert goes to the photographer's address after commit`() {
        val user = Mockito.mock(User::class.java)
        Mockito.`when`(user.email).thenReturn("juan@example.com")
        Mockito.`when`(user.name).thenReturn("Juan Dela Cruz")
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(user))

        PayoutHoldEmailListener(userRepository, emailService).onPayoutHoldPlaced(PayoutHoldPlacedEvent(photographerId))

        Mockito.verify(emailService).sendPayoutHoldAlert("juan@example.com", "Juan Dela Cruz")
    }

    @Test
    fun `a missing user is logged and skipped`() {
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.empty())

        PayoutHoldEmailListener(userRepository, emailService).onPayoutHoldPlaced(PayoutHoldPlacedEvent(photographerId))

        Mockito.verifyNoInteractions(emailService)
    }
}
