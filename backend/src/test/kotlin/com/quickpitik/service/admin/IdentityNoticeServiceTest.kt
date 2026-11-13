package com.quickpitik.service.admin

import com.quickpitik.config.IdentityProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.User
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import org.mockito.Mockito
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals

class IdentityNoticeServiceTest {
    private val users = Mockito.mock(UserRepository::class.java)
    private val email = Mockito.mock(EmailService::class.java)
    private val startsOn = LocalDate.of(2026, 12, 7)
    private val admin = UUID.randomUUID()
    private val passed = listOf(SellerDetailsStatus.AUTO_APPROVED, SellerDetailsStatus.APPROVED)

    private fun service(startsOn: LocalDate? = this.startsOn) =
        IdentityNoticeService(users, email, IdentityProperties(gateStartsOn = startsOn, noticeSendPerSecond = 1000))

    private fun photographer(name: String) =
        User(email = "$name@test.local", passwordHash = "x", name = name, role = Role.PHOTOGRAPHER)

    @Test
    fun `refuses with no start date`() {
        assertThrows<ValidationException> { service(startsOn = null).send(admin, dryRun = true) }
        Mockito.verifyNoInteractions(users, email)
    }

    @Test
    fun `dry run sends nothing and counts`() {
        Mockito.`when`(users.findIdentityNoticeRecipients(passed)).thenReturn(listOf(photographer("a"), photographer("b")))

        val result = service().send(admin, dryRun = true)

        assertEquals(2, result.recipients)
        assertEquals(0, result.sent)
        Mockito.verifyNoInteractions(email)
    }

    @Test
    fun `a real run mails each recipient once and counts only the mails Resend took`() {
        val a = photographer("a")
        val b = photographer("b")
        Mockito.`when`(users.findIdentityNoticeRecipients(passed)).thenReturn(listOf(a, b))
        Mockito.`when`(email.sendIdentityCheckNotice(a.email, "a", startsOn)).thenReturn(true)
        Mockito.`when`(email.sendIdentityCheckNotice(b.email, "b", startsOn)).thenReturn(false)

        val result = service().send(admin, dryRun = false)

        assertEquals(2, result.recipients)
        assertEquals(1, result.sent)
        Mockito.verify(email).sendIdentityCheckNotice(a.email, "a", startsOn)
        Mockito.verify(email).sendIdentityCheckNotice(b.email, "b", startsOn)
    }
}
