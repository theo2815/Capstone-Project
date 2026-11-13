package com.quickpitik.service.admin

import com.quickpitik.config.IdentityProperties
import com.quickpitik.dto.admin.IdentityNoticeResultDto
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.email.EmailService
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import java.util.UUID

// AA-I22 (C16): emails every photographer who has not passed identity about the gate date
// (IDENTITY_GATE_STARTS_ON). No transaction: it calls Resend, one mail at a time, paced.
// Send once per start date: each recipient is claimed (committed) before the mail, and un-claimed if the mail fails.
@Service
class IdentityNoticeService(
    private val userRepository: UserRepository,
    private val emailService: EmailService,
    private val identityProperties: IdentityProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun send(adminId: UUID, dryRun: Boolean): IdentityNoticeResultDto {
        val startsOn = identityProperties.gateStartsOn
            ?: throw ValidationException("Set IDENTITY_GATE_STARTS_ON before sending the identity notice.")
        val recipients = userRepository.findIdentityNoticeRecipients(PASSED, startsOn)
        if (dryRun) return IdentityNoticeResultDto(recipients.size, 0)

        val gapMs = 1000L / identityProperties.noticeSendPerSecond
        var sent = 0
        recipients.forEachIndexed { i, user ->
            if (i > 0) Thread.sleep(gapMs)
            if (userRepository.claimIdentityNotice(user.id, startsOn) == 0) return@forEachIndexed
            if (emailService.sendIdentityCheckNotice(user.email, user.name, startsOn)) sent++
            else userRepository.clearIdentityNotice(user.id, startsOn)
        }
        log.info("Identity notice run · admin={} startsOn={} recipients={} sent={}", adminId, startsOn, recipients.size, sent)
        return IdentityNoticeResultDto(recipients.size, sent)
    }

    private companion object {
        val PASSED = SellerDetailsStatus.entries.filter { it.passesIdentity() }
    }
}
