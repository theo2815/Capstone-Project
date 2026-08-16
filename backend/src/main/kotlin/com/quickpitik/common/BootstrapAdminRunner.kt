package com.quickpitik.common

import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.UserRepository
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.boot.ApplicationArguments
import org.springframework.boot.ApplicationRunner
import org.springframework.security.crypto.password.PasswordEncoder
import org.springframework.stereotype.Component
import java.time.OffsetDateTime

@Component
class BootstrapAdminRunner(
    private val userRepository: UserRepository,
    private val passwordEncoder: PasswordEncoder,
    @Value("\${app.admin.bootstrap-email:}") private val bootstrapEmail: String,
    @Value("\${app.admin.bootstrap-password:}") private val bootstrapPassword: String,
    @Value("\${app.admin.bootstrap-name:QuickPitik Admin}") private val bootstrapName: String,
) : ApplicationRunner {
    private val log = LoggerFactory.getLogger(javaClass)

    override fun run(args: ApplicationArguments?) {
        if (bootstrapEmail.isBlank() || bootstrapPassword.isBlank()) {
            log.info("Admin bootstrap skipped — ADMIN_BOOTSTRAP_EMAIL or _PASSWORD is empty")
            return
        }
        if (userRepository.existsByRole(Role.ADMIN)) {
            log.info("Admin bootstrap skipped — at least one ADMIN already exists")
            return
        }
        val admin = User(
            email = bootstrapEmail.trim().lowercase(),
            passwordHash = passwordEncoder.encode(bootstrapPassword),
            name = bootstrapName,
            role = Role.ADMIN,
            // Provisioned from env by whoever runs the service, not
            // self-registered — so there is nobody to send a link to, and the
            // default `admin@quickpitik.local` isn't a deliverable inbox at
            // all. Without this the admin would read as permanently
            // unverified, which is the wrong signal for the one account the
            // operator chose deliberately.
            emailVerifiedAt = OffsetDateTime.now(),
        )
        userRepository.save(admin)
        log.info("Bootstrap admin created: {}", admin.email)
    }
}
