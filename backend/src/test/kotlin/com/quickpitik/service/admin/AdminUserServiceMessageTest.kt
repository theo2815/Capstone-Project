package com.quickpitik.service.admin

import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.auth.RefreshTokenService
import com.quickpitik.service.photographer.PhotographerSettingsService
import com.quickpitik.service.profile.UserDtoMapper
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.SimpleTransactionStatus
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class AdminUserServiceMessageTest {
    private val userRepository = Mockito.mock(UserRepository::class.java)
    private val adminDecisionLogService = Mockito.mock(AdminDecisionLogService::class.java)
    private val runnerMessagesService = Mockito.mock(RunnerMessagesService::class.java)
    private val photographerMessages = Mockito.mock(PhotographerMessageRepository::class.java)
    private val runnerMessages = Mockito.mock(RunnerMessageRepository::class.java)
    private val adminId = UUID.randomUUID()
    private val photographerId = UUID.randomUUID()

    private fun service(): AdminUserService {
        val manager = Mockito.mock(PlatformTransactionManager::class.java)
        Mockito.`when`(manager.getTransaction(Mockito.any())).thenAnswer { SimpleTransactionStatus() }
        val commands = AdminActionCommandService(Mockito.mock(AdminActionCommandRepository::class.java), manager,
            com.fasterxml.jackson.databind.ObjectMapper(), java.time.Clock.systemUTC())
        return AdminUserService(userRepository, Mockito.mock(PhotographerSettingsRepository::class.java),
            Mockito.mock(SocialLinkRepository::class.java), Mockito.mock(PayoutAccountRepository::class.java),
            Mockito.mock(AdminDecisionLogRepository::class.java), Mockito.mock(RegionsService::class.java),
            adminDecisionLogService, Mockito.mock(StorageService::class.java), StorageProperties(),
            Mockito.mock(UserDtoMapper::class.java), runnerMessagesService, Mockito.mock(PhotographerSettingsService::class.java),
            Mockito.mock(RefreshTokenService::class.java), commands, photographerMessages, runnerMessages)
    }

    @Test
    fun `message text is trimmed and routed to photographer`() {
        val target = User(id = photographerId, email = "p@test.local", passwordHash = "unused", name = "P", role = Role.PHOTOGRAPHER)
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(target))
        val actor = User(id = adminId, email = "a@test.local", passwordHash = "unused", name = "A", role = Role.ADMIN)
        Mockito.`when`(userRepository.findById(adminId)).thenReturn(Optional.of(actor))
        val decision = AdminDecisionLog(adminId = adminId, targetUserId = photographerId, decision = "messaged", reason = "Hello")
        Mockito.`when`(adminDecisionLogService.logUserDecision(adminId, photographerId, "messaged", "Hello")).thenReturn(decision)
        val message = PhotographerMessage(photographerId = photographerId, kindWire = "admin_message", title = "Hello", body = "Message", sourceDecisionId = decision.id)
        Mockito.`when`(adminDecisionLogService.pushMessage(photographerId, PhotographerMessageKind.ADMIN_MESSAGE, "Message", adminId, decision.id, "Hello")).thenReturn(message)
        Mockito.`when`(photographerMessages.findByIdAndPhotographerId(message.id, photographerId)).thenReturn(Optional.of(message))
        val result = service().sendMessage(adminId, photographerId, " Hello ", " Message ")
        assertEquals(message.id, result.id); assertEquals(decision.id, result.sourceDecisionId)
        assertEquals("Hello", result.title); assertEquals("Message", result.body)
        Mockito.verifyNoInteractions(runnerMessagesService)
    }

    @Test
    fun `message text is trimmed and routed to runner`() {
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(User(id = photographerId,
            email = "r@test.local", passwordHash = "unused", name = "R", role = Role.RUNNER)))
        Mockito.`when`(userRepository.findById(adminId)).thenReturn(Optional.of(User(id = adminId,
            email = "a@test.local", passwordHash = "unused", name = "A", role = Role.ADMIN)))
        val decision = AdminDecisionLog(adminId = adminId, targetUserId = photographerId, decision = "messaged", reason = "Hello")
        Mockito.`when`(adminDecisionLogService.logUserDecision(adminId, photographerId, "messaged", "Hello")).thenReturn(decision)
        val message = RunnerMessage(runnerId = photographerId, kindWire = "admin_message", title = "Hello", body = "Message", sourceDecisionId = decision.id)
        Mockito.`when`(runnerMessagesService.pushMessage(photographerId, RunnerMessageKind.ADMIN_MESSAGE, "Message", adminId, decision.id, title = "Hello")).thenReturn(message)
        Mockito.`when`(runnerMessages.findByIdAndRunnerId(message.id, photographerId)).thenReturn(Optional.of(message))
        val result = service().sendMessage(adminId, photographerId, " Hello ", " Message ")
        assertEquals(message.id, result.id); assertEquals(decision.id, result.sourceDecisionId)
        assertEquals("Hello", result.title); assertEquals("Message", result.body)
        Mockito.verify(adminDecisionLogService, Mockito.never()).pushMessage(photographerId, PhotographerMessageKind.ADMIN_MESSAGE, "Message", adminId, decision.id, "Hello")
    }

    @Test
    fun `nonadmin sender is refused before sending`() {
        Mockito.`when`(userRepository.findById(adminId)).thenReturn(Optional.of(User(id = adminId,
            email = "r@test.local", passwordHash = "unused", name = "R", role = Role.RUNNER)))
        Mockito.`when`(userRepository.findById(photographerId)).thenReturn(Optional.of(User(id = photographerId,
            email = "p@test.local", passwordHash = "unused", name = "P", role = Role.PHOTOGRAPHER)))
        val ex = assertFailsWith<ApiException> { service().sendMessage(adminId, photographerId, "Hello", "Message") }
        assertEquals("FORBIDDEN", ex.code)
        Mockito.verifyNoInteractions(adminDecisionLogService, runnerMessagesService)
    }
}
