package com.quickpitik.support

import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.service.photos.PhotoErasureSweep
import com.quickpitik.service.removal.*
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.anyString
import org.mockito.Mockito.doReturn
import org.mockito.Mockito.reset
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.context.bean.override.mockito.MockitoBean
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import kotlin.test.*

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = ["app.photo-removal.cleanup-interval=PT24H", "app.photo-removal.in-request-cleanup-jobs=1"])
class PhotoRemovalCleanupBudgetIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var service: PhotoRemovalService
    @Autowired lateinit var queries: PhotoRemovalQueryService
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var erasureSweep: PhotoErasureSweep
    @Autowired lateinit var clock: java.time.Clock
    @MockitoBean lateinit var storage: StorageService

    @Test fun `decision attempts at most the configured jobs in-request and the sweep finishes the rest`() {
        val admin = users.saveAndFlush(User(email = "budget-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Synthetic ADMIN", role = Role.ADMIN))
        val owner = users.saveAndFlush(User(email = "budget-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Synthetic PHOTOGRAPHER", role = Role.PHOTOGRAPHER))
        val event = events.saveAndFlush(Event(slug = "budget-${UUID.randomUUID()}", name = "Synthetic run", date = LocalDate.of(2026, 10, 10), location = "Cebu", status = EventStatus.ACTIVE))
        val selected = (1..3).map { photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal("50"))) }
        val case = service.create(admin.id, CreateRemovalRequestCommand(event.id, "support-${UUID.randomUUID()}", "private contact", reason = "private reason", commandId = UUID.randomUUID()))
        service.verify(admin.id, case.id, VerifyRemovalCommand(UUID.randomUUID(), case.version, "Support identity check", "Private verification",
            selected.map { VerifyRemovalSelection(it.id, RemovalMatchSource.MANUAL, "Visually confirmed") }))
        var detail = queries.adminDetail(admin.id, case.id)
        service.respond(owner.id, case.id, RespondToRemovalCommand(UUID.randomUUID(), detail.version, detail.items.map { it.id }, PhotographerRemovalResponse.AGREE))
        detail = queries.adminDetail(admin.id, case.id)
        service.decide(admin.id, case.id, DecideRemovalCommand(UUID.randomUUID(), detail.version,
            detail.items.map { RemovalItemDecision(it.id, RemovalDecision.REMOVE, "Runner verified") }))
        val afterDecision = queries.adminDetail(admin.id, case.id)
        assertEquals(1, afterDecision.items.count { it.state == RemovalItemState.REMOVED })
        assertEquals(2, afterDecision.items.count { it.state == RemovalItemState.REMOVAL_AUTHORIZED })
        assertTrue(erasureSweep.sweepDue(clock.instant(), 50) >= 2)
        service.reconcile(case.id)
        assertTrue(queries.adminDetail(admin.id, case.id).items.all { it.state == RemovalItemState.REMOVED })
    }

    @Test fun `retry makes every selected job due so the sweep finishes what the request skipped`() {
        val admin = users.saveAndFlush(User(email = "budget-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Synthetic ADMIN", role = Role.ADMIN))
        val owner = users.saveAndFlush(User(email = "budget-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Synthetic PHOTOGRAPHER", role = Role.PHOTOGRAPHER))
        val event = events.saveAndFlush(Event(slug = "budget-${UUID.randomUUID()}", name = "Synthetic run", date = LocalDate.of(2026, 10, 10), location = "Cebu", status = EventStatus.ACTIVE))
        val selected = (1..3).map { photos.saveAndFlush(Photo(eventId = event.id, photographerId = owner.id, s3Key = "synthetic/${UUID.randomUUID()}", pricePhp = BigDecimal("50"))) }
        val case = service.create(admin.id, CreateRemovalRequestCommand(event.id, "support-${UUID.randomUUID()}", "private contact", reason = "private reason", commandId = UUID.randomUUID()))
        service.verify(admin.id, case.id, VerifyRemovalCommand(UUID.randomUUID(), case.version, "Support identity check", "Private verification",
            selected.map { VerifyRemovalSelection(it.id, RemovalMatchSource.MANUAL, "Visually confirmed") }))
        var detail = queries.adminDetail(admin.id, case.id)
        service.respond(owner.id, case.id, RespondToRemovalCommand(UUID.randomUUID(), detail.version, detail.items.map { it.id }, PhotographerRemovalResponse.AGREE))
        detail = queries.adminDetail(admin.id, case.id)
        doReturn(true).`when`(storage).exists(anyString()) // every delete looks unconfirmed: jobs back off
        service.decide(admin.id, case.id, DecideRemovalCommand(UUID.randomUUID(), detail.version,
            detail.items.map { RemovalItemDecision(it.id, RemovalDecision.REMOVE, "Runner verified") }))
        while (erasureSweep.sweepDue(clock.instant(), 50) > 0) {}
        reset(storage)
        detail = queries.adminDetail(admin.id, case.id)
        service.retryCleanup(admin.id, case.id, RetryRemovalCleanupCommand(UUID.randomUUID(), detail.version, detail.items.map { it.id }))
        while (erasureSweep.sweepDue(clock.instant(), 50) > 0) {}
        service.reconcile(case.id)
        assertTrue(queries.adminDetail(admin.id, case.id).items.all { it.state == RemovalItemState.REMOVED })
    }
}
