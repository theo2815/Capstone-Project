package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ResponseEnvelopeAdvice
import com.quickpitik.controller.AdminEventsController
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.security.AuthPrincipal
import org.springframework.http.HttpMethod
import org.springframework.http.MediaType
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import com.quickpitik.dto.admin.*
import com.quickpitik.dto.photographer.UpdateMyEventRequest
import com.quickpitik.entity.*
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.*
import com.quickpitik.service.admin.*
import com.quickpitik.service.photographer.PhotographerOwnedEventService
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import java.math.BigDecimal
import java.time.LocalDate
import java.util.UUID
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.sql.DataSource
import kotlin.test.*

class AdminTransitionConcurrencyIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var json: ObjectMapper
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var flags: FlagRepository
    @Autowired lateinit var reports: PayoutReportRepository
    @Autowired lateinit var disputes: DisputeRepository
    @Autowired lateinit var orders: OrderRepository
    @Autowired lateinit var cycles: PayoutCycleRepository
    @Autowired lateinit var flagAdmin: AdminFlagService
    @Autowired lateinit var reportAdmin: AdminPayoutReportService
    @Autowired lateinit var disputeAdmin: AdminDisputeService
    @Autowired lateinit var eventAdmin: AdminEventService
    @Autowired lateinit var photoOwner: com.quickpitik.service.photographer.PhotographerPhotoService
    @Autowired lateinit var refunds: com.quickpitik.service.orders.RefundService
    @Autowired lateinit var ownerEvents: PhotographerOwnedEventService
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var dataSource: DataSource
    @Autowired lateinit var txManager: PlatformTransactionManager

    private fun user(role: Role) = users.saveAndFlush(User(email = "transition-${UUID.randomUUID()}@test.local",
        passwordHash = "test-only", name = "Transition fixture", role = role))

    private fun event(review: EventReviewStatus = EventReviewStatus.APPROVED): Event = events.saveAndFlush(Event(
        slug = "transition-${UUID.randomUUID()}", name = "Run", date = LocalDate.of(2026, 10, 10), location = "Cebu",
        status = if (review == EventReviewStatus.PENDING) EventStatus.DRAFT else EventStatus.ACTIVE,
        reviewStatus = review, createdBy = user(Role.PHOTOGRAPHER).id, pricePerPhoto = BigDecimal("150.00")))

    private fun photo(watermark: String? = "test/watermark.jpg") = photos.saveAndFlush(Photo(eventId = event().id,
        photographerId = user(Role.PHOTOGRAPHER).id, s3Key = "test/original.jpg", pricePhp = BigDecimal("150.00"),
        status = PhotoStatus.LIVE, watermarkS3Key = watermark))

    private fun flag(p: Photo, status: FlagStatus = FlagStatus.OPEN) = flags.saveAndFlush(Flag(
        targetKindWire = "photo", targetId = p.id, reason = "other", statusWire = status.wire))

    private fun report(): PayoutReport {
        val p = user(Role.PHOTOGRAPHER).id
        val cycle = cycles.saveAndFlush(PayoutCycle(id = "PAY-TRANSITION-${UUID.randomUUID()}",
            photographerId = p, weekOf = LocalDate.now(), amountPhp = BigDecimal("1000.00")))
        return reports.saveAndFlush(PayoutReport(payoutId = cycle.id, photographerId = p, reasonWire = "other"))
    }

    private fun count(sql: String, vararg args: Any): Int = jdbc.queryForObject(sql, Int::class.java, *args)!!

    private fun eventEffects(id: UUID, expected: Int) {
        assertEquals(expected, count("SELECT count(*) FROM admin_decision_log WHERE target_event_id = ?", id))
        assertEquals(expected, count("SELECT count(*) FROM photographer_messages m JOIN admin_decision_log d ON d.id = m.source_decision_id WHERE d.target_event_id = ?", id))
    }

    private fun reportEffects(r: PayoutReport, expected: Int) {
        assertEquals(expected, count("SELECT count(*) FROM admin_decision_log WHERE target_payout_id = ?", r.payoutId))
        assertEquals(expected, count("SELECT count(*) FROM photographer_messages WHERE photographer_id = ?", r.photographerId))
    }

    private fun awaitBlocked(n: Int) {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
        while (count("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'") < n) {
            check(System.nanoTime() < deadline) { "Expected $n PostgreSQL lock waits" }
            Thread.sleep(20)
        }
    }

    // The winner pauses at its write; the other transaction must see its committed state after waiting.
    private fun race(table: String, first: () -> Any?, second: () -> Any?): List<Result<Any?>> {
        require(table in setOf("flags", "payout_reports", "disputes", "events"))
        val pool = Executors.newFixedThreadPool(2)
        val barrier = CyclicBarrier(2)
        try {
            dataSource.connection.use { lock ->
                lock.autoCommit = false
                lock.createStatement().use { it.execute("LOCK TABLE $table IN SHARE MODE") }
                val futures = listOf(first, second).map { action -> pool.submit<Result<Any?>> {
                    runCatching { barrier.await(10, TimeUnit.SECONDS); action() }
                } }
                try { awaitBlocked(2) } finally { lock.commit() }
                return futures.map { it.get(25, TimeUnit.SECONDS) }
            }
        } finally { pool.shutdownNow() }
    }

    private fun oneConflict(results: List<Result<Any?>>) {
        assertEquals(1, results.count { it.isSuccess }, results.toString())
        assertEquals(409, assertIs<ApiException>(results.single { it.isFailure }.exceptionOrNull()).status.value())
    }

    @Test fun `duplicate report acknowledge accepts one reply with one audit and inbox row`() {
        val r = report(); val a = user(Role.ADMIN).id
        oneConflict(race("payout_reports",
            { reportAdmin.acknowledge(a, r.id, AcknowledgePayoutReportRequest("First")) },
            { reportAdmin.acknowledge(a, r.id, AcknowledgePayoutReportRequest("Second")) }))
        assertEquals(PayoutReportStatus.ACKNOWLEDGED, reports.findById(r.id).get().status)
        reportEffects(r, 1)
    }

    @Test fun `report resolution cannot be overwritten by a stale acknowledgement`() {
        val r = report(); val a = user(Role.ADMIN).id
        val results = race("payout_reports",
            { reportAdmin.resolve(a, r.id, ResolvePayoutReportRequest("Done")) },
            { reportAdmin.acknowledge(a, r.id, AcknowledgePayoutReportRequest("Received")) })
        assertTrue(results[0].isSuccess, results.toString())
        assertEquals(PayoutReportStatus.RESOLVED, reports.findById(r.id).get().status)
        reportEffects(r, results.count { it.isSuccess })
    }

    @Test fun `duplicate resolution accepts one audit and notice`() {
        val r = report(); val a = user(Role.ADMIN).id
        oneConflict(race("payout_reports",
            { reportAdmin.resolve(a, r.id, ResolvePayoutReportRequest("First")) },
            { reportAdmin.resolve(a, r.id, ResolvePayoutReportRequest("Second")) }))
        reportEffects(r, 1)
    }

    @Test fun `two escalations produce one decision and one notice per inbox`() {
        val p = photo(); val runner = user(Role.RUNNER).id
        val order = orders.saveAndFlush(Order(eventId = p.eventId, recipientEmail = "test@example.test",
            userId = runner, paymentMethodWire = "gcash", totalPhp = BigDecimal("150.00")))
        val d = disputes.saveAndFlush(Dispute(orderId = order.id, photoId = p.id, runnerId = runner, reasonWire = "other"))
        val a = user(Role.ADMIN).id
        oneConflict(race("disputes", { disputeAdmin.escalate(a, d.id, null) }, { disputeAdmin.escalate(a, d.id, null) }))
        assertEquals(DisputeStatus.ESCALATED, disputes.findById(d.id).get().status)
        assertEquals(1, count("SELECT count(*) FROM admin_decision_log WHERE target_dispute_id = ?", d.id))
        assertEquals(1, count("SELECT count(*) FROM photographer_messages WHERE photographer_id = ?", p.photographerId!!))
        assertEquals(1, count("SELECT count(*) FROM runner_messages WHERE runner_id = ?", runner))
    }

    @Test fun `hide versus dismiss sees the locked current flag state`() {
        val p = photo(); val f = flag(p); val a = user(Role.ADMIN).id
        val results = race("flags", { flagAdmin.hide(a, f.id, null) }, { flagAdmin.dismiss(a, f.id, null) })
        assertTrue(results[1].isSuccess, results.toString())
        assertEquals(FlagStatus.DISMISSED, flags.findById(f.id).get().status)
        assertEquals(PhotoStatus.LIVE, photos.findById(p.id).get().status)
    }

    @Test fun `closing two hidden flags restores the photo after the last one`() {
        val p = photo(); jdbc.update("UPDATE photos SET status = 'HIDDEN' WHERE id = ?", p.id)
        val f = flag(p, FlagStatus.HIDDEN); val g = flag(p, FlagStatus.HIDDEN); val a = user(Role.ADMIN).id
        val results = race("flags", { flagAdmin.dismiss(a, f.id, null) }, { flagAdmin.resolve(a, g.id, null) })
        assertTrue(results.all { it.isSuccess }, results.toString())
        assertEquals(PhotoStatus.LIVE, photos.findById(p.id).get().status)
        assertEquals(0, count("SELECT count(*) FROM flags WHERE target_id = ? AND status = 'hidden'", p.id))
    }

    @Test fun `a different hidden flag protects the photo and no watermark restores processing`() {
        val p = photo(null); val f = flag(p); val g = flag(p); val a = user(Role.ADMIN).id
        flagAdmin.hide(a, f.id, null); flagAdmin.hide(a, g.id, null)
        flagAdmin.dismiss(a, f.id, null)
        assertEquals(PhotoStatus.HIDDEN, photos.findById(p.id).get().status)
        flagAdmin.resolve(a, g.id, null)
        assertEquals(PhotoStatus.PROCESSING, photos.findById(p.id).get().status)
    }

    @Test fun `duplicate flag hide accepts only once`() {
        val p = photo(); val f = flag(p); val a = user(Role.ADMIN).id
        oneConflict(race("flags", { flagAdmin.hide(a, f.id, null) }, { flagAdmin.hide(a, f.id, null) }))
        assertEquals(FlagStatus.HIDDEN, flags.findById(f.id).get().status)
        assertEquals(PhotoStatus.HIDDEN, photos.findById(p.id).get().status)
    }

    @Test fun `duplicate event delete performs only one deletion and audit`() {
        val e = event(); val a = user(Role.ADMIN).id
        val results = race("events", { eventAdmin.delete(a, e.id) }, { eventAdmin.delete(a, e.id) })
        assertTrue(results.all { it.isSuccess }, results.toString())
        assertEquals(1, results.count { (it.getOrThrow() as AdminEventDeleteResponseDto).removed })
        assertNotNull(events.findById(e.id).get().deletedAt)
        assertEquals(1, count("SELECT count(*) FROM admin_decision_log WHERE target_event_id = ?", e.id))
    }

    @Test fun `admin text updates preserve both accepted edits and override entries`() {
        val e = event(); val a = user(Role.ADMIN).id
        val results = race("events",
            { eventAdmin.update(a, e.id, UpdateAdminEventRequest(title = "New title")) },
            { eventAdmin.update(a, e.id, UpdateAdminEventRequest(location = "New venue")) })
        assertTrue(results.all { it.isSuccess }, results.toString())
        val saved = events.findById(e.id).get()
        assertEquals("New title", saved.name); assertEquals("New venue", saved.location)
        assertEquals(2, saved.adminOverrides.size)
        assertEquals(2, count("SELECT count(*) FROM admin_decision_log WHERE target_event_id = ?", e.id))
    }

    @Test fun `an owner edit cannot overwrite a concurrent event approval`() {
        val e = event(EventReviewStatus.PENDING); val a = user(Role.ADMIN).id
        val results = race("events", { eventAdmin.approve(a, e.id) },
            { ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(title = "Owner edit"), null) })
        assertTrue(results.all { it.isSuccess }, results.toString())
        val saved = events.findById(e.id).get()
        assertEquals(EventReviewStatus.APPROVED, saved.reviewStatus); assertEquals(EventStatus.ACTIVE, saved.status)
        assertEquals("Owner edit", saved.name)
        eventEffects(e.id, 1)
    }

    @Test fun `admin update versus delete cannot resurrect deletion`() {
        val e = event(); val a = user(Role.ADMIN).id
        val results = race("events", { eventAdmin.delete(a, e.id) },
            { eventAdmin.update(a, e.id, UpdateAdminEventRequest(title = "Edited")) })
        assertTrue(results[0].isSuccess, results.toString())
        assertNotNull(events.findById(e.id).get().deletedAt)
        assertEquals(1, count("SELECT count(*) FROM admin_decision_log WHERE target_event_id = ? AND decision = 'event_deleted'", e.id))
        results[1].exceptionOrNull()?.let { assertIs<com.quickpitik.exception.NotFoundException>(it) }
    }

    @Test fun `a report transaction rollback leaves no decision or inbox row`() {
        val r = report(); val a = user(Role.ADMIN).id
        assertFailsWith<IllegalStateException> { TransactionTemplate(txManager).executeWithoutResult {
            reportAdmin.acknowledge(a, r.id, AcknowledgePayoutReportRequest("Received"))
            error("Rollback")
        } }
        assertEquals(PayoutReportStatus.OPEN, reports.findById(r.id).get().status)
        reportEffects(r, 0)
    }

    private fun reviewRequest(actor: UUID, e: Event, action: String, revision: Long?, reason: String = "Send back"): org.springframework.mock.web.MockHttpServletResponse {
        val principal = AuthPrincipal(actor, "admin@test.local", Role.ADMIN)
        SecurityContextHolder.getContext().authentication = UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
        try {
            val mvc = MockMvcBuilders.standaloneSetup(AdminEventsController(eventAdmin))
                .setControllerAdvice(GlobalExceptionHandler(), ResponseEnvelopeAdvice())
                .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver()).build()
            val body = mutableMapOf<String, Any>()
            if (revision != null) body["expectedRevision"] = revision
            if (action == "reject") body["reason"] = reason
            return mvc.perform(post("/api/v1/admin/events/${e.id}/$action")
                .contentType(MediaType.APPLICATION_JSON).content(json.writeValueAsBytes(body))).andReturn().response
        } finally { SecurityContextHolder.clearContext() }
    }

    private fun editRequest(actor: UUID, e: Event, revision: String?): org.springframework.mock.web.MockHttpServletResponse {
        val principal = AuthPrincipal(actor, "admin@test.local", Role.ADMIN)
        SecurityContextHolder.getContext().authentication = UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
        try {
            val mvc = MockMvcBuilders.standaloneSetup(AdminEventsController(eventAdmin))
                .setControllerAdvice(GlobalExceptionHandler(), ResponseEnvelopeAdvice())
                .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver()).build()
            val request = multipart(HttpMethod.PATCH, "/api/v1/admin/events/${e.id}").param("title", "Edited")
            if (revision != null) request.param("expectedRevision", revision)
            return mvc.perform(request).andReturn().response
        } finally { SecurityContextHolder.clearContext() }
    }

    private fun revision(e: Event): Long = jdbc.queryForObject("SELECT review_revision FROM events WHERE id = ?", Long::class.java, e.id)!!

    @Test fun `an old approval cannot approve a replacement pricing submission`() {
        val e = event(EventReviewStatus.CHANGE_PENDING)
        e.pendingChange = mapOf("pricingMode" to "paid", "pricePerPhoto" to "200.00", "watermarkPolicy" to "platform")
        events.saveAndFlush(e)
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(pricePerPhoto = BigDecimal("250")), null)
        val response = reviewRequest(user(Role.ADMIN).id, e, "approve", 0)
        assertEquals(409, response.status, response.contentAsString)
        val saved = events.findById(e.id).get()
        assertEquals(EventReviewStatus.CHANGE_PENDING, saved.reviewStatus)
        assertEquals("250.00", saved.pendingChange?.get("pricePerPhoto"))
        assertEquals(BigDecimal("150.00"), saved.pricePerPhoto)
        eventEffects(e.id, 0)
    }

    @Test fun `an old rejection cannot discard edited pending evidence`() {
        val e = event(EventReviewStatus.PENDING)
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(title = "Replacement evidence"), null)
        val response = reviewRequest(user(Role.ADMIN).id, e, "reject", 0)
        assertEquals(409, response.status, response.contentAsString)
        assertEquals(EventReviewStatus.PENDING, events.findById(e.id).get().reviewStatus)
        eventEffects(e.id, 0)
    }

    @Test fun `accepted review responses expose increasing revision and old v1 calls still work`() {
        val e = event(EventReviewStatus.PENDING); val a = user(Role.ADMIN).id
        assertEquals(0, revision(e))
        val response = reviewRequest(a, e, "reject", 0)
        assertEquals(200, response.status, response.contentAsString)
        assertEquals(1, json.readTree(response.contentAsString).path("data").path("reviewRevision").asLong(-1))
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(title = "Resubmitted"), null)
        assertEquals(2, revision(e))
        assertEquals(409, reviewRequest(a, e, "approve", 0).status)
        assertEquals(200, reviewRequest(a, e, "approve", null).status)
        assertEquals(3, revision(e))
        eventEffects(e.id, 2)
    }

    @Test fun `pending replacement withdrawal and later submission advance generations but identical text is a no-op`() {
        val e = event()
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(pricePerPhoto = BigDecimal("200")), null)
        assertEquals(1, revision(e))
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(pricePerPhoto = BigDecimal("200")), null)
        assertEquals(1, revision(e))
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(title = "Reviewed evidence"), null)
        assertEquals(2, revision(e))
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(title = "Reviewed evidence"), null)
        assertEquals(2, revision(e))
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(withdrawPendingChange = true), null)
        assertEquals(3, revision(e))
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(withdrawPendingChange = true), null)
        assertEquals(3, revision(e))
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(pricePerPhoto = BigDecimal("250")), null)
        assertEquals(4, revision(e))
        assertEquals(409, reviewRequest(user(Role.ADMIN).id, e, "approve", 1).status)
        eventEffects(e.id, 0)
    }

    @Test fun `admin multipart edit preconditions compare the locked current generation`() {
        val e = event(EventReviewStatus.PENDING); val a = user(Role.ADMIN).id
        ownerEvents.update(e.createdBy!!, e.id, UpdateMyEventRequest(title = "Replacement"), null)
        assertEquals(409, editRequest(a, e, "0").status)
        assertEquals("Replacement", events.findById(e.id).get().name)
        eventEffects(e.id, 0)
        assertEquals(200, editRequest(a, e, "1").status)
        assertEquals(2, revision(e))
        assertEquals(200, editRequest(a, e, "2").status)
        assertEquals(2, revision(e))
        assertEquals(1, count("SELECT count(*) FROM admin_decision_log WHERE target_event_id = ?", e.id))
    }

    @Test fun `negative event review and edit preconditions are rejected before writes`() {
        val a = user(Role.ADMIN).id
        for (action in listOf("approve", "reject")) {
            val e = event(EventReviewStatus.PENDING)
            assertEquals(400, reviewRequest(a, e, action, -1).status)
            eventEffects(e.id, 0)
        }
        val e = event(EventReviewStatus.PENDING)
        assertEquals(400, editRequest(a, e, "-1").status)
        eventEffects(e.id, 0)
    }

    @Test fun `event revision migration defaults to zero and rejects negative values`() {
        val e = event()
        assertEquals(0, revision(e))
        assertFailsWith<org.springframework.dao.DataIntegrityViolationException> {
            jdbc.update("UPDATE events SET review_revision = -1 WHERE id = ?", e.id)
        }
        assertEquals(0, revision(e))
    }

    @Test fun `review rollback leaves the pending generation and zero side effects`() {
        val e = event(EventReviewStatus.PENDING); val a = user(Role.ADMIN).id
        assertFailsWith<IllegalStateException> { TransactionTemplate(txManager).executeWithoutResult {
            eventAdmin.approve(a, e.id); error("Rollback")
        } }
        assertEquals(0, revision(e))
        assertEquals(EventReviewStatus.PENDING, events.findById(e.id).get().reviewStatus)
        eventEffects(e.id, 0)
    }

    @ParameterizedTest
    @ValueSource(booleans = [false, true])
    fun `admin repricing does not deadlock a photo writer updating its event count`(approval: Boolean) {
        val p = photo(); val a = user(Role.ADMIN).id
        if (approval) {
            val e = events.findById(p.eventId).get()
            e.reviewStatus = EventReviewStatus.CHANGE_PENDING
            e.pendingChange = mapOf("pricingMode" to "paid", "pricePerPhoto" to "200.00", "watermarkPolicy" to "platform")
            events.saveAndFlush(e)
        }
        val pool = Executors.newSingleThreadExecutor()
        try {
            dataSource.connection.use { writer ->
                writer.autoCommit = false
                writer.prepareStatement("SELECT id FROM photos WHERE id = ? FOR UPDATE").use {
                    it.setObject(1, p.id); it.executeQuery().close()
                }
                val edit = pool.submit<Result<AdminListEventDto>> {
                    runCatching {
                        if (approval) eventAdmin.approve(a, p.eventId)
                        else eventAdmin.update(a, p.eventId, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("200")))
                    }
                }
                try {
                    awaitBlocked(1)
                    // The same photo -> coverage -> event order used by restore/trash/upload.
                    val counter = runCatching { writer.prepareStatement("UPDATE events SET photo_count = photo_count + 1 WHERE id = ?").use {
                        it.setObject(1, p.eventId); it.executeUpdate()
                    } }
                    writer.commit()
                    val result = edit.get(20, TimeUnit.SECONDS)
                    assertTrue(counter.isSuccess, counter.toString())
                    assertTrue(result.isSuccess, result.toString())
                } finally { writer.rollback() }
            }
            assertEquals(BigDecimal("200.00"), photos.findById(p.id).get().pricePhp)
            assertEquals(1, events.findById(p.eventId).get().photoCount)
        } finally { pool.shutdownNow() }
    }

    @Test fun `approving a submission leaves its photo rows free for a concurrent photo writer`() {
        val e = event(EventReviewStatus.PENDING); val a = user(Role.ADMIN).id
        val p = photos.saveAndFlush(Photo(eventId = e.id, photographerId = e.createdBy, s3Key = "test/original.jpg",
            pricePhp = BigDecimal("150.00"), status = PhotoStatus.LIVE))
        TransactionTemplate(txManager).executeWithoutResult {
            eventAdmin.approve(a, e.id)
            // The approval is still open here: a photo writer must get the row at once.
            dataSource.connection.use { writer ->
                writer.autoCommit = false
                writer.prepareStatement("SELECT id FROM photos WHERE id = ? FOR UPDATE NOWAIT").use {
                    it.setObject(1, p.id); it.executeQuery().use { rows -> assertTrue(rows.next()) }
                }
                writer.rollback()
            }
        }
        assertEquals(EventReviewStatus.APPROVED, events.findById(e.id).get().reviewStatus)
    }

    @Test fun `the repricing photo lock lets a cart row for a locked photo be inserted`() {
        val p = photo(); val runner = user(Role.RUNNER).id
        TransactionTemplate(txManager).executeWithoutResult {
            assertEquals(listOf(p.id), photos.lockIdsByEventIdForUpdate(p.eventId))
            // The repricing lock is still held here: a runner adding the photo to a cart must not wait for it.
            dataSource.connection.use { writer ->
                writer.autoCommit = false
                writer.createStatement().use { it.execute("SET LOCAL lock_timeout = '2s'") }
                writer.prepareStatement("INSERT INTO cart_items (user_id, photo_id, event_id, price_php_at_add) VALUES (?, ?, ?, 150.00)").use {
                    it.setObject(1, runner); it.setObject(2, p.id); it.setObject(3, p.eventId)
                    assertEquals(1, it.executeUpdate())
                }
                writer.rollback()
            }
        }
    }

    @Test fun `withdrawn dispute cannot be resurrected by a delayed escalation`() {
        val p = photo(); val runner = user(Role.RUNNER).id
        val order = orders.saveAndFlush(Order(eventId = p.eventId, recipientEmail = "test@example.test",
            userId = runner, paymentMethodWire = "gcash", totalPhp = BigDecimal("150.00")))
        val d = disputes.saveAndFlush(Dispute(orderId = order.id, photoId = p.id, runnerId = runner, reasonWire = "other"))
        refunds.withdraw(runner, d.id)
        val error = assertFailsWith<ApiException> { disputeAdmin.escalate(user(Role.ADMIN).id, d.id, null) }
        assertEquals(409, error.status.value())
        assertEquals(DisputeStatus.WITHDRAWN, disputes.findById(d.id).get().status)
        assertEquals(0, count("SELECT count(*) FROM admin_decision_log WHERE target_dispute_id = ?", d.id))
        assertEquals(0, count("SELECT count(*) FROM photographer_messages WHERE photographer_id = ?", p.photographerId!!))
        assertEquals(0, count("SELECT count(*) FROM runner_messages WHERE runner_id = ?", runner))
        disputes.saveAndFlush(Dispute(orderId = order.id, photoId = p.id, runnerId = runner, reasonWire = "other"))
    }

    @Test fun `bulk trash and event repricing lock overlapping photos in the same order`() {
        val e = event(); val seller = e.createdBy!!; val admin = user(Role.ADMIN).id
        // Reverse physical order so the unordered bulk UPDATE reaches the larger UUID first.
        val high = UUID.fromString("70000000-0000-0000-0000-000000000002")
        val low = UUID.fromString("10000000-0000-0000-0000-000000000001")
        for (id in listOf(high, low)) photos.saveAndFlush(Photo(id = id, eventId = e.id,
            photographerId = seller, s3Key = "test/$id.jpg", pricePhp = BigDecimal("150"), status = PhotoStatus.LIVE))
        jdbc.update("UPDATE events SET photo_count = 2 WHERE id = ?", e.id)
        jdbc.execute("""CREATE FUNCTION pause_transition_trash() RETURNS trigger LANGUAGE plpgsql AS ${'$'}${'$'}
            BEGIN
              IF NEW.id = '$high' AND NEW.deleted_at IS NOT NULL AND OLD.deleted_at IS NULL THEN
                PERFORM pg_advisory_xact_lock(78202);
              END IF;
              RETURN NEW;
            END ${'$'}${'$'}""")
        jdbc.execute("CREATE TRIGGER pause_transition_trash BEFORE UPDATE ON photos FOR EACH ROW EXECUTE FUNCTION pause_transition_trash()")
        val pool = Executors.newFixedThreadPool(2)
        try {
            dataSource.connection.use { pause ->
                pause.autoCommit = false
                pause.createStatement().use { it.execute("SELECT pg_advisory_xact_lock(78202)") }
                val trash = pool.submit<Result<Int>> { runCatching { TransactionTemplate(txManager).execute {
                    jdbc.execute("SET LOCAL enable_indexscan = off")
                    jdbc.execute("SET LOCAL enable_bitmapscan = off")
                    photoOwner.trash(seller, e.id, listOf(high, low))
                }!! } }
                try {
                    awaitBlocked(1)
                    val edit = pool.submit<Result<AdminListEventDto>> { runCatching {
                        eventAdmin.update(admin, e.id, UpdateAdminEventRequest(pricePerPhoto = BigDecimal("200")))
                    } }
                    awaitBlocked(2)
                    pause.commit()
                    val trashed = trash.get(25, TimeUnit.SECONDS)
                    val edited = edit.get(25, TimeUnit.SECONDS)
                    assertTrue(trashed.isSuccess, trashed.toString())
                    assertTrue(edited.isSuccess, edited.toString())
                    assertEquals(2, trashed.getOrThrow())
                } finally { pause.rollback() }
            }
            assertEquals(0, events.findById(e.id).get().photoCount)
            for (id in listOf(high, low)) {
                val saved = photos.findById(id).get()
                assertNotNull(saved.deletedAt)
                assertEquals(BigDecimal("200.00"), saved.pricePhp)
            }
        } finally {
            pool.shutdownNow()
            jdbc.execute("DROP TRIGGER pause_transition_trash ON photos")
            jdbc.execute("DROP FUNCTION pause_transition_trash()")
        }
    }

}
