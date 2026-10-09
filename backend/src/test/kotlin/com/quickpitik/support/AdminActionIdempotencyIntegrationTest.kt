package com.quickpitik.support

import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.common.ResponseEnvelopeAdvice
import com.quickpitik.controller.AdminUsersController
import com.quickpitik.entity.*
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminUserService
import com.quickpitik.service.admin.AdminActionCommandService
import com.quickpitik.service.admin.AdminActionResult
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.repository.AdminActionCommandRepository
import com.quickpitik.service.runner.RunnerMessagesService
import org.springframework.dao.DataAccessException
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.support.TransactionTemplate
import javax.sql.DataSource
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import org.junit.jupiter.params.provider.ValueSource
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.http.MediaType
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import java.util.UUID
import java.time.Clock
import java.time.Instant
import java.time.ZoneId
import java.time.ZoneOffset
import java.util.concurrent.CountDownLatch
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.*

class AdminActionIdempotencyIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var admin: AdminUserService
    @Autowired lateinit var jdbc: JdbcTemplate
    @Autowired lateinit var mapper: ObjectMapper
    @Autowired lateinit var commands: AdminActionCommandService
    @Autowired lateinit var receipts: AdminActionCommandRepository
    @Autowired lateinit var log: AdminDecisionLogService
    @Autowired lateinit var inbox: RunnerMessagesService
    @Autowired lateinit var manager: PlatformTransactionManager
    @Autowired lateinit var dataSource: DataSource

    @Autowired lateinit var eventAdmin: com.quickpitik.service.admin.AdminEventService
    @Autowired lateinit var events: com.quickpitik.repository.EventRepository
    @org.springframework.test.context.bean.override.mockito.MockitoBean lateinit var storage: com.quickpitik.service.storage.StorageService
    @org.springframework.test.context.bean.override.mockito.MockitoBean lateinit var eventClock: Clock
    private var now = Instant.parse("2026-10-08T12:00:00Z")
    private val objects = java.util.concurrent.ConcurrentHashMap<String, ByteArray>()
    private val puts = CopyOnWriteArrayList<String>()
    private val deletes = CopyOnWriteArrayList<String>()
    private var afterPut: (String) -> Unit = {}
    private var beforePut: () -> Unit = {}

    @org.junit.jupiter.api.BeforeEach fun storageFake() {
        objects.clear(); puts.clear(); deletes.clear(); afterPut = {}; beforePut = {}
        now = Instant.parse("2026-10-08T12:00:00Z")
        org.mockito.Mockito.`when`(eventClock.instant()).thenAnswer { now }
        org.mockito.Mockito.`when`(storage.put(org.mockito.Mockito.anyString(), anyArg<ByteArray>(), org.mockito.Mockito.anyString())).thenAnswer {
            assertFalse(org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive())
            beforePut()
            val key = it.getArgument<String>(0); val bytes = it.getArgument<ByteArray>(1)
            puts.add(key); objects[key] = bytes; afterPut(key)
            com.quickpitik.service.storage.StoredObject(key, bytes.size.toLong(), "image/jpeg")
        }
        org.mockito.Mockito.`when`(storage.metadata(org.mockito.Mockito.anyString())).thenAnswer {
            assertFalse(org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive())
            val key = it.getArgument<String>(0)
            objects[key]?.let { bytes -> com.quickpitik.service.storage.StoredObject(key, bytes.size.toLong(), "image/jpeg") }
        }
        org.mockito.Mockito.`when`(storage.getBytes(org.mockito.Mockito.anyString())).thenAnswer {
            assertFalse(org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive())
            objects[it.getArgument<String>(0)] ?: error("missing fixture object")
        }
        org.mockito.Mockito.doAnswer {
            assertFalse(org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive())
            val key = it.getArgument<String>(0)
            assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM events WHERE cover_s3_key = ?", Int::class.java, key))
            deletes.add(key); objects.remove(key); null
        }.`when`(storage).delete(org.mockito.Mockito.anyString())
        org.mockito.Mockito.`when`(storage.presignedGetUrl(org.mockito.Mockito.anyString(), anyArg())).thenAnswer {
            "https://fixture.local/${it.getArgument<String>(0)}?fresh=${UUID.randomUUID()}"
        }
    }

    private fun <T> anyArg(): T = org.mockito.Mockito.any()

    private fun image(color: Int = 0): ByteArray {
        val image = java.awt.image.BufferedImage(40, 30, java.awt.image.BufferedImage.TYPE_INT_RGB)
        image.setRGB(20, 15, color)
        return java.io.ByteArrayOutputStream().also { javax.imageio.ImageIO.write(image, "png", it) }.toByteArray()
    }

    private fun eventRequest(actor: UUID, key: String?, title: String, id: UUID? = null, cover: ByteArray? = null, revision: Long? = null, region: String? = null, province: String? = null): org.springframework.mock.web.MockHttpServletResponse {
        val principal = AuthPrincipal(actor, "fixture@test.local", users.findById(actor).orElseThrow().role)
        SecurityContextHolder.getContext().authentication = UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
        try {
            val mvc = MockMvcBuilders.standaloneSetup(com.quickpitik.controller.AdminEventsController(eventAdmin))
                .setControllerAdvice(GlobalExceptionHandler(), ResponseEnvelopeAdvice())
                .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver()).build()
            val request = org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart(if (id == null) "/api/v1/admin/events" else "/api/v1/admin/events/$id")
            request.param("title", title).param("date", "2026-10-15").param("location", "Cebu").param("pricePerPhoto", "25.00")
            if (id != null) request.with { it.method = "PATCH"; it }
            if (key != null) request.header("Idempotency-Key", key)
            if (revision != null) request.param("expectedRevision", revision.toString())
            if (region != null) request.param("regionCode", region)
            if (province != null) request.param("provinceCode", province)
            if (cover != null) request.file(org.springframework.mock.web.MockMultipartFile("cover", "cover.png", "image/png", cover))
            return mvc.perform(request).andReturn().response
        } finally { SecurityContextHolder.clearContext() }
    }
    private fun eventId(response: org.springframework.mock.web.MockHttpServletResponse): UUID {
        assertEquals(200, response.status, response.contentAsString)
        return UUID.fromString(mapper.readTree(response.contentAsString)["data"]["id"].asText())
    }
    private fun eventEffects(id: UUID, expected: Int) = assertEquals(expected, count("admin_decision_log", "target_event_id", id))
    private fun receiptState(key: String) = jdbc.queryForObject("SELECT state FROM admin_action_commands WHERE command_id = ?", String::class.java, UUID.fromString(key))

    @ParameterizedTest @ValueSource(booleans = [false, true])
    fun `event create lost response replays one identity and allows a new intent with the same title`(hasCover: Boolean) {
        val actor=user(Role.ADMIN); val key=UUID.randomUUID().toString(); val title="Intent ${UUID.randomUUID()}"
        val bytes=if (hasCover) image() else null
        val first=eventRequest(actor,key," $title ",cover=bytes); val id=eventId(first)
        val second=eventRequest(actor,key,title,cover=bytes)
        assertEquals(id,eventId(second)); eventEffects(id,1)
        assertEquals(mapper.readTree(first.contentAsString)["data"]["slug"],mapper.readTree(second.contentAsString)["data"]["slug"])
        assertEquals(if(hasCover) 1 else 0,puts.size)
        if(hasCover) assertNotEquals(mapper.readTree(first.contentAsString)["data"]["bannerUrl"],mapper.readTree(second.contentAsString)["data"]["bannerUrl"])
        assertNotEquals(id,eventId(eventRequest(actor,UUID.randomUUID().toString(),title,cover=bytes)))
    }

    private fun lookup(actor: UUID, key: String): org.springframework.mock.web.MockHttpServletResponse {
        val principal = AuthPrincipal(actor, "fixture@test.local", Role.ADMIN)
        SecurityContextHolder.getContext().authentication = UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
        try {
            return MockMvcBuilders.standaloneSetup(com.quickpitik.controller.AdminCommandsController(commands))
                .setControllerAdvice(GlobalExceptionHandler(), ResponseEnvelopeAdvice())
                .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver()).build()
                .perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get("/api/v1/admin/commands/$key"))
                .andReturn().response
        } finally { SecurityContextHolder.clearContext() }
    }

    @Test fun `lookup after a lost response shows the original result only to its own admin`() {
        val actor=user(Role.ADMIN); val key=UUID.randomUUID().toString()
        val id=eventId(eventRequest(actor,key,"Lookup ${UUID.randomUUID()}"))
        val found=lookup(actor,key); assertEquals(200,found.status,found.contentAsString)
        val data=mapper.readTree(found.contentAsString)["data"]
        assertEquals(setOf("commandId","state","resourceId","decisionId","messageId","completedAt"),data.fieldNames().asSequence().toSet())
        assertEquals(key,data["commandId"].asText()); assertEquals("SUCCEEDED",data["state"].asText())
        assertEquals(id.toString(),data["resourceId"].asText()); assertFalse(data["completedAt"].isNull)
        assertEquals(404,lookup(user(Role.ADMIN),key).status)
        assertEquals(404,lookup(actor,UUID.randomUUID().toString()).status)
        assertEquals(400,lookup(actor,"bad-key").status)

        val target=user(Role.RUNNER); val messageKey=UUID.randomUUID().toString()
        val sent=mapper.readTree(send(actor,target,messageKey).contentAsString)["data"]
        val message=mapper.readTree(lookup(actor,messageKey).contentAsString)["data"]
        assertEquals(sent["id"],message["messageId"]); assertEquals(sent["sourceDecisionId"],message["decisionId"])
        eventEffects(id,1); effects(target,Role.RUNNER,1)
    }

    @Test fun `completed cover edit replay precedes old revision and never overwrites a later edit`() {
        val actor=user(Role.ADMIN); val id=eventId(eventRequest(actor,null,"Edit ${UUID.randomUUID()}",cover=image()))
        val key=UUID.randomUUID().toString(); val bytes=image(0xffffff)
        eventId(eventRequest(actor,key,"Changed",id,bytes,0)); val firstKey=events.findById(id).orElseThrow().coverS3Key
        eventId(eventRequest(actor,UUID.randomUUID().toString(),"Later",id,image(0xff0000),1))
        eventId(eventRequest(actor,key,"Changed",id,bytes,0))
        assertEquals("Later",events.findById(id).orElseThrow().name); assertNotEquals(firstKey,events.findById(id).orElseThrow().coverS3Key)
        assertEquals(2,events.findById(id).orElseThrow().reviewRevision); eventEffects(id,3); assertEquals(3,puts.size); assertEquals(2,deletes.size)
    }

    @ParameterizedTest @ValueSource(strings=["actor","bytes","operation","target","classification"])
    fun `event key rejects changed identity before upload`(field: String) {
        val actor=user(Role.ADMIN); val key=UUID.randomUUID().toString(); val title="Mismatch ${UUID.randomUUID()}"; val bytes=image()
        val id=eventId(eventRequest(actor,key,title,cover=bytes))
        val response=when(field) {
            "actor" -> eventRequest(user(Role.ADMIN),key,title,cover=bytes)
            "bytes" -> eventRequest(actor,key,title,cover=image(0xffffff))
            "operation" -> eventRequest(actor,key,title,id,bytes)
            "classification" -> eventRequest(actor,key,title,cover=bytes,region="region-7",province="cebu")
            else -> { val updateKey=UUID.randomUUID().toString(); eventId(eventRequest(actor,updateKey,"Changed",id,bytes)); eventRequest(actor,updateKey,"Changed",UUID.randomUUID(),bytes) }
        }
        assertEquals(409,response.status); assertEquals(if(field=="target") 2 else 1,puts.size)
    }

    @Test fun `event malformed key and unauthorized replay are rejected`() {
        val actor=user(Role.ADMIN); val title="Auth ${UUID.randomUUID()}"; val key=UUID.randomUUID().toString()
        assertEquals(400,eventRequest(actor,"bad-key",title).status)
        eventId(eventRequest(actor,key,title)); jdbc.update("UPDATE users SET suspended_at=now() WHERE id=?",actor)
        assertEquals(403,eventRequest(actor,key,title).status)
    }

    @Test fun `suspended event command actor cannot replay a success`() {
        val actor=user(Role.ADMIN); val title="Replay auth ${UUID.randomUUID()}"; val key=UUID.randomUUID().toString()
        eventId(eventRequest(actor,key,title)); jdbc.update("UPDATE users SET suspended_at=now() WHERE id=?",actor)
        assertEquals(403,eventRequest(actor,key,title).status)
    }

    @Test fun `uncertain PUT recovers the fixed object without a second upload`() {
        val actor=user(Role.ADMIN); val title="Uncertain ${UUID.randomUUID()}"; val key=UUID.randomUUID().toString(); val bytes=image()
        afterPut={ throw IllegalStateException("lost storage response") }
        assertEquals(409,eventRequest(actor,key,title,cover=bytes).status); assertEquals("UNKNOWN",receiptState(key))
        afterPut={}; val id=eventId(eventRequest(actor,key,title,cover=bytes)); eventEffects(id,1); assertEquals(1,puts.size)
        assertEquals(puts.single(),events.findById(id).orElseThrow().coverS3Key)
    }

    @ParameterizedTest @ValueSource(booleans=[false,true])
    fun `expired process loss reconciles the same key before or after PUT`(after: Boolean) {
        val actor=user(Role.ADMIN); val title="Crash ${UUID.randomUUID()}"; val key=UUID.randomUUID().toString(); val bytes=image()
        if(after) afterPut={ throw AssertionError("process lost") } else beforePut={ throw AssertionError("process lost") }
        assertEquals(500,eventRequest(actor,key,title,cover=bytes).status)
        assertEquals("PENDING",receiptState(key)); beforePut={}; afterPut={}
        assertEquals(409,eventRequest(actor,key,title,cover=bytes).status)
        now=now.plusSeconds(301)
        val id=eventId(eventRequest(actor,key,title,cover=bytes)); eventEffects(id,1); assertEquals(1,puts.size)
    }

    @Test fun `conflicting recovery object remains UNKNOWN without a new event`() {
        val actor=user(Role.ADMIN); val title="Conflict ${UUID.randomUUID()}"; val key=UUID.randomUUID().toString(); val bytes=image()
        afterPut={ throw IllegalStateException("lost") }; eventRequest(actor,key,title,cover=bytes)
        val reserved=puts.single(); objects[reserved]=ByteArray(objects[reserved]!!.size) { 5 }; afterPut={}
        assertEquals(409,eventRequest(actor,key,title,cover=bytes).status); assertEquals("UNKNOWN",receiptState(key)); assertEquals(1,puts.size)
        assertEquals(0,jdbc.queryForObject("SELECT count(*) FROM events WHERE name=?",Int::class.java,title))
    }

    @Test fun `active claim blocks a duplicate and fences late original completion after takeover`() {
        val actor=user(Role.ADMIN); val title="Late ${UUID.randomUUID()}"; val key=UUID.randomUUID().toString(); val bytes=image()
        val entered=CountDownLatch(1); val release=CountDownLatch(1); val pool=Executors.newSingleThreadExecutor()
        afterPut={ entered.countDown(); check(release.await(20,TimeUnit.SECONDS)) }
        try {
            val first=pool.submit<org.springframework.mock.web.MockHttpServletResponse> { eventRequest(actor,key,title,cover=bytes) }
            assertTrue(entered.await(15,TimeUnit.SECONDS))
            val duplicate=eventRequest(actor,key,title,cover=bytes)
            assertEquals(409,duplicate.status); assertEquals("IDEMPOTENT_REPLAY_IN_PROGRESS",mapper.readTree(duplicate.contentAsString)["errors"][0]["code"].asText())
            now=now.plusSeconds(301); afterPut={}
            val id=eventId(eventRequest(actor,key,title,cover=bytes)); release.countDown()
            assertEquals(id,eventId(first.get(15,TimeUnit.SECONDS))); eventEffects(id,1); assertEquals(1,puts.size)
        } finally {release.countDown();pool.shutdownNow()}
    }

    @Test fun `late original is fenced while recovery still owns an unfinished PUT`() {
        val actor=user(Role.ADMIN); val title="Fence ${UUID.randomUUID()}"; val key=UUID.randomUUID().toString(); val bytes=image()
        val originalWaiting=CountDownLatch(1); val releaseOriginal=CountDownLatch(1)
        val recoveryWaiting=CountDownLatch(1); val releaseRecovery=CountDownLatch(1)
        val attempts=AtomicInteger(); val pool=Executors.newFixedThreadPool(2)
        beforePut={
            if(attempts.incrementAndGet()==1) { originalWaiting.countDown(); check(releaseOriginal.await(20,TimeUnit.SECONDS)) }
            else { recoveryWaiting.countDown(); check(releaseRecovery.await(20,TimeUnit.SECONDS)) }
        }
        try {
            val first=pool.submit<org.springframework.mock.web.MockHttpServletResponse> {eventRequest(actor,key,title,cover=bytes)}
            assertTrue(originalWaiting.await(10,TimeUnit.SECONDS)); now=now.plusSeconds(301)
            val second=pool.submit<org.springframework.mock.web.MockHttpServletResponse> {eventRequest(actor,key,title,cover=bytes)}
            assertTrue(recoveryWaiting.await(10,TimeUnit.SECONDS)); releaseOriginal.countDown()
            assertEquals(409,first.get(10,TimeUnit.SECONDS).status)
            assertEquals("PENDING",receiptState(key))
            assertEquals(0,jdbc.queryForObject("SELECT count(*) FROM events WHERE name=?",Int::class.java,title))
            releaseRecovery.countDown(); val id=eventId(second.get(10,TimeUnit.SECONDS)); eventEffects(id,1)
            assertEquals(1,puts.toSet().size)
        } finally {releaseOriginal.countDown();releaseRecovery.countDown();pool.shutdownNow()}
    }

    @Test fun `delete during cover PUT prevents edit commit and keeps old cover`() {
        val actor=user(Role.ADMIN); val id=eventId(eventRequest(actor,null,"Deleted ${UUID.randomUUID()}",cover=image())); val old=events.findById(id).orElseThrow().coverS3Key
        afterPut={ eventAdmin.delete(actor,id) }
        val key=UUID.randomUUID().toString()
        assertEquals(404,eventRequest(actor,key,"Changed",id,image(0xffffff),0).status)
        assertEquals(old,events.findById(id).orElseThrow().coverS3Key); assertTrue(deletes.isEmpty()); eventEffects(id,2)
        afterPut={}; assertEquals(404,eventRequest(actor,key,"Changed",id,image(0xffffff),0).status); assertEquals(2,puts.size)
    }

    @ParameterizedTest @ValueSource(booleans=[false,true])
    fun `PUT success then database rollback retains provenance and retries one commit`(editing: Boolean) {
        val actor=user(Role.ADMIN); val title="Rollback ${UUID.randomUUID()}"; val key=UUID.randomUUID().toString(); val bytes=image(0xffffff)
        val id=if(editing) eventId(eventRequest(actor,null,title,cover=image())) else null
        val old=id?.let { events.findById(it).orElseThrow().coverS3Key }
        jdbc.execute("CREATE FUNCTION fail_event_receipt() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.command_id = '$key' AND NEW.state = 'SUCCEEDED' THEN RAISE EXCEPTION 'fixture completion failure'; END IF; RETURN NEW; END; $$")
        jdbc.execute("CREATE TRIGGER fail_event_receipt BEFORE UPDATE ON admin_action_commands FOR EACH ROW EXECUTE FUNCTION fail_event_receipt()")
        try {
            assertEquals(409,eventRequest(actor,key,"$title changed",id,bytes,if(editing) 0 else null).status)
            assertEquals("UNKNOWN",receiptState(key)); assertTrue(deletes.isEmpty())
            if(id!=null) { assertEquals(old,events.findById(id).orElseThrow().coverS3Key); eventEffects(id,1) }
            else assertEquals(0,jdbc.queryForObject("SELECT count(*) FROM events WHERE name=?",Int::class.java,"$title changed"))
        } finally { jdbc.execute("DROP TRIGGER fail_event_receipt ON admin_action_commands"); jdbc.execute("DROP FUNCTION fail_event_receipt()") }
        val resultId=eventId(eventRequest(actor,key,"$title changed",id,bytes,if(editing) 0 else null))
        eventEffects(resultId,if(editing) 2 else 1); assertEquals(if(editing) 2 else 1,puts.size)
        assertEquals(if(editing) listOf(old) else emptyList(),deletes.toList())
    }

    private fun reviewEdit(service: AdminActionCommandService, key: IdempotencyKey, actor: UUID, id: UUID,
        prepare: () -> String? = { null }, write: (String, Boolean) -> Unit = { _, _ -> error("no cover") },
        beforeAction: () -> Unit = {}): AdminActionResult =
        service.executeEvent(key, actor, AdminActionOperation.EVENT_UPDATE, id, "a".repeat(64), {}, {
            events.findByIdForReview(id)!!.requireReviewRevision(0)
        }, prepare, write, { _, _ ->
            beforeAction()
            val event = events.findByIdForReview(id)!!
            event.requireReviewRevision(0)
            event.name = "Reviewed edit"; event.reviewRevision++
            events.save(event)
            AdminActionResult(id, log.logEventDecision(actor, id, "edited").id, null)
        })

    @ParameterizedTest @ValueSource(booleans = [false, true])
    fun `event completion between receipt reads replays authoritative result`(expired: Boolean) {
        val actor = user(Role.ADMIN); val id = eventId(eventRequest(actor, null, "Read race ${UUID.randomUUID()}"))
        val key = IdempotencyKey(UUID.randomUUID().toString())
        if (expired) receipts.saveAndFlush(AdminActionCommand(commandId = UUID.fromString(key.value), actorId = actor,
            operation = AdminActionOperation.EVENT_UPDATE, targetId = id, inputHash = "a".repeat(64),
            leaseExpiresAt = now.minusSeconds(1), intendedEventId = id, claimedAt = now, updatedAt = now))
        val paused = CountDownLatch(1); val resume = CountDownLatch(1)
        val prepared = java.util.concurrent.atomic.AtomicBoolean()
        val delegate = org.mockito.AdditionalAnswers.delegatesTo<Any>(receipts)
        val delayedReceipts = org.mockito.Mockito.mock(AdminActionCommandRepository::class.java) { invocation ->
            val name = invocation.method.name
            if (expired && prepared.get() && name == "findForUpdate") {
                paused.countDown(); check(resume.await(20, TimeUnit.SECONDS))
            }
            val result = delegate.answer(invocation)
            if (!expired && !prepared.get() && name in listOf("findById", "findForUpdate")) {
                paused.countDown(); check(resume.await(20, TimeUnit.SECONDS))
            }
            result
        }
        val delayed = AdminActionCommandService(delayedReceipts, manager, mapper, eventClock)
        val pool = Executors.newSingleThreadExecutor()
        try {
            val loser = pool.submit<AdminActionResult> { reviewEdit(delayed, key, actor, id,
                prepare = { prepared.set(true); null }) }
            assertTrue(paused.await(15, TimeUnit.SECONDS))
            val winner = try { reviewEdit(commands, key, actor, id) } finally { resume.countDown() }
            assertEquals(winner, loser.get(20, TimeUnit.SECONDS))
            assertEquals(1, events.findById(id).orElseThrow().reviewRevision); eventEffects(id, 2)
            assertEquals("SUCCEEDED", receiptState(key.value))
        } finally { resume.countDown(); pool.shutdownNow() }
    }

    @Test fun `initial same key event claims keep receipt before event lock order`() {
        val actor = user(Role.ADMIN); val id = eventId(eventRequest(actor, null, "Lock race ${UUID.randomUUID()}"))
        val key = IdempotencyKey(UUID.randomUUID().toString())
        val secondMiss = CountDownLatch(1); val firstClaim = CountDownLatch(1)
        val secondReady = CountDownLatch(1); val firstReceipt = CountDownLatch(1)
        val lookups = AtomicInteger(); val writes = AtomicInteger(); val actions = AtomicInteger()
        val delegate = org.mockito.AdditionalAnswers.delegatesTo<Any>(receipts)
        val delayedReceipts = org.mockito.Mockito.mock(AdminActionCommandRepository::class.java) { invocation ->
            if (invocation.method.name == "insertEventClaim") {
                secondMiss.countDown(); check(firstClaim.await(20, TimeUnit.SECONDS))
                secondReady.countDown(); check(firstReceipt.await(20, TimeUnit.SECONDS))
            }
            val result = delegate.answer(invocation)
            // Delay the old unlocked claim read before it can acquire the event lock.
            if (invocation.method.name == "findById" && lookups.incrementAndGet() == 2) {
                secondMiss.countDown(); check(firstClaim.await(20, TimeUnit.SECONDS))
            }
            result
        }
        val delayed = AdminActionCommandService(delayedReceipts, manager, mapper, eventClock)
        val pool = Executors.newFixedThreadPool(2)
        try {
            val first = pool.submit<AdminActionResult> { reviewEdit(commands, key, actor, id, prepare = {
                check(secondMiss.await(20, TimeUnit.SECONDS)); "b".repeat(64)
            }, write = { _, _ ->
                assertFalse(org.springframework.transaction.support.TransactionSynchronizationManager.isActualTransactionActive())
                writes.incrementAndGet(); firstClaim.countDown(); check(secondReady.await(20, TimeUnit.SECONDS))
            }, beforeAction = { actions.incrementAndGet(); firstReceipt.countDown() }) }
            val second = pool.submit<AdminActionResult> { reviewEdit(delayed, key, actor, id,
                prepare = { "b".repeat(64) }, beforeAction = { actions.incrementAndGet() }) }
            assertEquals(first.get(30, TimeUnit.SECONDS), second.get(30, TimeUnit.SECONDS))
            assertEquals(1, writes.get()); assertEquals(1, actions.get())
            assertEquals(1, events.findById(id).orElseThrow().reviewRevision); eventEffects(id, 2)
            assertEquals("SUCCEEDED", receiptState(key.value))
        } finally {
            secondMiss.countDown(); firstClaim.countDown(); secondReady.countDown(); firstReceipt.countDown(); pool.shutdownNow()
        }
    }


    private fun user(role: Role) = users.saveAndFlush(User(email = "message-${UUID.randomUUID()}@test.local",
        passwordHash = "unused", name = "Message fixture", role = role)).id

    private fun send(actor: UUID, target: UUID, key: String? = null, subject: String = "Hello", body: String = "Message"):
        org.springframework.mock.web.MockHttpServletResponse {
        val principal = AuthPrincipal(actor, "fixture@test.local", users.findById(actor).orElseThrow().role)
        SecurityContextHolder.getContext().authentication = UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
        try {
            val mvc = MockMvcBuilders.standaloneSetup(AdminUsersController(admin))
                .setControllerAdvice(GlobalExceptionHandler(), ResponseEnvelopeAdvice())
                .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver()).build()
            val request = post("/api/v1/admin/users/$target/message").contentType(MediaType.APPLICATION_JSON)
                .content(mapper.writeValueAsBytes(mapOf("subject" to subject, "body" to body)))
            if (key != null) request.header("Idempotency-Key", key)
            return mvc.perform(request).andReturn().response
        } finally { SecurityContextHolder.clearContext() }
    }

    private fun count(table: String, column: String, id: UUID): Int =
        jdbc.queryForObject("SELECT count(*) FROM $table WHERE $column = ?", Int::class.java, id)!!

    private fun effects(target: UUID, role: Role, expected: Int) {
        assertEquals(expected, count("admin_decision_log", "target_user_id", target))
        assertEquals(if (role == Role.PHOTOGRAPHER) expected else 0, count("photographer_messages", "photographer_id", target))
        assertEquals(if (role == Role.RUNNER) expected else 0, count("runner_messages", "runner_id", target))
    }

    @ParameterizedTest @EnumSource(value = Role::class, names = ["PHOTOGRAPHER", "RUNNER"])
    fun `lost response replay returns original decision and message in the correct inbox`(role: Role) {
        val actor = user(Role.ADMIN); val target = user(role); val key = UUID.randomUUID().toString()
        val first = send(actor, target, key, " Hello ", " Message ")
        assertEquals(200, first.status)
        val second = send(actor, target, key)
        assertEquals(200, second.status)
        val original = mapper.readTree(first.contentAsString)["data"]
        val replay = mapper.readTree(second.contentAsString)["data"]
        assertEquals(original["id"], replay["id"])
        assertEquals(original["sourceDecisionId"], replay["sourceDecisionId"])
        assertEquals(original["createdAt"], replay["createdAt"])
        assertEquals("Hello", replay["title"].asText())
        effects(target, role, 1)
        val deliberate = send(actor, target, UUID.randomUUID().toString())
        assertEquals(200, deliberate.status)
        assertNotEquals(original["id"], mapper.readTree(deliberate.contentAsString)["data"]["id"])
        effects(target, role, 2)
    }

    @ParameterizedTest @ValueSource(strings = ["subject", "body", "recipient", "actor"])
    fun `same key rejects each changed identity field`(field: String) {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER); val other = user(Role.RUNNER)
        val key = UUID.randomUUID().toString()
        assertEquals(200, send(actor, target, key).status)
        val response = when (field) {
            "subject" -> send(actor, target, key, subject = "Other")
            "body" -> send(actor, target, key, body = "Other")
            "recipient" -> send(actor, other, key)
            else -> send(user(Role.ADMIN), target, key)
        }
        assertEquals(409, response.status)
        assertEquals("CONFLICT", mapper.readTree(response.contentAsString)["errors"][0]["code"].asText())
        effects(target, Role.RUNNER, 1); effects(other, Role.RUNNER, 0)
    }

    @Test fun `malformed key and invalid message do not claim receipts`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        assertEquals(400, send(actor, target, "invalid").status)
        val key = UUID.randomUUID().toString()
        assertEquals(400, send(actor, target, key, subject = " ").status)
        assertEquals(0, count("admin_action_commands", "command_id", UUID.fromString(key)))
        effects(target, Role.RUNNER, 0)
        assertEquals(200, send(actor, target, key).status)
    }

    @Test fun `authorization is checked before successful replay`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER); val key = UUID.randomUUID().toString()
        assertEquals(200, send(actor, target, key).status)
        jdbc.update("UPDATE users SET suspended_at = now() WHERE id = ?", actor)
        assertEquals(403, send(actor, target, key).status)
        assertEquals(403, send(user(Role.RUNNER), target, key).status)
        effects(target, Role.RUNNER, 1)
    }

    @Test fun `legacy unkeyed sends remain separate intentional messages`() {
        val actor = user(Role.ADMIN); val target = user(Role.PHOTOGRAPHER)
        assertEquals(200, send(actor, target).status); assertEquals(200, send(actor, target).status)
        effects(target, Role.PHOTOGRAPHER, 2)
    }

    @ParameterizedTest @EnumSource(value = Role::class, names = ["PHOTOGRAPHER", "RUNNER"])
    fun `simultaneous requests commit one message and replay its IDs`(role: Role) {
        val actor = user(Role.ADMIN); val target = user(role); val key = UUID.randomUUID().toString()
        val barrier = CyclicBarrier(2); val pool = Executors.newFixedThreadPool(2)
        try {
            val futures = (1..2).map { pool.submit<org.springframework.mock.web.MockHttpServletResponse> {
                barrier.await(10, TimeUnit.SECONDS); send(actor, target, key)
            } }
            val responses = futures.map { it.get(25, TimeUnit.SECONDS) }
            responses.forEach { assertEquals(200, it.status) }
            assertEquals(mapper.readTree(responses[0].contentAsString)["data"]["id"],
                mapper.readTree(responses[1].contentAsString)["data"]["id"])
            effects(target, role, 1)
        } finally { pool.shutdownNow() }
    }

    private fun action(actor: UUID, target: UUID): AdminActionResult {
        val decision = log.logUserDecision(actor, target, "messaged", "Hello")
        val message = inbox.pushMessage(target, RunnerMessageKind.ADMIN_MESSAGE, "Message", actor, decision.id, title = "Hello")
        return AdminActionResult(target, decision.id, message.id)
    }

    @Test fun `uniqueness loser rereads original result in a fresh transaction`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        val key = IdempotencyKey(UUID.randomUUID().toString()); val hash = "a".repeat(64)
        val transactions = CopyOnWriteArrayList<Long>(); val executions = AtomicInteger()
        val pool = Executors.newFixedThreadPool(2)
        try {
            dataSource.connection.use { blocker ->
                blocker.autoCommit = false
                blocker.createStatement().use { it.execute("LOCK TABLE admin_action_commands IN SHARE MODE") }
                val futures = (1..2).map { pool.submit<AdminActionResult> {
                    commands.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash,
                        { transactions.add(jdbc.queryForObject("SELECT txid_current()", Long::class.java)!!) },
                        { executions.incrementAndGet(); action(actor, target) })
                } }
                try {
                    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(15)
                    while (jdbc.queryForObject("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'", Int::class.java)!! < 2) {
                        check(System.nanoTime() < deadline) { "Both claims must reach the real PostgreSQL insert" }; Thread.sleep(20)
                    }
                } finally { blocker.commit() }
                val results = futures.map { it.get(25, TimeUnit.SECONDS) }
                assertEquals(results[0], results[1]); assertEquals(1, executions.get())
                assertEquals(3, transactions.size); assertEquals(3, transactions.toSet().size)
                effects(target, Role.RUNNER, 1)
            }
        } finally { pool.shutdownNow() }
    }

    @Test fun `failed message transaction removes receipt decision and inbox before same-key retry`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        val key = IdempotencyKey(UUID.randomUUID().toString()); val hash = "a".repeat(64)
        assertFailsWith<IllegalStateException> {
            commands.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash, {}, {
                action(actor, target); error("simulated write failure")
            })
        }
        TransactionTemplate(manager).executeWithoutResult {
            assertEquals(0, count("admin_action_commands", "command_id", UUID.fromString(key.value)))
            effects(target, Role.RUNNER, 0)
        }
        val result = commands.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash, {}, { action(actor, target) })
        assertNotNull(result.messageId); effects(target, Role.RUNNER, 1)
    }

    @Test fun `completion failure rolls back already-written message and decision`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        val key = IdempotencyKey(UUID.randomUUID().toString())
        assertFailsWith<DataAccessException> { commands.execute(key, actor, AdminActionOperation.USER_MESSAGE,
            target, "a".repeat(64), {}, { action(actor, target); AdminActionResult(null, null, null) }) }
        TransactionTemplate(manager).executeWithoutResult {
            assertEquals(0, count("admin_action_commands", "command_id", UUID.fromString(key.value)))
            effects(target, Role.RUNNER, 0)
        }
    }

    @Test fun `database protects completed identity results state and retention`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER); val key = UUID.randomUUID().toString()
        assertEquals(200, send(actor, target, key).status)
        val id = UUID.fromString(key)
        for (assignment in listOf("actor_id = gen_random_uuid()", "operation = 'EVENT_CREATE'", "target_id = gen_random_uuid()",
            "input_hash = repeat('b',64)", "result_message_id = gen_random_uuid()", "state = 'PENDING'",
            "claim_owner = gen_random_uuid()", "intended_storage_key = 'new-key'", "completed_at = NULL")) {
            assertFailsWith<DataAccessException> { jdbc.update("UPDATE admin_action_commands SET $assignment WHERE command_id = ?", id) }
        }
        assertFailsWith<DataAccessException> { jdbc.update("DELETE FROM admin_action_commands WHERE command_id = ?", id) }
        assertEquals(200, send(actor, target, key).status); effects(target, Role.RUNNER, 1)
    }

    @Test fun `receipt states provide typed outcomes and never execute a new message`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        for ((state, code) in listOf("PENDING" to "IDEMPOTENT_REPLAY_IN_PROGRESS", "UNKNOWN" to "ADMIN_COMMAND_RECOVERY_REQUIRED", "FAILED" to "CONFLICT")) {
            val key = UUID.randomUUID()
            receipts.saveAndFlush(AdminActionCommand(commandId = key, actorId = actor, operation = AdminActionOperation.USER_MESSAGE,
                targetId = target, inputHash = commands.inputHash(listOf("Hello", "Message")),
                state = AdminActionCommandState.valueOf(state), claimedAt = java.time.Instant.now(), updatedAt = java.time.Instant.now()))
            val response = send(actor, target, key.toString())
            assertEquals(409, response.status); assertEquals(code, mapper.readTree(response.contentAsString)["errors"][0]["code"].asText())
        }
        effects(target, Role.RUNNER, 0)
    }

    @Test fun `winner committing after lookup but before claim still replays`() {
        val actor = user(Role.ADMIN); val target = user(Role.RUNNER)
        val key = IdempotencyKey(UUID.randomUUID().toString()); val hash = "a".repeat(64)
        val missed = CountDownLatch(1); val resume = CountDownLatch(1)
        val calls = AtomicInteger(); val loserActions = AtomicInteger()
        val delayedClock = object : Clock() {
            override fun getZone(): ZoneId = ZoneOffset.UTC
            override fun withZone(zone: ZoneId): Clock = this
            override fun instant(): Instant {
                if (calls.incrementAndGet() == 1) {
                    missed.countDown(); check(resume.await(15, TimeUnit.SECONDS))
                }
                return Instant.now()
            }
        }
        val delayed = AdminActionCommandService(receipts, manager, mapper, delayedClock)
        val pool = Executors.newSingleThreadExecutor()
        try {
            val loser = pool.submit<AdminActionResult> { delayed.execute(key, actor, AdminActionOperation.USER_MESSAGE,
                target, hash, {}, { loserActions.incrementAndGet(); action(actor, target) }) }
            assertTrue(missed.await(10, TimeUnit.SECONDS), "Loser must observe no receipt before winner commits")
            val winner = try { commands.execute(key, actor, AdminActionOperation.USER_MESSAGE, target, hash, {}, { action(actor, target) }) }
                finally { resume.countDown() }
            assertEquals(winner, loser.get(20, TimeUnit.SECONDS))
            assertEquals(0, loserActions.get()); effects(target, Role.RUNNER, 1)
        } finally { resume.countDown(); pool.shutdownNow() }
    }
}
