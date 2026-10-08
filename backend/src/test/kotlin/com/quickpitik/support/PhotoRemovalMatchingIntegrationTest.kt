package com.quickpitik.support

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.ai.FaceMatch
import com.quickpitik.dto.ai.FacesSearchResult
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import com.quickpitik.security.JwtTokenProvider
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.removal.*
import com.quickpitik.service.storage.StorageService
import com.quickpitik.service.storage.StoredObject
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentMatchers.*
import org.mockito.Mockito.*
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.boot.test.web.server.LocalServerPort
import org.springframework.http.HttpStatus
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.math.BigDecimal
import java.net.URI
import java.net.http.*
import java.time.LocalDate
import java.util.UUID
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import javax.imageio.ImageIO
import kotlin.test.*

// Supplemental unit boundaries for the behavior qualified by the integration RED above.
class PhotoRemovalMatchingUnitTest {
    private val requests = mock(PhotoRemovalRequestRepository::class.java)
    private val photos = mock(PhotoRepository::class.java)
    private val evidence = mock(PhotoRemovalEvidenceService::class.java)
    private val search = mock(com.quickpitik.service.photos.PhotoSearchService::class.java)
    private val storage = mock(StorageService::class.java)
    private val commands = mock(PhotoRemovalCommandRepository::class.java)
    private val manager = mock(org.springframework.transaction.PlatformTransactionManager::class.java)
    private val mapper = com.fasterxml.jackson.module.kotlin.jacksonObjectMapper().findAndRegisterModules()
    private val service = PhotoRemovalQueryService(requests, mock(PhotoRemovalItemRepository::class.java),
        mock(PhotoRemovalReviewRepository::class.java), mock(PhotoRemovalHistoryRepository::class.java), photos,
        mock(UserRepository::class.java), commands, evidence, search, storage, com.quickpitik.config.StorageProperties(),
        org.springframework.transaction.support.TransactionTemplate(manager), mapper, java.time.Clock.systemUTC(), mock(PhotoErasureJobRepository::class.java))

    init {
        `when`(manager.getTransaction(any() ?: org.springframework.transaction.support.DefaultTransactionDefinition()))
            .thenReturn(org.springframework.transaction.support.SimpleTransactionStatus())
    }

    @Test fun `gallery rejects an absent case before loading photos or storage`() {
        val id = UUID.randomUUID()
        `when`(requests.findById(id)).thenReturn(java.util.Optional.empty())
        assertEquals(404, assertFailsWith<com.quickpitik.exception.ApiException> {
            service.photos(UUID.randomUUID(), id, null, 0, 50)
        }.status.value())
        verifyNoInteractions(photos, storage, evidence, search)
    }

    @Test fun `stale matching is rejected before any evidence or provider read`() {
        val admin = UUID.randomUUID()
        val request = PhotoRemovalRequest(eventIdSnapshot = UUID.randomUUID(), eventNameSnapshot = "Synthetic event",
            requesterContact = "contact", supportReference = "ticket", requesterReason = "privacy", publicToOwnerReason = "privacy",
            creatorId = admin, creationCommandId = UUID.randomUUID(), version = 3)
        `when`(requests.findForUpdate(request.id)).thenReturn(request)
        assertEquals(409, assertFailsWith<com.quickpitik.exception.ApiException> {
            service.matches(admin, request.id, MatchRemovalCommand(UUID.randomUUID(), 2))
        }.status.value())
        verifyNoInteractions(evidence, search, storage, photos)
        assertNull(request.candidateCache)
    }
}

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = ["app.photo-removal.cleanup-interval=PT24H"])
class PhotoRemovalMatchingIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var service: PhotoRemovalService
    @Autowired lateinit var evidence: PhotoRemovalEvidenceService
    @Autowired lateinit var queries: PhotoRemovalQueryService
    @Autowired lateinit var requests: PhotoRemovalRequestRepository
    @Autowired lateinit var photos: PhotoRepository
    @Autowired lateinit var items: PhotoRemovalItemRepository
    @Autowired lateinit var selfies: UserSelfieRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var tokens: JwtTokenProvider
    @Autowired lateinit var mapper: ObjectMapper
    @Autowired lateinit var jdbc: JdbcTemplate
    @MockitoBean lateinit var storage: StorageService
    @MockitoBean lateinit var ai: AiApiClient
    @MockitoSpyBean lateinit var aiProperties: AiApiProperties
    @LocalServerPort var port = 0
    private val client = HttpClient.newHttpClient()
    private lateinit var admin: User
    private lateinit var event: Event
    private val selfie = ByteArrayOutputStream().apply {
        ImageIO.write(BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB), "png", this)
    }.toByteArray()

    @BeforeEach fun fixture() {
        admin = user(Role.ADMIN)
        event = event()
        doReturn(true).`when`(aiProperties).enabled
        doReturn(true).`when`(aiProperties).searchFallbackOnError
        `when`(storage.put(anyString(), any(ByteArray::class.java) ?: byteArrayOf(), anyString())).thenAnswer {
            outsideTransaction()
            StoredObject(it.getArgument(0), it.getArgument<ByteArray>(1).size.toLong(), it.getArgument(2))
        }
        `when`(storage.open(anyString())).thenAnswer { outsideTransaction(); selfie.inputStream() }
        `when`(storage.exists(anyString())).thenAnswer { outsideTransaction(); true }
        `when`(storage.presignedGetUrl(anyString(), any() ?: java.time.Duration.ZERO)).thenAnswer {
            outsideTransaction(); "https://synthetic.test/preview"
        }
        doAnswer { outsideTransaction(); null }.`when`(storage).delete(anyString())
        provider(FacesSearchResult())
    }

    @Test fun `adminCandidatesIncludeAllOwnersInOneEvent`() {
        val ownerA = user(Role.PHOTOGRAPHER)
        val ownerB = user(Role.PHOTOGRAPHER)
        val ownerC = user(Role.PHOTOGRAPHER)
        val a = photo(ownerA, "a")
        val b = photo(ownerB, "b", PhotoStatus.HIDDEN)
        val c = photo(ownerC, "c")
        jdbc.update("UPDATE photos SET deleted_at = now(), removal_state = 'REVIEW' WHERE id = ?", c.id)
        provider(FacesSearchResult(listOf(FaceMatch("a", similarity = 0.81), FaceMatch("b", similarity = 0.92), FaceMatch("c", similarity = 0.73))))
        val case = case()
        val response = match(case)
        assertEquals(setOf(ownerA.id, ownerB.id, ownerC.id), response["items"].map { UUID.fromString(it["photographerId"].asText()) }.toSet())
        assertEquals(setOf(a.id, b.id, c.id), ids(response))
        assertEquals(0.92, response["items"].single { it["photo"]["id"].asText() == b.id.toString() }["matchScore"].asDouble())
        assertTrue(response["items"].all { it["matchSource"].asText() == "SELFIE" && it["photo"]["cleanUrl"].isNull && it["photo"]["downloadUrl"].isNull })
        assertEquals(0L, selfies.countByUserId(admin.id))
        assertTrue(items.findByRequestIdOrderByIdAsc(case.id).isEmpty(), "discovery must not freeze selected items or owners")
        assertFalse(response.toString().contains("photo-removal/"))
        for (role in listOf(Role.RUNNER, Role.PHOTOGRAPHER)) {
            assertEquals(403, post(case.id, UUID.randomUUID(), case.version, user(role)).statusCode())
            assertEquals(403, get(case.id, actor = user(role)).statusCode())
        }
    }

    @Test fun `fallbackGalleryIsNeverARemovalMatch`() {
        photo(user(Role.PHOTOGRAPHER), "fallback")
        val case = case()
        doAnswer { outsideTransaction(); throw AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "synthetic outage") }
            .`when`(ai).facesSearch(any(ByteArray::class.java) ?: byteArrayOf(), anyString(), anyString(), any(UUID::class.java) ?: event.id, anyDouble(), anyInt())
        val response = post(case.id, UUID.randomUUID(), case.version)
        assertEquals(503, response.statusCode())
        assertEquals("AI_API_UNAVAILABLE", mapper.readTree(response.body())["errors"][0]["code"].asText())
        assertEquals(case.version, requests.findById(case.id).orElseThrow().version)
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM photo_removal_commands WHERE request_id = ? AND operation = 'MATCH_CANDIDATES'", Int::class.java, case.id))
    }

    @Test fun `disabled AI is explicitly unavailable`() {
        val case = case()
        doReturn(false).`when`(aiProperties).enabled
        assertEquals(503, post(case.id, UUID.randomUUID(), case.version).statusCode())
        verify(ai, never()).facesSearch(any(ByteArray::class.java) ?: byteArrayOf(), anyString(), anyString(), any(UUID::class.java) ?: event.id, anyDouble(), anyInt())
    }

    @Test fun `crossEventMatchIsDiscarded`() {
        val owner = user(Role.PHOTOGRAPHER)
        val local = photo(owner, "same-person")
        val foreign = photos.saveAndFlush(Photo(eventId = event().id, photographerId = owner.id,
            s3Key = "synthetic/${UUID.randomUUID()}.jpg", pricePhp = BigDecimal("50.00"), facePersons = mutableSetOf(PhotoFacePersonEmbed(0, "same-person"))))
        provider(FacesSearchResult(listOf(FaceMatch("same-person", similarity = 0.9))))
        val response = match(case())
        assertEquals(setOf(local.id), ids(response))
        assertFalse(foreign.id in ids(response))
    }

    @Test fun `candidateLimitIsVisible`() {
        val owner = user(Role.PHOTOGRAPHER)
        photo(owner, "p-0")
        provider(FacesSearchResult(List(50) { FaceMatch("p-$it", similarity = if (it == 0) 0.9 else 0.1) }))
        val result = match(case())
        assertEquals(50, result["candidateLimit"].asInt())
        assertTrue(result["possiblyTruncated"].asBoolean())
        assertEquals(1, result["items"].size())
    }

    @Test fun `no face differs from zero qualifying matches and provider errors`() {
        val case = case()
        provider(mapper.readValue("{\"matches\":[],\"noFaces\":true}", FacesSearchResult::class.java))
        val noFace = post(case.id, UUID.randomUUID(), case.version)
        assertEquals(422, noFace.statusCode())
        assertEquals("NO_FACES", mapper.readTree(noFace.body())["errors"][0]["code"].asText())
        provider(FacesSearchResult())
        val none = match(case)
        assertTrue(none["items"].isEmpty)
        assertEquals(50, none["candidateLimit"].asInt())
        assertFalse(none["possiblyTruncated"].asBoolean())
        providerError(AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "LOW_QUALITY", "internal provider text"))
        val invalid = post(case.id, UUID.randomUUID(), current(case.id).version)
        assertEquals(422, invalid.statusCode())
        assertEquals("LOW_QUALITY", mapper.readTree(invalid.body())["errors"][0]["code"].asText())
        assertFalse(invalid.body().contains("internal provider text"))
    }

    @Test fun `gallery bib filtering pagination orphan owner and absent bytes are server owned`() {
        val owner = user(Role.PHOTOGRAPHER)
        val a = photo(owner, "a", PhotoStatus.HIDDEN, "7202")
        val b = photo(null, "b", bib = "7203")
        photo(owner, "c", bib = "9999")
        jdbc.update("UPDATE photos SET deleted_at = now(), removal_state = 'REVIEW' WHERE id = ?", a.id)
        `when`(storage.exists(b.s3Key)).thenAnswer { outsideTransaction(); false }
        val case = case()
        val first = page(get(case.id, "?bib=720&limit=1&offset=0"))
        val second = page(get(case.id, "?bib=720&limit=1&offset=1"))
        assertEquals(2L, first["total"].asLong())
        assertEquals(1, first["limit"].asInt())
        assertEquals(1, second["offset"].asInt())
        assertEquals(setOf(a.id, b.id), ids(first) + ids(second))
        val orphan = (first["items"].toList() + second["items"].toList()).single { it["photo"]["id"].asText() == b.id.toString() }
        assertTrue(orphan["photographerId"].isNull)
        assertFalse(orphan["bytesAvailable"].asBoolean())
        assertTrue(orphan["photo"]["imageUrl"].isNull)
        assertEquals(100, page(get(case.id, "?limit=500&offset=-1"))["limit"].asInt())
        assertEquals(404, get(UUID.randomUUID()).statusCode())
    }

    @Test fun `cache stores only latest server provenance and exact committed version`() {
        val photo = photo(user(Role.PHOTOGRAPHER), "candidate")
        provider(FacesSearchResult(listOf(FaceMatch("candidate", similarity = 0.88))))
        val case = case()
        val result = match(case)
        val latest = current(case.id)
        val cache = cache(case.id)
        assertEquals(latest.version, result["requestVersion"].asLong())
        assertEquals(latest.version, cache["requestVersion"].asLong())
        assertEquals(latest.evidenceGeneration, cache["evidenceGeneration"].asLong())
        assertEquals(setOf("requestVersion", "evidenceGeneration", "candidates"), cache.fieldNames().asSequence().toSet())
        assertEquals(setOf("photoId", "score", "source"), cache["candidates"][0].fieldNames().asSequence().toSet())
        assertEquals(photo.id.toString(), cache["candidates"][0]["photoId"].asText())
        assertEquals(0.88, cache["candidates"][0]["score"].asDouble())
        assertEquals("SELFIE", cache["candidates"][0]["source"].asText())
        assertFalse(cache.toString().contains("candidate\""))
        provider(FacesSearchResult())
        match(latest)
        assertTrue(cache(case.id)["candidates"].isEmpty)
        assertEquals(current(case.id).version, cache(case.id)["requestVersion"].asLong())
    }

    @Test fun `matching replay preserves its original result without new inference or cache overwrite`() {
        val photo = photo(user(Role.PHOTOGRAPHER), "replay")
        provider(FacesSearchResult(listOf(FaceMatch("replay", similarity = 0.9))))
        val case = case()
        val command = UUID.randomUUID()
        val original = match(case, command)
        clearInvocations(ai, storage)
        assertEquals(original, match(case, command))
        verifyNoInteractions(ai, storage)
        assertEquals(409, post(case.id, command, case.version + 1).statusCode())
        assertEquals(409, post(case.id, command, case.version, user(Role.ADMIN)).statusCode())
        assertEquals(409, post(case.id, UUID.randomUUID(), case.version).statusCode())
        val beforeReplace = current(case.id)
        evidence.replace(admin.id, case.id, UUID.randomUUID(), beforeReplace.version, image())
        assertTrue(cache(case.id).isNull)
        clearInvocations(ai, storage)
        assertEquals(original, match(case, command))
        verifyNoInteractions(ai, storage)
        assertTrue(cache(case.id).isNull)
        assertEquals(setOf(photo.id), ids(original))
    }

    @Test fun `cancellation invalidates cached discovery`() {
        val case = case()
        match(case)
        assertFalse(cache(case.id).isNull)
        service.cancel(admin.id, case.id, CancelRemovalCommand(UUID.randomUUID(), current(case.id).version, "withdraw"))
        assertTrue(cache(case.id).isNull)
        assertEquals(409, post(case.id, UUID.randomUUID(), current(case.id).version).statusCode())
    }

    @Test fun `replacement racing inference cannot install earlier evidence candidates`() = inferenceRace { case ->
        val latest = current(case.id)
        evidence.replace(admin.id, case.id, UUID.randomUUID(), latest.version, image())
    }

    @Test fun `cancellation racing inference cannot install candidates`() = inferenceRace { case ->
        service.cancel(admin.id, case.id, CancelRemovalCommand(UUID.randomUUID(), current(case.id).version, "withdraw"))
    }

    @Test fun `two matching commands with one version cannot both install candidates`() {
        val case = case()
        match(case) // The first RED assertion is feature behavior, before starting workers.
        val version = current(case.id).version
        val entered = CountDownLatch(2)
        val release = CountDownLatch(1)
        providerAnswer { entered.countDown(); assertTrue(release.await(20, TimeUnit.SECONDS)); FacesSearchResult() }
        val pool = Executors.newFixedThreadPool(2)
        try {
            val calls = List(2) { pool.submit<HttpResponse<String>> { post(case.id, UUID.randomUUID(), version) } }
            assertTrue(entered.await(20, TimeUnit.SECONDS))
            release.countDown()
            assertEquals(listOf(200, 409), calls.map { it.get(20, TimeUnit.SECONDS).statusCode() }.sorted())
            assertEquals(current(case.id).version, cache(case.id)["requestVersion"].asLong())
        } finally { release.countDown(); pool.shutdownNow() }
    }

    @Test fun `missing evidence bytes cannot install a match or leak storage error`() {
        val case = case()
        `when`(storage.open(anyString())).thenThrow(IllegalStateException("private evidence key failure"))
        val result = post(case.id, UUID.randomUUID(), case.version)
        assertEquals(503, result.statusCode())
        assertFalse(result.body().contains("private evidence key"))
        assertEquals(case.version, current(case.id).version)
        verify(ai, never()).facesSearch(any(ByteArray::class.java) ?: byteArrayOf(), anyString(), anyString(), any(UUID::class.java) ?: event.id, anyDouble(), anyInt())
    }

    private fun inferenceRace(change: (PhotoRemovalRequestDto) -> Unit) {
        val photo = photo(user(Role.PHOTOGRAPHER), "racing")
        val case = case()
        match(case)
        val version = current(case.id).version
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        providerAnswer {
            entered.countDown(); assertTrue(release.await(20, TimeUnit.SECONDS))
            FacesSearchResult(listOf(FaceMatch("racing", similarity = 0.9)))
        }
        val pool = Executors.newSingleThreadExecutor()
        try {
            val call = pool.submit<HttpResponse<String>> { post(case.id, UUID.randomUUID(), version) }
            assertTrue(entered.await(20, TimeUnit.SECONDS))
            change(case)
            release.countDown()
            assertEquals(409, call.get(20, TimeUnit.SECONDS).statusCode())
            assertTrue(cache(case.id).isNull)
            assertTrue(items.findByRequestIdOrderByIdAsc(case.id).none { it.photoIdSnapshot == photo.id })
        } finally { release.countDown(); pool.shutdownNow() }
    }

    private fun outsideTransaction() = assertFalse(TransactionSynchronizationManager.isActualTransactionActive(), "storage and AI must run outside a DB transaction")
    private fun provider(result: FacesSearchResult) = providerAnswer { result }
    private fun providerError(error: AiApiException) = providerAnswer { throw error }
    private fun providerAnswer(result: () -> FacesSearchResult) {
        doAnswer {
            outsideTransaction()
            assertEquals(event.id, it.getArgument<UUID>(3))
            assertEquals(aiProperties.faceMatchThresholdDefault, it.getArgument<Double>(4))
            assertEquals(50, it.getArgument<Int>(5))
            assertContentEquals(selfie, it.getArgument<ByteArray>(0))
            result()
        }.`when`(ai).facesSearch(any(ByteArray::class.java) ?: byteArrayOf(), anyString(), anyString(), any(UUID::class.java) ?: event.id, anyDouble(), anyInt())
    }
    private fun image() = SearchImage(selfie, "image/png", "synthetic.png")
    private fun case(): PhotoRemovalRequestDto {
        val case = service.create(admin.id, CreateRemovalRequestCommand(event.id, "ticket-${UUID.randomUUID()}", "synthetic contact", reason = "privacy", commandId = UUID.randomUUID()))
        return evidence.replace(admin.id, case.id, UUID.randomUUID(), case.version, image())
    }
    private fun current(id: UUID) = queries.adminDetail(admin.id, id)
    private fun event() = events.saveAndFlush(Event(slug = "matching-${UUID.randomUUID()}", name = "Synthetic event", date = LocalDate.of(2026, 10, 6), location = "Cebu", status = EventStatus.ACTIVE, visibility = EventVisibility.UNLISTED))
    private fun user(role: Role) = users.saveAndFlush(User(email = "matching-${UUID.randomUUID()}@test.local", passwordHash = "unused", name = "Synthetic $role", role = role))
    private fun photo(owner: User?, person: String, status: PhotoStatus = PhotoStatus.LIVE, bib: String = "183") = photos.saveAndFlush(Photo(
        eventId = event.id, photographerId = owner?.id, s3Key = "synthetic/${UUID.randomUUID()}.jpg", thumbnailS3Key = "synthetic/${UUID.randomUUID()}-preview.jpg",
        status = status, pricePhp = BigDecimal("50.00"), facePersons = mutableSetOf(PhotoFacePersonEmbed(0, person)), bibs = mutableSetOf(PhotoBibEmbed(bib, BigDecimal("0.9")))))
    private fun cache(id: UUID): JsonNode = jdbc.queryForObject("SELECT candidate_cache::text FROM photo_removal_requests WHERE id = ?", String::class.java, id)?.let { mapper.readTree(it) } ?: mapper.nullNode()
    private fun ids(result: JsonNode) = result["items"].map { UUID.fromString(it["photo"]["id"].asText()) }.toSet()
    private fun match(case: PhotoRemovalRequestDto, command: UUID = UUID.randomUUID()) = page(post(case.id, command, case.version))
    private fun page(response: HttpResponse<String>): JsonNode {
        assertEquals(200, response.statusCode(), response.body())
        assertTrue(response.headers().firstValue("Cache-Control").orElse("").contains("no-store"))
        return mapper.readTree(response.body())["data"]
    }
    private fun get(id: UUID, query: String = "", actor: User = admin) = send(HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/admin/photo-removal-requests/$id/photos$query")).GET(), actor)
    private fun post(id: UUID, command: UUID, version: Long, actor: User = admin) = send(HttpRequest.newBuilder(URI("http://localhost:$port/api/v1/admin/photo-removal-requests/$id/matches"))
        .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(mapOf("commandId" to command, "expectedVersion" to version)))), actor)
    private fun send(builder: HttpRequest.Builder, actor: User) = client.send(builder.header("Authorization", "Bearer ${tokens.createAccessToken(actor)}").build(), HttpResponse.BodyHandlers.ofString())
}
