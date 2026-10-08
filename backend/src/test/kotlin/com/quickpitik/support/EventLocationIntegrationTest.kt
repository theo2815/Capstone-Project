package com.quickpitik.support

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ResponseEnvelopeAdvice
import com.quickpitik.controller.AdminEventsController
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.events.EventDtoMapper
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.CsvSource
import org.mockito.Mockito
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.dao.DataAccessException
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.mock.web.MockMultipartFile
import org.springframework.mock.web.MockHttpServletResponse
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.context.bean.override.mockito.MockitoBean
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import java.util.UUID
import kotlin.test.*

class EventLocationIntegrationTest : PostgresIntegrationTest() {
    @Autowired lateinit var admin: AdminEventService
    @Autowired lateinit var users: UserRepository
    @Autowired lateinit var events: EventRepository
    @Autowired lateinit var dtos: EventDtoMapper
    @Autowired lateinit var json: ObjectMapper
    @Autowired lateinit var jdbc: JdbcTemplate
    @MockitoBean lateinit var storage: StorageService
    private lateinit var actor: User

    @BeforeEach fun setUp() {
        actor = users.save(User(email = "location-${UUID.randomUUID()}@test.local", name = "Admin",
            passwordHash = "unused", role = Role.ADMIN))
        Mockito.`when`(storage.put(Mockito.anyString(), anyArg(), Mockito.anyString())).thenAnswer {
            com.quickpitik.service.storage.StoredObject(it.getArgument(0), it.getArgument<ByteArray>(1).size.toLong(), "image/jpeg")
        }
    }

    private fun <T> anyArg(): T = Mockito.any()

    private fun request(region: String? = null, province: String? = null, id: UUID? = null,
        key: UUID? = null, cover: Boolean = false, title: String = "Location run"): MockHttpServletResponse {
        val principal = AuthPrincipal(actor.id, actor.email, Role.ADMIN)
        SecurityContextHolder.getContext().authentication =
            UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
        try {
            val mvc = MockMvcBuilders.standaloneSetup(AdminEventsController(admin))
                .setControllerAdvice(GlobalExceptionHandler(), ResponseEnvelopeAdvice())
                .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver()).build()
            val form = multipart(if (id == null) "/api/v1/admin/events" else "/api/v1/admin/events/$id")
            form.param("title", title).param("date", "2026-11-08").param("location", "Venue, Cebu City")
                .param("pricePerPhoto", "25.00")
            if (id != null) form.with { it.method = "PATCH"; it }
            if (region != null) form.param("regionCode", region)
            if (province != null) form.param("provinceCode", province)
            if (key != null) form.header("Idempotency-Key", key)
            if (cover) {
                val image = java.awt.image.BufferedImage(40, 30, java.awt.image.BufferedImage.TYPE_INT_RGB)
                val bytes = java.io.ByteArrayOutputStream().also { javax.imageio.ImageIO.write(image, "png", it) }.toByteArray()
                form.file(MockMultipartFile("cover", "cover.png", "image/png", bytes))
            }
            return mvc.perform(form).andReturn().response
        } finally { SecurityContextHolder.clearContext() }
    }

    private fun data(response: MockHttpServletResponse): JsonNode {
        assertEquals(200, response.status, response.contentAsString)
        return json.readTree(response.contentAsString)["data"]
    }
    private fun id(data: JsonNode): UUID = UUID.fromString(data["id"].asText())
    private fun pair(node: JsonNode, region: String?, province: String?) {
        assertTrue(node.has("regionCode"), node.toString())
        assertTrue(node.has("provinceCode"), node.toString())
        assertEquals(region, node["regionCode"].takeUnless { it.isNull }?.asText())
        assertEquals(province, node["provinceCode"].takeUnless { it.isNull }?.asText())
    }
    private fun decisions(id: UUID): Int = jdbc.queryForObject(
        "SELECT count(*) FROM admin_decision_log WHERE target_event_id = ?", Int::class.java, id)!!

    @Test fun `valid trimmed pair persists maps everywhere and identical or omitted patch is a no-op`() {
        val created = data(request(" region-7 ", " cebu "))
        val eventId = id(created)
        pair(created, "region-7", "cebu")
        val event = events.findById(eventId).orElseThrow()
        pair(json.valueToTree(dtos.toListDto(event)), "region-7", "cebu")
        pair(json.valueToTree(dtos.toDetailDto(event)), "region-7", "cebu")
        assertEquals("Venue, Cebu City", event.location)
        assertEquals("Cebu City", created["city"].asText())
        val revision = created["reviewRevision"].asLong()
        for ((r, p) in listOf(null to null, "region-7" to "cebu")) {
            val unchanged = data(request(r, p, eventId))
            pair(unchanged, "region-7", "cebu")
            assertEquals(revision, unchanged["reviewRevision"].asLong())
            assertEquals(1, decisions(eventId))
        }
        val changed = data(request("ncr", "metro-manila", eventId))
        pair(changed, "ncr", "metro-manila")
        val history = changed["adminOverrides"].last()
        pair(history["before"], "region-7", "cebu")
        pair(history["after"], "ncr", "metro-manila")
        assertEquals(revision + 1, changed["reviewRevision"].asLong())
        assertEquals(2, decisions(eventId))
    }

    @Test fun `legacy null pair stays readable and editable without classification`() {
        val created = data(request())
        pair(created, null, null)
        val eventId = id(created)
        val changed = data(request(id = eventId, title = "Renamed legacy run"))
        pair(changed, null, null)
        val event = events.findById(eventId).orElseThrow()
        pair(json.valueToTree(dtos.toListDto(event)), null, null)
        pair(json.valueToTree(dtos.toDetailDto(event)), null, null)
        assertEquals("Renamed legacy run", event.name)
    }

    @ParameterizedTest
    @CsvSource("region-7,NULL", "NULL,cebu", "'',cebu", "region-7,''", "'   ','   '",
        "unknown,cebu", "region-7,metro-manila", "REGION-7,cebu", "region-7,unknown", nullValues = ["NULL"])
    fun `invalid pair rejects create and patch before any receipt or storage work`(region: String?, province: String?) {
        val eventId = id(data(request("region-7", "cebu")))
        for (target in listOf(null, eventId)) {
            val key = UUID.randomUUID()
            val result = request(region, province, target, key, cover = true)
            assertEquals(400, result.status, result.contentAsString)
            assertEquals("INVALID_REGION", json.readTree(result.contentAsString)["errors"][0]["code"].asText())
            assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM admin_action_commands WHERE command_id = ?", Int::class.java, key))
        }
        pair(json.valueToTree(dtos.toAdminListDto(events.findById(eventId).orElseThrow())), "region-7", "cebu")
        assertEquals(1, decisions(eventId))
        Mockito.verifyNoInteractions(storage)
    }

    @Test fun `both code lengths are bounded at the multipart boundary`() {
        for ((r, p) in listOf("r".repeat(21) to "cebu", "region-7" to "p".repeat(41))) {
            val result = request(r, p)
            assertEquals(400, result.status, result.contentAsString)
        }
        Mockito.verifyNoInteractions(storage)
    }

    @Test fun `create receipts reject changed classification and replay normalized pairs`() {
        val createKey = UUID.randomUUID()
        val created = data(request("region-7", "cebu", key = createKey))
        val eventId = id(created)
        assertEquals(eventId, id(data(request(" region-7 ", " cebu ", key = createKey))))
        assertEquals(409, request("ncr", "metro-manila", key = createKey).status)
        assertEquals(1, decisions(eventId))
    }

    @Test fun `update receipts reject changed classification and replay normalized pairs`() {
        val eventId = id(data(request("region-7", "cebu")))
        val updateKey = UUID.randomUUID()
        data(request("ncr", "metro-manila", eventId, updateKey))
        assertEquals(409, request("region-7", "cebu", eventId, updateKey).status)
        pair(data(request(" ncr ", " metro-manila ", eventId, updateKey)), "ncr", "metro-manila")
        assertEquals(2, decisions(eventId))
    }

    @Test fun `old omitted-pair receipt fingerprint still replays after location support`() {
        val key = UUID.randomUUID()
        val eventId = id(data(request(key = key)))
        val oldFields = listOf("Location run", "2026-11-08", "Venue, Cebu City", "25.00", "", "", "")
        val oldHash = java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256")
            .digest(json.writeValueAsBytes(oldFields)))
        jdbc.update("UPDATE admin_action_commands SET input_hash = ? WHERE command_id = ?", oldHash, key)
        assertEquals(eventId, id(data(request(key = key))))
        assertEquals(409, request("region-7", "cebu", key = key).status)
        assertEquals(1, decisions(eventId))
    }

    @Test fun `database rejects half and blank pairs without altering the legacy row`() {
        val eventId = id(data(request()))
        for ((r, p) in listOf("region-7" to null, null to "cebu", " " to "cebu", "region-7" to "")) {
            val error = assertFailsWith<DataAccessException> {
                jdbc.update("UPDATE events SET region_code = ?, province_code = ? WHERE id = ?", r, p, eventId)
            }
            assertEquals("23514", (error.mostSpecificCause as java.sql.SQLException).sqlState)
        }
        pair(json.valueToTree(dtos.toAdminListDto(events.findById(eventId).orElseThrow())), null, null)
    }

    @Test fun `failed audit rolls back the location pair revision and override together`() {
        val eventId = id(data(request("region-7", "cebu")))
        jdbc.execute("""CREATE FUNCTION reject_location_audit() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN IF NEW.target_event_id = '$eventId'::uuid THEN RAISE EXCEPTION 'location audit failure'; END IF;
            RETURN NEW; END $$""")
        jdbc.execute("CREATE TRIGGER reject_location_audit BEFORE INSERT ON admin_decision_log FOR EACH ROW EXECUTE FUNCTION reject_location_audit()")
        try {
            assertEquals(500, request("ncr", "metro-manila", eventId).status)
            val stored = json.valueToTree<JsonNode>(dtos.toAdminListDto(events.findById(eventId).orElseThrow()))
            pair(stored, "region-7", "cebu")
            assertEquals(0, stored["reviewRevision"].asInt())
            assertTrue(stored["adminOverrides"].isEmpty)
            assertEquals(1, decisions(eventId))
        } finally {
            jdbc.execute("DROP TRIGGER reject_location_audit ON admin_decision_log")
            jdbc.execute("DROP FUNCTION reject_location_audit()")
        }
    }
}
