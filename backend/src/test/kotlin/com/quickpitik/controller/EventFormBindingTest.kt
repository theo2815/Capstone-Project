package com.quickpitik.controller

import com.quickpitik.dto.admin.CreateAdminEventRequest
import com.quickpitik.dto.photographer.CreateMyEventRequest
import com.quickpitik.dto.photographer.UpdateMyEventRequest
import com.quickpitik.entity.Role
import com.quickpitik.exception.GlobalExceptionHandler
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.admin.AdminEventService
import com.quickpitik.service.photographer.PhotoUploadService
import com.quickpitik.service.photographer.PhotographerBundleService
import com.quickpitik.service.photographer.PhotographerEventService
import com.quickpitik.service.photographer.PhotographerOwnedEventService
import com.quickpitik.service.photographer.PhotographerPhotoService
import com.quickpitik.service.ratelimit.RateLimiter
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.http.HttpMethod
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken
import org.springframework.security.core.context.SecurityContextHolder
import org.springframework.security.web.method.annotation.AuthenticationPrincipalArgumentResolver
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.status
import org.springframework.test.web.servlet.setup.MockMvcBuilders
import java.math.BigDecimal
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// P1.7 (2026-09-12): both event multipart forms bind to a `@Valid
// @ModelAttribute` DTO instead of nine loose `@RequestParam`s plus hand-rolled
// blank checks and a duplicated price parser. The controller's
// StringTrimmerEditor is what keeps the old contract — FormData sends every
// field as a string and an untouched input as "", so blank has to read as
// "absent" and fall back to the DTO default (create) or "no change" (patch).
// These run through the real Spring binding path because that is the part that
// changed; a plain unit call on the handler would prove nothing.
//
// The forms are recorded with doAnswer rather than an ArgumentCaptor: Kotlin
// null-checks the platform-typed `capture()` at the call site and blows up
// before Mockito sees it.
class EventFormBindingTest {

    private val ownedEvents = Mockito.mock(PhotographerOwnedEventService::class.java)
    private val adminEvents = Mockito.mock(AdminEventService::class.java)
    private val rateLimiter = Mockito.mock(RateLimiter::class.java)
    private var created: CreateMyEventRequest? = null
    private var updated: UpdateMyEventRequest? = null
    private var adminCreated: CreateAdminEventRequest? = null
    private lateinit var photographer: MockMvc
    private lateinit var admin: MockMvc

    @BeforeEach
    fun setUp() {
        Mockito.`when`(rateLimiter.tryAcquire(anyArg(), anyArg()))
            .thenReturn(RateLimiter.Decision(allowed = true))
        Mockito.`when`(ownedEvents.create(anyArg(), anyArg(), anyArg()))
            .thenAnswer { created = it.getArgument(1); null }
        Mockito.`when`(ownedEvents.update(anyArg(), anyArg(), anyArg(), anyArg()))
            .thenAnswer { updated = it.getArgument(2); null }
        Mockito.`when`(adminEvents.create(anyArg(), anyArg(), anyArg()))
            .thenAnswer { adminCreated = it.getArgument(1); null }
        photographer = mvc(
            MePhotographerController(
                Mockito.mock(PhotographerEventService::class.java),
                ownedEvents,
                Mockito.mock(PhotographerPhotoService::class.java),
                Mockito.mock(PhotographerBundleService::class.java),
                Mockito.mock(PhotoUploadService::class.java),
                rateLimiter,
                Mockito.mock(com.quickpitik.service.photographer.PhotoMetadataSettingsService::class.java),
            ),
        )
        admin = mvc(AdminEventsController(adminEvents))
        val principal = AuthPrincipal(UUID.randomUUID(), "photographer@test.local", Role.PHOTOGRAPHER)
        SecurityContextHolder.getContext().authentication =
            UsernamePasswordAuthenticationToken(principal, null, principal.authorities)
    }

    @AfterEach
    fun clearContext() = SecurityContextHolder.clearContext()

    @Test
    fun `a photographer create form arrives trimmed, with blanks back on the DTO defaults`() {
        photographer.perform(
            multipart("/api/v1/me/photographer/events")
                .param("title", "  Cebu Marathon  ")
                .param("date", " 2026-11-08 ")
                .param("location", "  Cebu City ")
                .param("organizerName", "")
                .param("description", "  ")
                .param("visibility", "")
                .param("pricingMode", "")
                .param("pricePerPhoto", "250.00")
                .param("watermarkPolicy", ""),
        ).andExpect(status().isOk)

        val form = assertNotNull(created)
        assertEquals("Cebu Marathon", form.title)
        assertEquals("2026-11-08", form.date)
        assertEquals("Cebu City", form.location)
        assertNull(form.organizerName)
        assertNull(form.description)
        assertEquals("public", form.visibility)
        assertEquals("paid", form.pricingMode)
        assertEquals(0, BigDecimal("250.00").compareTo(form.pricePerPhoto))
        assertNull(form.watermarkPolicy)
    }

    @Test
    fun `a blank title is a 400 against the title field, and the service is never reached`() {
        photographer.perform(
            multipart("/api/v1/me/photographer/events")
                .param("title", "   ")
                .param("date", "2026-11-08")
                .param("location", "Cebu City"),
        )
            .andExpect(status().isBadRequest)
            .andExpect(jsonPath("$.errors[0].code").value("VALIDATION_ERROR"))
            .andExpect(jsonPath("$.errors[0].field").value("title"))

        assertNull(created)
    }

    @Test
    fun `a price that is not a number is a 400 against the price field`() {
        photographer.perform(
            multipart("/api/v1/me/photographer/events")
                .param("title", "Cebu Marathon")
                .param("date", "2026-11-08")
                .param("location", "Cebu City")
                .param("pricePerPhoto", "two hundred"),
        )
            .andExpect(status().isBadRequest)
            .andExpect(jsonPath("$.errors[0].code").value("VALIDATION_ERROR"))
            .andExpect(jsonPath("$.errors[0].field").value("pricePerPhoto"))
            .andExpect(jsonPath("$.errors[0].message").value("Invalid value for 'pricePerPhoto'"))

        assertNull(created)
    }

    @Test
    fun `a patch whose pricing fields are blank is not a pricing edit`() {
        val eventId = UUID.randomUUID()
        photographer.perform(
            multipart(HttpMethod.PATCH, "/api/v1/me/photographer/events/$eventId")
                .param("title", " Renamed Run ")
                .param("pricingMode", "")
                .param("pricePerPhoto", "")
                .param("watermarkPolicy", ""),
        ).andExpect(status().isOk)

        val form = assertNotNull(updated)
        assertEquals("Renamed Run", form.title)
        assertNull(form.pricingMode)
        assertNull(form.pricePerPhoto)
        assertNull(form.watermarkPolicy)
    }

    @Test
    fun `an admin create with no price still defaults to zero`() {
        admin.perform(
            multipart("/api/v1/admin/events")
                .param("title", "Admin Run")
                .param("date", "2026-11-08")
                .param("location", "Mandaue")
                .param("pricePerPhoto", ""),
        ).andExpect(status().isOk)

        assertEquals(0, BigDecimal.ZERO.compareTo(assertNotNull(adminCreated).pricePerPhoto))
    }

    @Test
    fun `an admin create with a blank location is a 400 against the location field`() {
        admin.perform(
            multipart("/api/v1/admin/events")
                .param("title", "Admin Run")
                .param("date", "2026-11-08")
                .param("location", " "),
        )
            .andExpect(status().isBadRequest)
            .andExpect(jsonPath("$.errors[0].code").value("VALIDATION_ERROR"))
            .andExpect(jsonPath("$.errors[0].field").value("location"))

        assertNull(adminCreated)
    }

    private fun mvc(controller: Any): MockMvc =
        MockMvcBuilders.standaloneSetup(controller)
            .setCustomArgumentResolvers(AuthenticationPrincipalArgumentResolver())
            .setControllerAdvice(GlobalExceptionHandler())
            .build()

    private fun <T> anyArg(): T = Mockito.any()
}
