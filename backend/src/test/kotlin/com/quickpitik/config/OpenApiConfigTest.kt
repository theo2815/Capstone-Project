package com.quickpitik.config

import io.swagger.v3.oas.models.media.ObjectSchema
import io.swagger.v3.oas.models.media.StringSchema
import org.junit.jupiter.api.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

// The schema rewrite is the only real logic in OpenApiConfig, and it's the part
// that decides whether the generated docs describe the actual wire shape.
// springdoc reads the DECLARED return type, but ResponseEnvelopeAdvice wraps
// every controller result — so without this the spec would promise
// `AuthResponse` where the response really carries `{success, data, errors}`.
class OpenApiConfigTest {

    private val config = OpenApiConfig()

    @Test
    fun `a payload schema is wrapped in the envelope`() {
        val inner = ObjectSchema().addProperty("accessToken", StringSchema())

        val wrapped = config.envelope(inner)

        val props = assertNotNull(wrapped.properties)
        assertEquals(setOf("success", "data", "errors"), props.keys)
        assertSame(inner, props["data"])
    }

    // springdoc hands the same Schema instance to more than one operation, so
    // the customizer can see a schema it has already rewritten. Without this
    // guard the docs end up describing `data.data.data`.
    @Test
    fun `wrapping an already-wrapped schema is a no-op`() {
        val once = config.envelope(ObjectSchema().addProperty("id", StringSchema()))

        val twice = config.envelope(once)

        assertSame(once, twice)
    }

    // Endpoints that return Unit have no schema at all; they still answer with
    // an envelope, so they still need one.
    @Test
    fun `a bodyless response still gets an envelope`() {
        val wrapped = config.envelope(null)

        val props = assertNotNull(wrapped.properties)
        assertEquals(setOf("success", "data", "errors"), props.keys)
    }

    @Test
    fun `the errors array carries code, message and field`() {
        val wrapped = config.envelope(ObjectSchema())

        val errorItem = assertNotNull(wrapped.properties["errors"]?.items)
        assertEquals(setOf("code", "message", "field"), errorItem.properties.keys)
    }

    @Test
    fun `the bearer scheme is declared and required`() {
        val api = config.quickPitikOpenApi()

        assertNotNull(api.components.securitySchemes[OpenApiConfig.BEARER_SCHEME])
        assertTrue(api.security.any { it.containsKey(OpenApiConfig.BEARER_SCHEME) })
    }
}
