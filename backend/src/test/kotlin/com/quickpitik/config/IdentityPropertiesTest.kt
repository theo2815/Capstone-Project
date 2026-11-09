package com.quickpitik.config

import com.quickpitik.entity.SellerIdType
import org.junit.jupiter.api.Test
import org.springframework.boot.context.properties.bind.Binder
import org.springframework.boot.env.YamlPropertySourceLoader
import org.springframework.core.env.MapPropertySource
import org.springframework.core.env.StandardEnvironment
import org.springframework.core.io.ClassPathResource
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class IdentityPropertiesTest {

    @Test
    fun `an invalid id number format fails boot`() {
        assertFailsWith<IllegalArgumentException> { IdentityProperties(idNumberFormats = mapOf(SellerIdType.UMID to "[A-Z")) }
    }

    @Test
    fun `a blank id number secret fails boot`() {
        assertFailsWith<IllegalArgumentException> { IdentityProperties(idNumberHmacSecret = " ") }
    }

    @Test
    fun `application yml binds the secret and one format env per id type`() {
        val env = StandardEnvironment()
        env.propertySources.addFirst(MapPropertySource("test", mapOf("ID_NUMBER_FORMAT_DRIVERS_LICENSE" to "[A-Z]\\d{10}")))
        YamlPropertySourceLoader().load("yml", ClassPathResource("application.yml")).forEach { env.propertySources.addLast(it) }

        val bound = Binder.get(env).bind("app.identity", IdentityProperties::class.java).get()

        assertEquals("dev-only-id-number-hmac-secret", bound.idNumberHmacSecret)
        assertEquals(SellerIdType.entries.toSet(), bound.idNumberFormats.keys)
        assertEquals("[A-Z]\\d{10}", bound.idNumberFormats[SellerIdType.DRIVERS_LICENSE])
        assertTrue(bound.idNumberFormats.filterKeys { it != SellerIdType.DRIVERS_LICENSE }.values.all { it.isBlank() })
    }
}
