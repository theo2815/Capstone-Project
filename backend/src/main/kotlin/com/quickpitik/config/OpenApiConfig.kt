package com.quickpitik.config

import io.swagger.v3.oas.models.Components
import io.swagger.v3.oas.models.OpenAPI
import io.swagger.v3.oas.models.info.Info
import io.swagger.v3.oas.models.media.ArraySchema
import io.swagger.v3.oas.models.media.BooleanSchema
import io.swagger.v3.oas.models.media.ObjectSchema
import io.swagger.v3.oas.models.media.Schema
import io.swagger.v3.oas.models.media.StringSchema
import io.swagger.v3.oas.models.security.SecurityRequirement
import io.swagger.v3.oas.models.security.SecurityScheme
import org.springdoc.core.customizers.OpenApiCustomizer
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

/**
 * OpenAPI 3 description of the public API, generated from the controllers and
 * served at `/swagger-ui.html` when `API_DOCS_ENABLED=true`.
 *
 * The interesting part is [envelopeCustomizer]. Every controller return value
 * is wrapped by [com.quickpitik.common.ResponseEnvelopeAdvice] before it
 * reaches the wire, but springdoc reads the *declared* return type — so out of
 * the box the spec would promise `AuthResponse` on `POST /auth/login` while the
 * response actually carries `{"success":true,"data":{…}}`. A generated document
 * that misdescribes every controller is worse than none, so the schemas get
 * rewritten to match reality.
 *
 * That advice is scoped `basePackages = ["com.quickpitik.controller"]`, so
 * springdoc's own `/v3/api-docs` response is untouched by it and the spec
 * itself is served unwrapped, as clients expect.
 */
@Configuration
class OpenApiConfig {

    @Bean
    fun quickPitikOpenApi(): OpenAPI = OpenAPI()
        .info(
            Info()
                .title("QuickPitik API")
                .version("v1")
                .description(
                    """
                    Marathon photography platform for Cebu. Public API for the website and the
                    Android app; `ai-api` sits behind this service and is never called by clients
                    directly.

                    **Response envelope.** Every response — success or failure — is
                    `{ "success": boolean, "data": T | null, "errors": [{ "code", "message",
                    "field"? }] }`. The schemas below already show it. Read `errors[].code`, not
                    the message: codes are the stable contract (`common/ErrorCodes.kt`), messages
                    are copy and change freely.

                    **Auth.** Bearer JWT, 15-minute access tokens, rotated refresh tokens. Click
                    *Authorize* and paste the `accessToken` from `POST /auth/login`. Roles are
                    UPPERCASE in JSON: `ADMIN` | `PHOTOGRAPHER` | `RUNNER`.

                    **A caveat on the padlocks.** The bearer requirement is declared globally
                    rather than annotated onto individual methods, so genuinely public endpoints
                    (`/auth/*`, `GET /events/**`, face search, the guest order routes) show a lock
                    they do not enforce. They work signed-out; the authoritative list is
                    `config/SecurityConfig.kt`.
                    """.trimIndent(),
                ),
        )
        .components(
            Components().addSecuritySchemes(
                BEARER_SCHEME,
                SecurityScheme()
                    .type(SecurityScheme.Type.HTTP)
                    .scheme("bearer")
                    .bearerFormat("JWT"),
            ),
        )
        .addSecurityItem(SecurityRequirement().addList(BEARER_SCHEME))

    /**
     * Rewrites every documented response body into the `ApiResponse` envelope
     * the service actually emits.
     *
     * Applied to all responses, not just 2xx: `GlobalExceptionHandler` returns
     * the same envelope for errors, with `data` null and `errors` populated.
     */
    @Bean
    fun envelopeCustomizer(): OpenApiCustomizer = OpenApiCustomizer { openApi ->
        openApi.paths?.values?.forEach { pathItem ->
            pathItem.readOperations().forEach { operation ->
                operation.responses?.values?.forEach { response ->
                    response.content?.values?.forEach { mediaType ->
                        mediaType.schema = envelope(mediaType.schema)
                    }
                }
            }
        }
    }

    /**
     * `T` -> `{ success, data: T, errors }`.
     *
     * Idempotent by inspection: springdoc may hand the same schema instance to
     * more than one operation, and double-wrapping would describe
     * `data.data.data`. Returns the input untouched if it already looks wrapped.
     */
    internal fun envelope(inner: Schema<*>?): Schema<*> {
        if (inner != null && inner.properties?.containsKey("success") == true) return inner

        val data = inner ?: ObjectSchema().nullable(true)
        return ObjectSchema()
            .addProperty("success", BooleanSchema().description("False when `errors` is populated."))
            .addProperty("data", data)
            .addProperty(
                "errors",
                ArraySchema()
                    .nullable(true)
                    .description("Absent on success.")
                    .items(
                        ObjectSchema()
                            .addProperty("code", StringSchema().description("Stable code from ErrorCodes.kt."))
                            .addProperty("message", StringSchema())
                            .addProperty("field", StringSchema().nullable(true)),
                    ),
            )
    }

    internal companion object {
        const val BEARER_SCHEME = "bearer-jwt"
    }
}
