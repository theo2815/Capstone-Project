package com.quickpitik.service.disbursement

import com.quickpitik.common.toCentavos
import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.orders.PaymongoTransferAccount
import com.quickpitik.dto.orders.PaymongoTransferRequest
import com.quickpitik.dto.orders.PaymongoTransferRequestAttributes
import com.quickpitik.dto.orders.PaymongoTransferResponse
import org.slf4j.LoggerFactory
import org.springframework.http.HttpHeaders
import org.springframework.http.client.SimpleClientHttpRequestFactory
import org.springframework.stereotype.Component
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClientResponseException
import java.nio.charset.StandardCharsets
import java.util.Base64

/**
 * PayMongo Disbursement (Transfers v2) rail. Registered as a bean but INACTIVE
 * unless app.platform.disbursement-provider=paymongo — DisbursementService only
 * ever resolves the configured provider, so this cannot move money while the
 * default "manual" is in force.
 *
 * The HTTP contract follows PayMongo's Money Movement API. A test-mode account
 * is still needed to verify wallet funding, receiving institutions and callbacks.
 *
 * Idempotency uses the cycle id. PayMongo retains keys for about 24 hours;
 * ambiguous timeouts must be reconciled before retrying beyond that window.
 */
@Component
class PayMongoDisbursementProvider(
    private val properties: PaymongoProperties,
) : DisbursementProvider {
    override val providerName = "paymongo"
    private val log = LoggerFactory.getLogger(javaClass)

    private val authHeader: String by lazy {
        "Basic " + Base64.getEncoder()
            .encodeToString("${properties.secretKey}:".toByteArray(StandardCharsets.UTF_8))
    }

    private val restClient: RestClient by lazy {
        val requestFactory = SimpleClientHttpRequestFactory().apply {
            setConnectTimeout(properties.connectTimeout)
            setReadTimeout(properties.readTimeout)
        }
        RestClient.builder()
            .requestFactory(requestFactory)
            .baseUrl(properties.disbursementBaseUrl)
            .defaultHeader(HttpHeaders.AUTHORIZATION, authHeader)
            .defaultHeader(HttpHeaders.ACCEPT, "application/json")
            .defaultHeader(HttpHeaders.CONTENT_TYPE, "application/json")
            .build()
    }

    override fun initiate(request: DisbursementRequest): DisbursementResult {
        val body = buildTransferRequest(request, properties)
        val response = try {
            restClient.post()
                .uri("/batch_transfers")
                // Stable per cycle → PayMongo dedupes a retry to the same transfer.
                .header("Idempotency-Key", request.cycleId)
                .body(body)
                .retrieve()
                .body(PaymongoTransferResponse::class.java)
        } catch (ex: RestClientResponseException) {
            log.error(
                "PayMongo POST /batch_transfers failed for cycle {}: status={}",
                request.cycleId,
                ex.statusCode,
            )
            throw ex
        }
        val transferId = response?.transfers?.singleOrNull()?.id?.takeIf { it.isNotBlank() }
            ?: throw IllegalStateException(
                "PayMongo transfer created but no id returned for cycle ${request.cycleId}",
            )
        return DisbursementResult(providerTransferId = transferId)
    }

    companion object {
        internal fun buildTransferRequest(
            request: DisbursementRequest,
            properties: PaymongoProperties,
        ): PaymongoTransferRequest {
            val bic = when (request.method.lowercase()) {
                "gcash" -> properties.transferGcashBic
                "maya" -> properties.transferMayaBic
                "gotyme" -> properties.transferGotymeBic
                else -> throw IllegalArgumentException("Unsupported transfer method '${request.method}'")
            }
            require(listOf(bic, properties.transferSourceNumber, properties.transferSourceName,
                properties.transferSourceBic, properties.transferCallbackUrl).all { it.isNotBlank() }) {
                "PayMongo transfer account, institution and callback configuration is required"
            }
            require(properties.transferRail in setOf("instapay", "pesonet")) { "Unsupported transfer rail" }
            require(request.amountPhp.signum() > 0) { "Transfer amount must be positive" }
            return PaymongoTransferRequest(
                listOf(
                    PaymongoTransferRequestAttributes(
                        amount = request.amountPhp.toCentavos(),
                        description = "QuickPitik payout ${request.cycleId}",
                        sourceAccount = PaymongoTransferAccount(
                            properties.transferSourceNumber, properties.transferSourceName, properties.transferSourceBic,
                        ),
                        destinationAccount = PaymongoTransferAccount(request.accountNumber, request.accountName, bic),
                        provider = properties.transferRail,
                        callbackUrl = properties.transferCallbackUrl,
                    ),
                ),
            )
        }
    }
}
