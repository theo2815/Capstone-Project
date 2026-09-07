package com.quickpitik.service.disbursement

import com.quickpitik.config.PaymongoProperties
import com.quickpitik.dto.orders.PaymongoTransferDestination
import com.quickpitik.dto.orders.PaymongoTransferRequest
import com.quickpitik.dto.orders.PaymongoTransferRequestAttributes
import com.quickpitik.dto.orders.PaymongoTransferRequestData
import com.quickpitik.dto.orders.PaymongoTransferResponse
import org.slf4j.LoggerFactory
import org.springframework.http.HttpHeaders
import org.springframework.http.client.SimpleClientHttpRequestFactory
import org.springframework.stereotype.Component
import org.springframework.web.client.RestClient
import org.springframework.web.client.RestClientResponseException
import java.math.BigDecimal
import java.nio.charset.StandardCharsets
import java.util.Base64

/**
 * PayMongo Disbursement (Transfers v2) rail. Registered as a bean but INACTIVE
 * unless app.platform.disbursement-provider=paymongo — DisbursementService only
 * ever resolves the configured provider, so this cannot move money while the
 * default "manual" is in force.
 *
 * ⚠ NOT VERIFIED AGAINST A LIVE ACCOUNT (KYB-gated — docs §I). Before flipping
 * the config to "paymongo", verify in a test-mode account: the exact
 * /batch_transfers attribute shape, the receiving-institution (bank_code) codes
 * from GET /v2/transfers/receiving_institutions, the wallet Available-balance
 * gate, and the transfer/webhook status strings. The request BUILDER is
 * unit-tested; the HTTP call and the doc-ambiguous fields are the owed part.
 *
 * Idempotency: the transfer carries the cycle id as both reference_number and
 * the Idempotency-Key, so a retry never double-sends.
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
        val body = buildTransferRequest(request)
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
                "PayMongo POST /batch_transfers failed for cycle {}: status={} body={}",
                request.cycleId,
                ex.statusCode,
                ex.responseBodyAsString,
            )
            throw ex
        }
        val transferId = response?.data?.id
            ?: throw IllegalStateException(
                "PayMongo transfer created but no id returned for cycle ${request.cycleId}",
            )
        return DisbursementResult(providerTransferId = transferId)
    }

    companion object {
        // ⚠ VERIFY-IN-TEST-MODE: real receiving-institution / BIC codes come from
        // GET /v2/transfers/receiving_institutions. These placeholders let the
        // adapter compile + be unit-tested; they MUST be replaced with the real
        // codes for each e-wallet before PayMongo is enabled.
        internal fun bankCodeFor(method: String): String = when (method.lowercase()) {
            "gcash" -> "GXCHPHM2XXX" // ⚠ placeholder
            "maya" -> "PAPHPHM1XXX" // ⚠ placeholder
            "gotyme" -> "GOTYPHM1XXX" // ⚠ placeholder
            else -> throw IllegalArgumentException(
                "No PayMongo receiving-institution mapping for method '$method'",
            )
        }

        // Pesos → centavos, exact (no rounding on a money value).
        internal fun toCentavos(amountPhp: BigDecimal): Long =
            amountPhp.movePointRight(2).longValueExact()

        internal fun buildTransferRequest(request: DisbursementRequest): PaymongoTransferRequest =
            PaymongoTransferRequest(
                PaymongoTransferRequestData(
                    PaymongoTransferRequestAttributes(
                        amount = toCentavos(request.amountPhp),
                        referenceNumber = request.cycleId,
                        description = "QuickPitik payout ${request.cycleId}",
                        destination = PaymongoTransferDestination(
                            accountNumber = request.accountNumber,
                            accountHolderName = request.accountName,
                            bankCode = bankCodeFor(request.method),
                        ),
                    ),
                ),
            )
    }
}
