package com.quickpitik.service.disbursement

import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.quickpitik.config.PaymongoProperties
import com.sun.net.httpserver.HttpServer
import org.junit.jupiter.api.Test
import java.math.BigDecimal
import java.net.InetSocketAddress
import kotlin.test.*

class PayMongoDisbursementProviderTest {
    private val req = DisbursementRequest("PAY-TEST-001", BigDecimal("1234.50"), "gcash", "09171234567", "Test Recipient")
    private val properties = PaymongoProperties(
        secretKey = "sk_test_fixture", transferSourceNumber = "source-wallet",
        transferSourceName = "Test Platform", transferSourceBic = "TESTSOURCE",
        transferGcashBic = "TESTGCASH", transferCallbackUrl = "https://example.test/paymongo-transfers",
    )

    @Test fun `HTTP request and response use the Money Movement batch contract`() {
        val mapper = jacksonObjectMapper()
        var received = ""
        var key: String? = null
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        server.createContext("/v2/batch_transfers") { exchange ->
            received = exchange.requestBody.bufferedReader().readText()
            key = exchange.requestHeaders.getFirst("Idempotency-Key")
            val body = """{"transfers":[{"id":"tr_test","status":"pending"}]}""".toByteArray()
            exchange.responseHeaders.add("Content-Type", "application/json")
            exchange.sendResponseHeaders(200, body.size.toLong())
            exchange.responseBody.use { it.write(body) }
        }
        server.start()
        try {
            val adapter = PayMongoDisbursementProvider(properties.copy(disbursementBaseUrl = "http://127.0.0.1:${server.address.port}/v2"))
            assertEquals("tr_test", adapter.initiate(req).providerTransferId)
            val json = mapper.readTree(received)
            assertFalse(json.has("data"))
            assertEquals(1, json["transfers"].size())
            val transfer = json["transfers"][0]
            assertEquals(123450L, transfer["amount"].asLong())
            assertEquals("PHP", transfer["currency"].asText())
            assertEquals("instapay", transfer["provider"].asText())
            assertEquals("source-wallet", transfer["source_account"]["number"].asText())
            assertEquals("09171234567", transfer["destination_account"]["number"].asText())
            assertEquals("TESTGCASH", transfer["destination_account"]["bic"].asText())
            assertEquals(properties.transferCallbackUrl, transfer["callback_url"].asText())
            assertEquals(req.cycleId, key)
        } finally {
            server.stop(0)
        }
    }

    @Test fun `missing institution configuration and invalid amounts fail before sending`() {
        assertFailsWith<IllegalArgumentException> { PayMongoDisbursementProvider.buildTransferRequest(req, PaymongoProperties()) }
        assertFailsWith<IllegalArgumentException> { PayMongoDisbursementProvider.buildTransferRequest(req.copy(method = "unknown"), properties) }
        assertFailsWith<IllegalArgumentException> { PayMongoDisbursementProvider.buildTransferRequest(req.copy(amountPhp = BigDecimal("50000.01")), properties) }
        assertFailsWith<ArithmeticException> { PayMongoDisbursementProvider.toCentavos(BigDecimal("1.001")) }
        assertEquals(50000L, PayMongoDisbursementProvider.toCentavos(BigDecimal("500")))
    }
}
