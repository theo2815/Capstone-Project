package com.quickpitik.config

import org.apache.hc.client5.http.config.ConnectionConfig
import org.apache.hc.client5.http.config.RequestConfig
import org.apache.hc.client5.http.impl.classic.HttpClients
import org.apache.hc.client5.http.impl.io.PoolingHttpClientConnectionManager
import org.apache.hc.core5.util.Timeout
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory
import org.springframework.web.client.RestClient

@Configuration
class RestClientConfig {

    @Bean(name = ["aiApiRestClient"])
    fun aiApiRestClient(props: AiApiProperties): RestClient {
        // Pooled HTTP client. Async indexing (PhotoIndexingService) issues many
        // concurrent ai-api calls — face enroll + bib OCR per photo, fanned
        // across the imageProcessing pool. SimpleClientHttpRequestFactory opens
        // a fresh connection per request, serializing throughput under load.
        // Timeouts are set on the Apache client (connect + socket + the wait for
        // a pooled connection) so they survive the photo-indexing fan-out.
        val connectTimeout = Timeout.ofMilliseconds(props.connectTimeout.toMillis())
        val readTimeout = Timeout.ofMilliseconds(props.readTimeout.toMillis())

        val connectionManager = PoolingHttpClientConnectionManager().apply {
            maxTotal = props.maxConnections
            defaultMaxPerRoute = props.maxConnectionsPerRoute
            setDefaultConnectionConfig(
                ConnectionConfig.custom()
                    .setConnectTimeout(connectTimeout)
                    .setSocketTimeout(readTimeout)
                    .build(),
            )
        }
        val requestConfig = RequestConfig.custom()
            .setConnectionRequestTimeout(connectTimeout)
            .build()
        val httpClient = HttpClients.custom()
            .setConnectionManager(connectionManager)
            .setDefaultRequestConfig(requestConfig)
            .build()

        return RestClient.builder()
            .baseUrl(props.baseUrl.trimEnd('/'))
            .requestFactory(HttpComponentsClientHttpRequestFactory(httpClient))
            .defaultHeader("X-API-Key", props.apiKey)
            .build()
    }
}
