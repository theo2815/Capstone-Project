package com.quickpitik.config

import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.http.client.SimpleClientHttpRequestFactory
import org.springframework.web.client.RestClient

@Configuration
class RestClientConfig {

    @Bean(name = ["aiApiRestClient"])
    fun aiApiRestClient(props: AiApiProperties): RestClient {
        val factory = SimpleClientHttpRequestFactory().apply {
            setConnectTimeout(props.connectTimeout)
            setReadTimeout(props.readTimeout)
        }
        return RestClient.builder()
            .baseUrl(props.baseUrl.trimEnd('/'))
            .requestFactory(factory)
            .defaultHeader("X-API-Key", props.apiKey)
            .build()
    }
}
