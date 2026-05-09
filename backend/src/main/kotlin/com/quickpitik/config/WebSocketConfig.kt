package com.quickpitik.config

import com.quickpitik.websocket.EventPhotoHandshakeInterceptor
import com.quickpitik.websocket.EventPhotoWebSocketHandler
import org.springframework.context.annotation.Configuration
import org.springframework.web.socket.config.annotation.EnableWebSocket
import org.springframework.web.socket.config.annotation.WebSocketConfigurer
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry

@Configuration
@EnableWebSocket
class WebSocketConfig(
    private val eventPhotoHandler: EventPhotoWebSocketHandler,
    private val handshakeInterceptor: EventPhotoHandshakeInterceptor,
) : WebSocketConfigurer {
    override fun registerWebSocketHandlers(registry: WebSocketHandlerRegistry) {
        registry
            .addHandler(eventPhotoHandler, "/ws/events/*/photos")
            .addInterceptors(handshakeInterceptor)
            .setAllowedOriginPatterns("*")
    }
}
