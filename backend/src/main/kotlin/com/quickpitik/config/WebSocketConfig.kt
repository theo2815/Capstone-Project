package com.quickpitik.config

import com.quickpitik.websocket.AdminNotificationHandshakeInterceptor
import com.quickpitik.websocket.AdminNotificationWebSocketHandler
import com.quickpitik.websocket.EventPhotoHandshakeInterceptor
import com.quickpitik.websocket.EventPhotoWebSocketHandler
import com.quickpitik.websocket.MeNotificationHandshakeInterceptor
import com.quickpitik.websocket.MeNotificationWebSocketHandler
import org.springframework.context.annotation.Configuration
import org.springframework.web.socket.config.annotation.EnableWebSocket
import org.springframework.web.socket.config.annotation.WebSocketConfigurer
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry

@Configuration
@EnableWebSocket
class WebSocketConfig(
    private val eventPhotoHandler: EventPhotoWebSocketHandler,
    private val eventPhotoHandshake: EventPhotoHandshakeInterceptor,
    private val meNotificationHandler: MeNotificationWebSocketHandler,
    private val meNotificationHandshake: MeNotificationHandshakeInterceptor,
    private val adminNotificationHandler: AdminNotificationWebSocketHandler,
    private val adminNotificationHandshake: AdminNotificationHandshakeInterceptor,
) : WebSocketConfigurer {
    override fun registerWebSocketHandlers(registry: WebSocketHandlerRegistry) {
        registry
            .addHandler(eventPhotoHandler, "/ws/events/*/photos")
            .addInterceptors(eventPhotoHandshake)
            .setAllowedOriginPatterns("*")

        registry
            .addHandler(meNotificationHandler, "/ws/me/photographer/notifications")
            .addInterceptors(meNotificationHandshake)
            .setAllowedOriginPatterns("*")

        // Runner-side inbox shares the same handler + handshake — the
        // handler is generic (keyed by userId, role-agnostic) and the
        // separate URL just lets the FE bell pick which channel to listen
        // on. Both endpoints land in the same MeNotificationSessionRegistry,
        // so a user with multiple tabs across roles still gets every push.
        registry
            .addHandler(meNotificationHandler, "/ws/me/runner/notifications")
            .addInterceptors(meNotificationHandshake)
            .setAllowedOriginPatterns("*")

        registry
            .addHandler(adminNotificationHandler, "/ws/admin/notifications")
            .addInterceptors(adminNotificationHandshake)
            .setAllowedOriginPatterns("*")
    }
}
