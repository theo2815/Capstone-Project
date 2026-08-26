package com.quickpitik.mobile.data.remote

import com.quickpitik.mobile.data.local.SessionManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit
import kotlin.math.min
import kotlin.math.pow

/**
 * Minimal envelope for peeking at a frame's discriminator before committing to
 * a full parse. Every backend frame carries a `type`; unknown ones are dropped.
 */
data class WsFrameEnvelope(val type: String?)

/**
 * Connection state of a [QpWebSocket]. [Failed] carries the number of
 * consecutive failed attempts so a screen can decide when to stop showing
 * "reconnecting…" and offer a manual refresh instead.
 */
sealed class WsState {
    object Idle : WsState()
    object Connecting : WsState()
    object Open : WsState()
    data class Failed(val attempts: Int) : WsState()
}

/**
 * Thin OkHttp WebSocket wrapper — the mobile equivalent of the website's
 * `use-websocket` family of hooks. Emits raw JSON frames; parsing is the
 * caller's job, because each channel deserializes into a different DTO.
 *
 * Deliberately NOT a singleton and NOT DI-managed: the module has no DI
 * framework and ViewModels reach `RetrofitClient.apiService` directly, so a
 * per-ViewModel instance owned in `viewModelScope` matches the surrounding
 * style. One instance per channel.
 *
 * Reconnect uses the same curve as the website (`use-runner-notifications-ws`):
 * `min(30s, 1000 * 2^min(attempts, 5))`. Callers that want a "give up" UI read
 * [state] for `Failed(attempts)` rather than having the socket stop retrying —
 * a race lasts hours and the connection should keep trying to heal itself.
 */
class QpWebSocket(
    private val sessionManager: SessionManager,
    private val scope: CoroutineScope,
) {
    private val _frames = MutableSharedFlow<String>(extraBufferCapacity = 64)

    /** Raw JSON text frames, in arrival order. */
    val frames: SharedFlow<String> = _frames

    private val _state = MutableStateFlow<WsState>(WsState.Idle)
    val state: StateFlow<WsState> = _state

    // All five fields below are mutated from two threads — the main thread
    // (connect/close/release, and the reconnect coroutine on viewModelScope)
    // and OkHttp's callback thread (onClosed/onFailure → scheduleReconnect).
    // Every mutating section is synchronized(this); the sections are short and
    // non-blocking (newWebSocket only enqueues), so the lock is uncontended in
    // practice. @Volatile alone wouldn't do — the check-then-act sequences
    // (connect's "already open?" guard, the subscriber count) are compound.
    private var socket: WebSocket? = null
    private var reconnectJob: Job? = null
    private var attempts = 0

    // Closed by us (screen backgrounded / event deselected) vs dropped by the
    // network. Only the latter should reconnect — without this flag a manual
    // close() immediately reconnects itself via onClosed/onFailure.
    private var closedByCaller = false

    // How many screens currently want this channel open. The runner inbox
    // socket is shared by the events list and the gallery cockpit, and Compose
    // Navigation gives no ordering guarantee between "new screen composes" and
    // "old screen disposes" — an unguarded close() from the outgoing screen
    // could land after the incoming screen's connect() and leave the bell dead.
    // Counting subscribers makes the pair order-independent.
    private var subscribers = 0

    // Own client rather than reusing RetrofitClient's: the shared one installs
    // TokenAuthenticator, which is REST-only (it retries a Request with a fresh
    // bearer). A WebSocket upgrade that 401s can't be "retried" that way — the
    // handshake carries the token, so the fix is to reconnect, which the backoff
    // below already does with a freshly-read token. No read timeout, because an
    // idle notification channel is silent for minutes at a time and the default
    // 10s would tear it down constantly.
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    /**
     * Registers interest in [path] (absolute, e.g. "/ws/me/runner/notifications")
     * and opens it if it isn't already. Balanced by [close].
     */
    fun connect(path: String) = synchronized(this) {
        subscribers += 1
        if (socket != null || reconnectJob != null) return
        closedByCaller = false
        attempts = 0
        open(path)
    }

    /**
     * Releases one subscriber's interest. The socket actually closes only when
     * the last one lets go — see [subscribers].
     */
    fun close(): Unit = synchronized(this) {
        if (subscribers > 0) subscribers -= 1
        if (subscribers > 0) return
        release()
    }

    /** Forces the socket shut regardless of subscriber count (ViewModel teardown). */
    fun release(): Unit = synchronized(this) {
        subscribers = 0
        closedByCaller = true
        reconnectJob?.cancel()
        reconnectJob = null
        socket?.close(NORMAL_CLOSURE, null)
        socket = null
        attempts = 0
        _state.value = WsState.Idle
    }

    private fun open(path: String) {
        // Read the token on EVERY attempt, never captured once. Access tokens
        // are 15-minute JWTs; a socket that reconnects an hour later with the
        // token it was constructed with would fail the handshake forever, and
        // the failure is invisible (the backend just refuses the upgrade).
        val token = sessionManager.getAccessToken()
        if (token == null) {
            _state.value = WsState.Idle
            return
        }
        _state.value = WsState.Connecting

        val request = Request.Builder()
            .url(RetrofitClient.wsOrigin + path)
            // Same channel the website uses: the BE handshake interceptors read
            // the first Sec-WebSocket-Protocol value as the bearer and echo it
            // back as the selected subprotocol. Both interceptors also accept
            // ?token= as a fallback if this ever proves troublesome on OkHttp.
            .header("Sec-WebSocket-Protocol", token)
            .build()

        socket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                synchronized(this@QpWebSocket) { attempts = 0 }
                _state.value = WsState.Open
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                _frames.tryEmit(text)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                scheduleReconnect(path)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                scheduleReconnect(path)
            }
        })
    }

    private fun scheduleReconnect(path: String): Unit = synchronized(this) {
        if (closedByCaller) return
        socket = null
        attempts += 1
        _state.value = WsState.Failed(attempts)
        reconnectJob?.cancel()
        reconnectJob = scope.launch {
            delay(backoffMs(attempts))
            synchronized(this@QpWebSocket) {
                reconnectJob = null
                if (!closedByCaller) open(path)
            }
        }
    }

    private fun backoffMs(attempt: Int): Long =
        min(MAX_BACKOFF_MS, (1000.0 * 2.0.pow(min(attempt, 5))).toLong())

    companion object {
        private const val NORMAL_CLOSURE = 1000
        private const val MAX_BACKOFF_MS = 30_000L

        /**
         * Attempts after which a screen should stop implying "reconnecting" and
         * surface a manual refresh. Matches the website's
         * `MAX_RECONNECT_ATTEMPTS` in `use-event-live-photos`.
         */
        const val MAX_QUIET_ATTEMPTS = 3
    }
}
