package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.gson.Gson
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.QpWebSocket
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.RunnerMessageDto
import com.quickpitik.mobile.data.remote.RunnerMessageFrame
import com.quickpitik.mobile.data.remote.WsState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

// Runner-side inbox — the mobile equivalent of the website's
// runner-notification-bell + runner-inbox-modal. Backed by
// MeRunnerMessagesController; carries refund outcomes, suspension notices, and
// admin messages, which had no mobile surface at all before this.
//
// Shared (hoisted to NavHost scope) so the bell badge agrees across the events
// list, the gallery cockpit, and anywhere else it's mounted.
//
// Mirrors the photographer inbox in PhotographerDashboardViewModel: direct
// RetrofitClient calls (no repository), refetch after each mutation, plus a
// WebSocket on /ws/me/runner/notifications so an admin decision lands on the
// bell without the runner navigating anywhere.
class RunnerInboxViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val gson = Gson()
    private val socket = QpWebSocket(sessionManager, viewModelScope)

    private val _messages = MutableStateFlow<List<RunnerMessageDto>>(emptyList())
    val messages: StateFlow<List<RunnerMessageDto>> = _messages

    val unreadCount: StateFlow<Int> = _messages
        .map { list -> list.count { it.readAt == null } }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 0)

    init {
        viewModelScope.launch {
            socket.state.collect { state ->
                // Reconnect grace: anything pushed while the socket was down was
                // missed entirely, so every (re)open pulls the authoritative
                // list. Same as the website's ws.onopen → refetch().
                if (state is WsState.Open) fetchMessages()
            }
        }
        viewModelScope.launch {
            socket.frames.collect { raw -> applyPush(raw) }
        }
    }

    // Called from the screens on lifecycle START/STOP rather than from init, so
    // a backgrounded app isn't holding a socket open in the user's pocket.
    fun connect() = socket.connect(CHANNEL)

    fun disconnect() = socket.close()

    private fun applyPush(raw: String) {
        val frame = runCatching { gson.fromJson(raw, RunnerMessageFrame::class.java) }.getOrNull()
        if (frame?.type != "message.created") return
        val message = frame.message ?: return
        // Dedupe by id: the reconnect refetch above and a push can legitimately
        // deliver the same row, and the backend may re-broadcast.
        val current = _messages.value
        if (current.any { it.id == message.id }) return
        _messages.value = listOf(message) + current
    }

    override fun onCleared() {
        socket.release()
        super.onCleared()
    }

    fun fetchMessages() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService.getRunnerMessages("Bearer $token")
                if (response.success && response.data != null) {
                    _messages.value = response.data
                }
            } catch (e: Exception) {
                // Fail silently: the inbox is ambient, and a failed poll must not
                // interrupt whatever the runner was actually doing. Same posture
                // as the photographer inbox.
            }
        }
    }

    fun markRead(messageId: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService
                    .markRunnerMessageRead("Bearer $token", messageId)
                if (response.success) fetchMessages()
            } catch (e: Exception) {
                // Fail silently
            }
        }
    }

    fun markAllRead() {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService.markAllRunnerMessagesRead("Bearer $token")
                if (response.success) fetchMessages()
            } catch (e: Exception) {
                // Fail silently
            }
        }
    }

    fun remove(messageId: String) {
        viewModelScope.launch {
            val token = sessionManager.getAccessToken() ?: return@launch
            try {
                val response = RetrofitClient.apiService
                    .removeRunnerMessage("Bearer $token", messageId)
                if (response.success) fetchMessages()
            } catch (e: Exception) {
                // Fail silently
            }
        }
    }

    private companion object {
        // Separate URL from the photographer channel even though the BE handler
        // and registry are shared — see WebSocketConfig. The split exists so the
        // client knows which inbox a frame belongs to.
        const val CHANNEL = "/ws/me/runner/notifications"
    }
}
