package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.RunnerMessageDto
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
// RetrofitClient calls (no repository), refetch after each mutation. WebSocket
// push (/ws/me/runner/notifications) is the faithful website behaviour and is
// deliberately deferred — refetch-on-open is the acceptable first pass.
class RunnerInboxViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)

    private val _messages = MutableStateFlow<List<RunnerMessageDto>>(emptyList())
    val messages: StateFlow<List<RunnerMessageDto>> = _messages

    val unreadCount: StateFlow<Int> = _messages
        .map { list -> list.count { it.readAt == null } }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), 0)

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
}
