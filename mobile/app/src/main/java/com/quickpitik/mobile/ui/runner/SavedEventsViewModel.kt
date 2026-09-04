package com.quickpitik.mobile.ui.runner

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.SavedEventSummaryDto
import com.quickpitik.mobile.data.repository.SavedEventsRepository
import com.quickpitik.mobile.data.repository.SavedEventsRepositoryImpl
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

// Shared (hoisted to NavHost scope) so the events-discovery heart toggle and the
// profile race log read/write the one saved-events list. Mirrors the website's
// useSavedEventsStore: optimistic toggles, toast-style messages on save/unsave.
class SavedEventsViewModel(application: Application) : AndroidViewModel(application) {
    private val sessionManager = SessionManager.getInstance(application)
    private val repository: SavedEventsRepository = SavedEventsRepositoryImpl()

    val savedEvents: StateFlow<List<SavedEventSummaryDto>> = repository.savedEvents

    val savedIds: StateFlow<Set<String>> = savedEvents
        .map { list -> list.map { it.id }.toSet() }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptySet())

    // One-shot, snackbar-style message ("Saved X." / "Removed X from saved.").
    private val _message = MutableStateFlow<String?>(null)
    val message: StateFlow<String?> = _message

    // Set alongside an unsave message so the snackbar can offer Undo (web
    // parity: the unsave toast carries an Undo action). (id, name).
    private val _undoCandidate = MutableStateFlow<Pair<String, String>?>(null)
    val undoCandidate: StateFlow<Pair<String, String>?> = _undoCandidate

    fun undoUnsave(eventId: String, eventName: String) {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            repository.saveById(token, eventId)
                .onSuccess { _message.value = "Restored $eventName." }
                .onFailure { _message.value = it.message ?: "Couldn't restore that event." }
        }
    }

    fun refresh() {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch { repository.refresh(token) }
    }

    fun isSaved(eventId: String): Boolean = repository.savedEvents.value.any { it.id == eventId }

    fun toggle(event: EventDto) {
        val token = sessionManager.getAccessToken() ?: run {
            _message.value = "Please log in to save events."
            return
        }
        val wasSaved = isSaved(event.id)
        viewModelScope.launch {
            if (wasSaved) {
                repository.unsave(token, event.id)
                    .onSuccess {
                        _undoCandidate.value = event.id to event.name
                        _message.value = "Removed ${event.name} from saved."
                    }
                    .onFailure { _message.value = it.message ?: "Couldn't update saved events." }
            } else {
                repository.save(token, event)
                    .onSuccess { _message.value = "Saved ${event.name}." }
                    .onFailure { _message.value = it.message ?: "Couldn't save that event." }
            }
        }
    }

    // Used by the race log's "Unsave" affordance, where we only have id + name.
    fun unsave(eventId: String, eventName: String) {
        val token = sessionManager.getAccessToken() ?: return
        viewModelScope.launch {
            repository.unsave(token, eventId)
                .onSuccess {
                    _undoCandidate.value = eventId to eventName
                    _message.value = "Removed $eventName from saved."
                }
                .onFailure { _message.value = it.message ?: "Couldn't update saved events." }
        }
    }

    fun clearMessage() {
        _message.value = null
        _undoCandidate.value = null
    }
}
