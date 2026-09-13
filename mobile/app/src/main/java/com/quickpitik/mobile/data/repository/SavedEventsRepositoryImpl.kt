package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.SaveEventRequest
import com.quickpitik.mobile.data.remote.SavedEventSummaryDto
import com.quickpitik.mobile.data.remote.apiCall
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.time.OffsetDateTime

class SavedEventsRepositoryImpl : SavedEventsRepository {
    private val api get() = RetrofitClient.apiService
    private val _saved = MutableStateFlow<List<SavedEventSummaryDto>>(emptyList())
    override val savedEvents: StateFlow<List<SavedEventSummaryDto>> = _saved.asStateFlow()

    override suspend fun refresh(token: String): Result<List<SavedEventSummaryDto>> =
        apiCall("Failed to load saved events") { api.getSavedEvents("Bearer $token") }
            .onSuccess { _saved.value = it }

    // Optimistically insert a summary derived from the event card we already have,
    // then reconcile with the server-returned row (which carries the canonical
    // savedAt + state). Roll back the optimistic insert on failure — same pattern
    // the cart uses, and the same optimistic behaviour as the website store.
    override suspend fun save(token: String, event: EventDto): Result<SavedEventSummaryDto> {
        val original = _saved.value
        if (original.none { it.id == event.id }) {
            val optimistic = SavedEventSummaryDto(
                id = event.id,
                slug = event.slug,
                name = event.name,
                date = event.date,
                state = event.status,
                bannerUrl = event.bannerUrl,
                location = event.location,
                savedAt = OffsetDateTime.now().toString()
            )
            _saved.value = original + optimistic
        }
        return apiCall("Failed to save event") { api.saveEvent("Bearer $token", SaveEventRequest(event.id)) }
            .onSuccess { row -> _saved.value = _saved.value.map { if (it.id == event.id) row else it } }
            // Skipped on cancellation (apiCall rethrows it); safe only while this repository is per-ViewModel and dies with its scope.
            .onFailure { _saved.value = original }
    }

    override suspend fun saveById(token: String, eventId: String): Result<SavedEventSummaryDto> =
        apiCall("Failed to save event") { api.saveEvent("Bearer $token", SaveEventRequest(eventId)) }
            .onSuccess { row -> if (_saved.value.none { it.id == eventId }) _saved.value = _saved.value + row }

    override suspend fun unsave(token: String, eventId: String): Result<Boolean> {
        val original = _saved.value
        _saved.value = original.filter { it.id != eventId }
        return apiCall("Failed to remove saved event") { api.unsaveEvent("Bearer $token", eventId) }
            .map { it.removed }
            // Skipped on cancellation (apiCall rethrows it); safe only while this repository is per-ViewModel and dies with its scope.
            .onFailure { _saved.value = original }
    }
}
