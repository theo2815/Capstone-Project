package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.SaveEventRequest
import com.quickpitik.mobile.data.remote.SavedEventSummaryDto
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.time.OffsetDateTime

class SavedEventsRepositoryImpl : SavedEventsRepository {
    private val api = RetrofitClient.apiService
    private val _saved = MutableStateFlow<List<SavedEventSummaryDto>>(emptyList())
    override val savedEvents: StateFlow<List<SavedEventSummaryDto>> = _saved.asStateFlow()

    override suspend fun refresh(token: String): Result<List<SavedEventSummaryDto>> {
        return try {
            val response = api.getSavedEvents("Bearer $token")
            if (response.success && response.data != null) {
                _saved.value = response.data
                Result.success(response.data)
            } else {
                Result.failure(Exception(response.error ?: "Failed to load saved events"))
            }
        } catch (e: Exception) {
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

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
        return try {
            val response = api.saveEvent("Bearer $token", SaveEventRequest(event.id))
            if (response.success && response.data != null) {
                _saved.value = _saved.value.map { if (it.id == event.id) response.data else it }
                Result.success(response.data)
            } else {
                _saved.value = original
                Result.failure(Exception(response.error ?: "Failed to save event"))
            }
        } catch (e: Exception) {
            _saved.value = original
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }

    override suspend fun unsave(token: String, eventId: String): Result<Boolean> {
        val original = _saved.value
        _saved.value = original.filter { it.id != eventId }
        return try {
            val response = api.unsaveEvent("Bearer $token", eventId)
            if (response.success && response.data != null) {
                Result.success(response.data.removed)
            } else {
                _saved.value = original
                Result.failure(Exception(response.error ?: "Failed to remove saved event"))
            }
        } catch (e: Exception) {
            _saved.value = original
            Result.failure(Exception(RetrofitClient.parseError(e)))
        }
    }
}
