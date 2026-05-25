package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.SavedEventSummaryDto
import kotlinx.coroutines.flow.StateFlow

// Single source of truth for the runner's saved (bookmarked) events — the mobile
// equivalent of the website's useSavedEventsStore. Both the events-discovery
// screen (heart toggle) and the profile race log observe the same `savedEvents`
// flow, so a save on one surface is immediately reflected on the other.
//
// Note: the website also supports a guest -> auth /merge of locally-cached IDs,
// because the web lets you browse /events logged out. Mobile requires login
// BEFORE the events screen, so there is never any guest state to merge — the
// /merge endpoint is intentionally not wired here.
interface SavedEventsRepository {
    val savedEvents: StateFlow<List<SavedEventSummaryDto>>

    suspend fun refresh(token: String): Result<List<SavedEventSummaryDto>>
    suspend fun save(token: String, event: EventDto): Result<SavedEventSummaryDto>
    suspend fun unsave(token: String, eventId: String): Result<Boolean>
}
