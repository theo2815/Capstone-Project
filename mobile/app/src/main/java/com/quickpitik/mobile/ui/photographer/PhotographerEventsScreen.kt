package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowRight
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import com.quickpitik.mobile.ui.runner.EventState
import com.quickpitik.mobile.ui.runner.deriveEventState
import com.quickpitik.mobile.ui.runner.extractCity
import com.quickpitik.mobile.ui.theme.*
import java.time.LocalDate
import java.time.temporal.ChronoUnit

private enum class StatusFilter(val label: String, val tone: StatusTone) {
    LIVE("Live", StatusTone.Approved),
    UPCOMING("Upcoming", StatusTone.Warning),
    COMPLETED("Completed", StatusTone.Neutral),
}

private fun PhotographerEventSummaryDto.filterBucket(today: LocalDate = LocalDate.now()): StatusFilter =
    when (deriveEventState(date, today)) {
        EventState.LIVE -> StatusFilter.LIVE
        EventState.UPCOMING -> StatusFilter.UPCOMING
        EventState.OPEN, EventState.PAST -> StatusFilter.COMPLETED
    }

@Composable
fun PhotographerEventsScreen(
    viewModel: PhotographerDashboardViewModel,
    modifier: Modifier = Modifier,
    onOpenShare: (PhotographerEventSummaryDto) -> Unit = {},
    onSyncEvent: (PhotographerEventSummaryDto) -> Unit = {},
) {
    val eventsState by viewModel.eventsState.collectAsState()
    val activeEvent by viewModel.activeEvent.collectAsState()
    val today = remember { LocalDate.now() }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(Bone)
            .padding(horizontal = 16.dp)
            .padding(top = 12.dp),
    ) {
        Kicker(text = "Covered events", color = Slate)
        Spacer(modifier = Modifier.height(16.dp))

        when (val state = eventsState) {
            is EventsState.Loading -> {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = Slate, strokeWidth = 2.dp)
                }
            }
            is EventsState.Error -> {
                ErrorView(
                    message = state.message,
                    onRetry = { viewModel.fetchEvents() },
                )
            }
            is EventsState.Success -> {
                // Drop the active-event injection from the prior pass — the picker
                // owns "what's active." This screen is about the photographer's
                // covered set, plus their upcoming roster.
                val all = state.events
                val buckets = remember(all) { all.groupBy { it.filterBucket(today) } }
                val liveCount = buckets[StatusFilter.LIVE]?.size ?: 0
                val upcomingCount = buckets[StatusFilter.UPCOMING]?.size ?: 0
                val completedCount = buckets[StatusFilter.COMPLETED]?.size ?: 0

                // Smart default: pick the bucket where the photographer's attention
                // most likely belongs — LIVE > UPCOMING > COMPLETED.
                var selectedFilter by remember(liveCount, upcomingCount, completedCount) {
                    mutableStateOf(
                        when {
                            liveCount > 0 -> StatusFilter.LIVE
                            upcomingCount > 0 -> StatusFilter.UPCOMING
                            else -> StatusFilter.COMPLETED
                        }
                    )
                }

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    StatusFilterChip(
                        label = StatusFilter.LIVE.label,
                        count = liveCount,
                        tone = StatusFilter.LIVE.tone,
                        selected = selectedFilter == StatusFilter.LIVE,
                        onClick = { selectedFilter = StatusFilter.LIVE },
                    )
                    StatusFilterChip(
                        label = StatusFilter.UPCOMING.label,
                        count = upcomingCount,
                        tone = StatusFilter.UPCOMING.tone,
                        selected = selectedFilter == StatusFilter.UPCOMING,
                        onClick = { selectedFilter = StatusFilter.UPCOMING },
                    )
                    StatusFilterChip(
                        label = StatusFilter.COMPLETED.label,
                        count = completedCount,
                        tone = StatusFilter.COMPLETED.tone,
                        selected = selectedFilter == StatusFilter.COMPLETED,
                        onClick = { selectedFilter = StatusFilter.COMPLETED },
                    )
                }

                Spacer(modifier = Modifier.height(20.dp))

                EventsSectionHeader(
                    label = when (selectedFilter) {
                        StatusFilter.LIVE -> "Live today"
                        StatusFilter.UPCOMING -> "Upcoming"
                        StatusFilter.COMPLETED -> "Completed"
                    }
                )

                Spacer(modifier = Modifier.height(12.dp))

                val visible = buckets[selectedFilter].orEmpty()
                if (visible.isEmpty()) {
                    EmptyEventsCard(
                        message = when (selectedFilter) {
                            StatusFilter.LIVE -> "Nothing live right now. Race-day coverage will land here."
                            StatusFilter.UPCOMING -> "No upcoming events on your roster yet."
                            StatusFilter.COMPLETED -> "Your completed coverage will land here."
                        },
                    )
                } else {
                    LazyColumn(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        verticalArrangement = Arrangement.spacedBy(12.dp),
                        contentPadding = PaddingValues(bottom = 24.dp),
                    ) {
                        items(visible, key = { it.id }) { event ->
                            // Per-event variant — picked from the same client-side
                            // lifecycle as the bucket. The filter union (OPEN+PAST →
                            // COMPLETED) does NOT change the card variant; each row
                            // still renders against its own state.
                            when (deriveEventState(event.date, today)) {
                                EventState.LIVE -> LiveEventCard(
                                    event = event,
                                    isActive = activeEvent?.id == event.id,
                                    onSync = { onSyncEvent(event) },
                                    onOpenShare = { onOpenShare(event) },
                                )
                                EventState.UPCOMING -> UpcomingEventCard(
                                    event = event,
                                    today = today,
                                )
                                EventState.OPEN, EventState.PAST -> CompletedEventCard(
                                    event = event,
                                    onOpenShare = { onOpenShare(event) },
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun StatusFilterChip(
    label: String,
    count: Int,
    tone: StatusTone,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val toneColor = when (tone) {
        StatusTone.Approved -> SuccessGreen
        StatusTone.Warning -> WarningOrange
        StatusTone.Danger -> ErrorRed
        StatusTone.Neutral -> Slate
    }
    val bg = if (selected) Ink else BoneDeep
    val textColor = if (selected) Bone else toneColor
    val dotColor = toneColor
    Row(
        modifier = modifier
            .clip(BadgeShape)
            .background(bg)
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        Box(
            modifier = Modifier
                .size(6.dp)
                .clip(CircleShape)
                .background(dotColor),
        )
        Text(
            text = "${label.uppercase()} · $count",
            style = Typography.labelMedium,
            color = textColor,
        )
    }
}

@Composable
private fun EventsSectionHeader(label: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Kicker(text = label, color = Slate)
        Divider(
            modifier = Modifier.weight(1f),
            thickness = 1.dp,
            color = Line,
        )
    }
}

@Composable
private fun LiveEventCard(
    event: PhotographerEventSummaryDto,
    isActive: Boolean,
    onSync: () -> Unit,
    onOpenShare: () -> Unit,
) {
    QpCard(modifier = Modifier.fillMaxWidth(), padding = 0.dp) {
        // 16:9 banner with StatusChip overlay.
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(16f / 9f)
                .background(Line),
        ) {
            val resolvedUrl = resolveImageUrl(event.bannerUrl)
            if (resolvedUrl != null) {
                AsyncImage(
                    model = resolvedUrl,
                    contentDescription = "Event banner",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                Text(
                    text = "BANNER · SOON",
                    style = Typography.labelSmall,
                    color = SlateSoft,
                    modifier = Modifier.align(Alignment.Center),
                )
            }
            StatusChip(
                text = if (isActive) "Live · syncing here" else "Live",
                tone = StatusTone.Approved,
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(10.dp),
            )
        }

        Column(modifier = Modifier.padding(16.dp)) {
            val city = extractCity(event.location).uppercase().ifBlank { "CEBU" }
            Kicker(text = "${event.date} · $city", color = Slate)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = event.name,
                color = Ink,
                fontSize = 16.sp,
                fontWeight = FontWeight.SemiBold,
                lineHeight = 22.sp,
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = event.location,
                color = SlateSoft,
                style = Typography.bodySmall,
            )

            Spacer(modifier = Modifier.height(14.dp))
            Divider(thickness = 1.dp, color = Line)
            Spacer(modifier = Modifier.height(14.dp))

            // Compact single-line stats (no per-stat StatNumber blocks — the Fresh
            // belongs to the Sync CTA, not the revenue figure on a live row).
            Text(
                text = "${event.photoCount} photos · ${event.salesCount} sold",
                color = Slate,
                style = Typography.bodyMedium,
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = "₱%,.2f earned".format(event.revenueKept),
                style = NumeralStyle.copy(fontSize = 16.sp),
                color = Ink,
            )

            Spacer(modifier = Modifier.height(16.dp))
            PrimaryCta(
                text = if (isActive) "Continue syncing" else "Sync from camera",
                onClick = onSync,
                modifier = Modifier.fillMaxWidth(),
            )
            if (event.photoCount > 0) {
                Spacer(modifier = Modifier.height(6.dp))
                TextButton(
                    onClick = onOpenShare,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = "View & share gallery",
                        style = Typography.bodyMedium,
                        color = Ink,
                        fontWeight = FontWeight.Medium,
                    )
                }
            }
        }
    }
}

@Composable
private fun UpcomingEventCard(
    event: PhotographerEventSummaryDto,
    today: LocalDate,
) {
    QpCard(modifier = Modifier.fillMaxWidth(), padding = 12.dp) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            EventCoverThumbnail(url = event.bannerUrl)
            Column(modifier = Modifier.weight(1f)) {
                val city = extractCity(event.location).uppercase().ifBlank { "CEBU" }
                Kicker(text = "${event.date} · $city", color = Slate)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = event.name,
                    color = Ink,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    lineHeight = 20.sp,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = formatOpensIn(event.date, today),
                    color = SlateSoft,
                    style = Typography.bodySmall,
                )
            }
        }
    }
}

@Composable
private fun CompletedEventCard(
    event: PhotographerEventSummaryDto,
    onOpenShare: () -> Unit,
) {
    QpCard(modifier = Modifier.fillMaxWidth(), padding = 0.dp) {
        Row(
            modifier = Modifier
                .clickable(onClick = onOpenShare)
                .padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            EventCoverThumbnail(url = event.bannerUrl)
            Column(modifier = Modifier.weight(1f)) {
                val city = extractCity(event.location).uppercase().ifBlank { "CEBU" }
                Kicker(text = "${event.date} · $city", color = Slate)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = event.name,
                    color = Ink,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    lineHeight = 20.sp,
                    maxLines = 2,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "₱%,.0f · %d sold".format(event.revenueKept, event.salesCount),
                    color = Slate,
                    style = Typography.bodySmall,
                )
            }
            Icon(
                imageVector = Icons.Default.KeyboardArrowRight,
                contentDescription = null,
                tint = SlateSoft,
                modifier = Modifier.size(20.dp),
            )
        }
    }
}

@Composable
private fun EventCoverThumbnail(url: String?, size: Int = 72) {
    val resolved = resolveImageUrl(url)
    Box(
        modifier = Modifier
            .size(size.dp)
            .clip(QpCardShape)
            .background(Line),
        contentAlignment = Alignment.Center,
    ) {
        if (resolved != null) {
            AsyncImage(
                model = resolved,
                contentDescription = null,
                contentScale = ContentScale.Crop,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            Text(
                text = "·",
                color = SlateSoft,
                fontSize = 18.sp,
            )
        }
    }
}

@Composable
private fun EmptyEventsCard(message: String) {
    Surface(
        shape = QpCardShape,
        color = Color.Transparent,
        border = BorderStroke(1.dp, Line),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = message,
                color = SlateSoft,
                style = Typography.bodyMedium,
                textAlign = TextAlign.Center,
            )
        }
    }
}

private fun formatOpensIn(date: String, today: LocalDate): String {
    val eventDate = runCatching { LocalDate.parse(date) }.getOrNull() ?: return "Opens soon"
    val days = ChronoUnit.DAYS.between(today, eventDate)
    return when {
        days <= 0L -> "Opens today"
        days == 1L -> "Opens tomorrow"
        else -> "Opens in $days days"
    }
}

private fun resolveImageUrl(url: String?): String? {
    if (url == null || url.trim().isEmpty()) return null
    if (url.startsWith("/")) {
        return "http://10.0.2.2:8080$url"
    }
    return url.replace("localhost", "10.0.2.2").replace("127.0.0.1", "10.0.2.2")
}
