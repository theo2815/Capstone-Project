package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.remote.RunnerMessageDto
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.ErrorView
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.Typography
import com.quickpitik.mobile.ui.theme.WarningOrange

// Runner inbox — mobile port of the website's runner-inbox-modal. A
// ModalBottomSheet rather than the photographer's AlertDialog: the Mobile
// Design skill prefers sheets for contextual lists, and a scrolling message
// list is exactly the case a dialog handles badly on a phone.
//
// Row tap marks read; if the message carries an orderId it also deep-links to
// that order's receipt, which is how a runner gets from "refund resolved" to
// the actual refund timeline.

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RunnerInboxSheet(
    messages: List<RunnerMessageDto>,
    onDismiss: () -> Unit,
    onMarkRead: (String) -> Unit,
    onMarkAllRead: () -> Unit,
    onRemove: (String) -> Unit,
    onOpenOrder: (String) -> Unit,
    // Non-null when the fetch failed with nothing cached — the sheet must
    // show the failure, not "No messages yet." (an error is not an empty
    // inbox). Retry re-runs the fetch.
    fetchError: String? = null,
    onRetry: () -> Unit = {},
) {
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    var removeTarget by remember { mutableStateOf<RunnerMessageDto?>(null) }
    val unreadCount = messages.count { it.readAt == null }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Bone,
        dragHandle = null,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp)
                .padding(top = 20.dp, bottom = 28.dp),
        ) {
            Text(
                text = "Inbox",
                color = Ink,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
            )
            Spacer(modifier = Modifier.height(12.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Kicker(
                    text = if (unreadCount > 0) "$unreadCount unread · ${messages.size} total"
                    else "${messages.size} total",
                    color = SlateSoft,
                )
                if (unreadCount > 0) {
                    TextButton(onClick = onMarkAllRead) {
                        Text(
                            text = "MARK ALL READ",
                            color = Slate,
                            style = Typography.labelMedium,
                        )
                    }
                }
            }
            Spacer(modifier = Modifier.height(8.dp))

            if (messages.isEmpty() && fetchError != null) {
                ErrorView(
                    message = fetchError,
                    title = "Couldn't load your inbox",
                    onRetry = onRetry,
                    modifier = Modifier.fillMaxWidth(),
                )
            } else if (messages.isEmpty()) {
                RunnerInboxEmptyState()
            } else {
                LazyColumn(
                    // Bounded so a long inbox scrolls inside the sheet rather
                    // than pushing the sheet past the top of the screen.
                    modifier = Modifier.heightIn(max = 480.dp),
                ) {
                    items(messages, key = { it.id }) { message ->
                        RunnerMessageRow(
                            message = message,
                            onClick = {
                                if (message.readAt == null) onMarkRead(message.id)
                                message.orderId?.let(onOpenOrder)
                            },
                            onRemove = { removeTarget = message },
                        )
                        Divider(color = Line)
                    }
                }
            }
        }
    }

    removeTarget?.let { target ->
        AlertDialog(
            onDismissRequest = { removeTarget = null },
            containerColor = Bone,
            title = {
                Text(text = "Remove this message?", color = Ink, fontWeight = FontWeight.Bold)
            },
            text = {
                Text(
                    text = "It will disappear from your inbox. This can't be undone.",
                    color = Slate,
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    onRemove(target.id)
                    removeTarget = null
                }) {
                    Text(text = "REMOVE", color = ErrorRed, style = Typography.labelMedium)
                }
            },
            dismissButton = {
                TextButton(onClick = { removeTarget = null }) {
                    Text(text = "KEEP", color = Ink, style = Typography.labelMedium)
                }
            },
        )
    }
}

// Bell affordance for the runner top bar. Deliberately identical in behaviour to
// PhotographerTopBar's bell (Fresh when unread, ErrorRed "9+" badge) so the two
// roles don't drift. Hidden entirely at zero messages, matching the website's
// runner-notification-bell.
@Composable
fun RunnerInboxBell(
    messageCount: Int,
    unreadCount: Int,
    onClick: () -> Unit,
) {
    if (messageCount == 0) return
    IconButton(
        onClick = onClick,
        // Explicit 48dp: the glyph is 24dp and the skill's minimum tap target
        // is 48dp.
        modifier = Modifier.size(48.dp),
    ) {
        Box {
            Icon(
                imageVector = Icons.Default.Notifications,
                contentDescription = "Inbox",
                tint = if (unreadCount > 0) Fresh else SlateSoft,
            )
            if (unreadCount > 0) {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .size(14.dp)
                        .clip(CircleShape)
                        .background(ErrorRed)
                        .border(2.dp, Bone, CircleShape),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = if (unreadCount > 9) "9+" else unreadCount.toString(),
                        color = Color.White,
                        fontSize = 8.sp,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }
    }
}

@Composable
private fun RunnerInboxEmptyState() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "No messages yet.",
            color = Ink,
            fontSize = 16.sp,
            fontWeight = FontWeight.SemiBold,
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "Updates on your refund requests will land here.",
            color = Slate,
            fontSize = 14.sp,
        )
    }
}

@Composable
private fun RunnerMessageRow(
    message: RunnerMessageDto,
    onClick: () -> Unit,
    onRemove: () -> Unit,
) {
    val isUnread = message.readAt == null
    val toneColor = runnerMessageKindTone(message.kind)
    val title = message.title?.trim().takeUnless { it.isNullOrEmpty() }
        ?: runnerMessageKindLabel(message.kind)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(if (isUnread) BoneDeep.copy(alpha = 0.4f) else Color.Transparent)
            .clickable { onClick() }
            .padding(horizontal = 8.dp, vertical = 14.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (isUnread) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(CircleShape)
                            .background(Fresh),
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                }
                Text(
                    text = runnerMessageKindLabel(message.kind).uppercase(),
                    style = Typography.labelMedium,
                    color = toneColor,
                )
            }
            Text(
                text = formatRunnerInboxDate(message.createdAt),
                style = Typography.labelMedium,
                color = SlateSoft,
            )
        }
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = title,
            color = if (isUnread) Ink else InkSoft,
            fontSize = 16.sp,
            fontWeight = if (isUnread) FontWeight.SemiBold else FontWeight.Normal,
            lineHeight = 22.sp,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = message.body,
            color = Slate,
            fontSize = 14.sp,
            lineHeight = 20.sp,
        )
        if (message.orderId != null) {
            Spacer(modifier = Modifier.height(6.dp))
            // Web copy + tone: "View receipt →" in fresh (was ink "VIEW ORDER").
            Text(
                text = "VIEW RECEIPT →",
                color = Fresh,
                style = Typography.labelMedium,
            )
        }
        Spacer(modifier = Modifier.height(10.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
        ) {
            Text(
                text = "REMOVE",
                color = SlateSoft,
                style = Typography.labelMedium,
                modifier = Modifier
                    .clickable { onRemove() }
                    .padding(horizontal = 6.dp, vertical = 4.dp),
            )
        }
    }
}

// All six backend RunnerMessageKind wire values (entity/RunnerMessage.kt).
private fun runnerMessageKindLabel(kind: String): String = when (kind) {
    "dispute_resolved" -> "Refund approved"
    "dispute_denied" -> "Refund declined"
    "dispute_escalated" -> "Refund escalated"
    "admin_message" -> "Message from admin"
    "account_suspended" -> "Account suspended"
    "account_unsuspended" -> "Account reinstated"
    else -> "Update"
}

private fun runnerMessageKindTone(kind: String): Color = when (kind) {
    "dispute_resolved", "account_unsuspended" -> Fresh
    // Web KIND_TONE: a DENIED refund is error-red; only escalation is amber.
    "dispute_denied" -> ErrorRed
    "dispute_escalated" -> WarningOrange
    "account_suspended" -> ErrorRed
    "admin_message" -> Ink
    else -> Slate
}

// "AUG 14, 2026" — the year matters: refund messages stay in the inbox for
// months, and the website's rows carry it (formatLongDate parity).
private fun formatRunnerInboxDate(iso: String): String {
    return try {
        val parts = iso.substring(0, 10).split("-")
        val months = listOf(
            "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
        )
        "${months[parts[1].toInt() - 1]} ${parts[2].toInt()}, ${parts[0]}"
    } catch (e: Exception) {
        iso
    }
}

@Preview(showBackground = true)
@Composable
private fun RunnerInboxEmptyPreview() {
    QuickPitikMobileTheme {
        RunnerInboxEmptyState()
    }
}

@Preview(showBackground = true)
@Composable
private fun RunnerMessageRowUnreadPreview() {
    QuickPitikMobileTheme {
        RunnerMessageRow(
            message = RunnerMessageDto(
                id = "1",
                kind = "dispute_resolved",
                title = null,
                body = "Your refund for 3 photos from Cebu Marathon 2026 was approved. ₱450.00 is on its way back to your original payment method.",
                orderId = "order-1",
                sourceDecisionId = null,
                createdAt = "2026-08-14T09:12:00Z",
                readAt = null,
            ),
            onClick = {},
            onRemove = {},
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun RunnerMessageRowReadPreview() {
    QuickPitikMobileTheme {
        RunnerMessageRow(
            message = RunnerMessageDto(
                id = "2",
                kind = "account_suspended",
                title = "Account under review",
                body = "Your account has been temporarily suspended while we review recent activity.",
                orderId = null,
                sourceDecisionId = null,
                createdAt = "2026-08-12T14:03:00Z",
                readAt = "2026-08-12T15:00:00Z",
            ),
            onClick = {},
            onRemove = {},
        )
    }
}
