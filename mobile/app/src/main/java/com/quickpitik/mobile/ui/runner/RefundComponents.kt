package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.OrderDetailDto
import com.quickpitik.mobile.data.remote.RunnerDisputeDto
import com.quickpitik.mobile.ui.theme.*

// ── Refund domain helpers (port of website src/lib/refund-helpers.ts) ────────

// Reason codes + labels — mirror website admin-disputes.ts DISPUTE_REASON_LABEL.
val REFUND_REASONS: List<Pair<String, String>> = listOf(
    "wrong_runner" to "Wrong runner",
    "low_quality" to "Low quality",
    "not_received" to "Not received",
    "duplicate_charge" to "Duplicate charge",
    "other" to "Other reason"
)

fun refundReasonLabel(code: String): String =
    REFUND_REASONS.firstOrNull { it.first == code }?.second ?: code

// Photo IDs the runner can still dispute. Excludes any photo already attached to
// an open / escalated / resolved dispute. Denied + withdrawn leave it eligible.
fun disputablePhotoIds(order: OrderDetailDto): List<String> {
    val blocked = order.disputes
        .filter { it.status != "denied" && it.status != "withdrawn" }
        .map { it.photoId }
        .toSet()
    return order.photoIds.filter { it !in blocked }
}

// Rollup mirrors website getOrderRefundStatus.kind: none|pending|partial|approved|rejected.
fun orderRefundKind(order: OrderDetailDto): String {
    val visible = order.disputes.filter { it.status != "withdrawn" }
    if (visible.isEmpty()) return "none"
    val photoCount = order.photoIds.size
    val pending = visible.count { it.status == "open" || it.status == "escalated" }
    val approved = visible.count { it.status == "resolved" && it.refundAmount != null }
    val rejected = visible.count { it.status == "denied" }
    return when {
        pending > 0 -> if (visible.size < photoCount) "partial" else "pending"
        approved > 0 && rejected == 0 -> "approved"
        rejected > 0 && approved == 0 -> "rejected"
        else -> "partial"
    }
}

// Can submit a new request once the order has no disputes, or every dispute was
// denied — and at least one photo is still eligible.
fun canRequestRefund(order: OrderDetailDto): Boolean {
    val kind = orderRefundKind(order)
    return (kind == "none" || kind == "rejected") && disputablePhotoIds(order).isNotEmpty()
}

// Only OPEN disputes are cancellable. ESCALATED is admin's territory;
// RESOLVED / DENIED / WITHDRAWN are terminal.
fun cancellableDispute(order: OrderDetailDto): RunnerDisputeDto? =
    order.disputes.firstOrNull { it.status == "open" }

// Chip wording shown next to a locked photo (mirror website refundChipLabel).
private fun refundChipLabel(status: String): String = when (status) {
    "open" -> "pending"
    "escalated" -> "in review"
    "resolved" -> "approved"
    "denied" -> "denied"
    "withdrawn" -> "withdrawn"
    else -> "in flight"
}

// Dependency-free ISO formatter — "2026-05-25T14:30:00+08:00" -> "2026-05-25 · 14:30".
private fun formatDisputeTimestamp(iso: String): String = try {
    val datePart = iso.substringBefore("T")
    val timePart = iso.substringAfter("T", "").take(5)
    if (timePart.length == 5) "$datePart · $timePart" else datePart
} catch (e: Exception) {
    iso
}

private data class StatusBadge(val color: Color, val label: String)

private fun statusBadge(status: String): StatusBadge = when (status) {
    "open" -> StatusBadge(Slate, "OPEN")
    "escalated" -> StatusBadge(Slate, "IN REVIEW")
    "resolved" -> StatusBadge(Fresh, "RESOLVED")
    "denied" -> StatusBadge(ErrorRed, "DENIED")
    "withdrawn" -> StatusBadge(SlateSoft, "WITHDRAWN")
    else -> StatusBadge(Slate, status.uppercase())
}

// ── Refund actions (request / cancel) ────────────────────────────────────────

@Composable
fun RefundActionsRow(
    order: OrderDetailDto,
    submitting: Boolean,
    onRequest: () -> Unit,
    onCancel: (disputeId: String) -> Unit
) {
    val canRequest = canRequestRefund(order)
    val kind = orderRefundKind(order)
    val cancellable = cancellableDispute(order)

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (canRequest) {
            OutlinedButton(
                onClick = onRequest,
                enabled = !submitting,
                border = BorderStroke(1.dp, Ink),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
                shape = RoundedCornerShape(percent = 100),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("REQUEST A REFUND", style = Typography.labelMedium)
            }
        } else if (kind != "none") {
            Text(
                text = if (kind == "approved") "Refund issued · cannot resubmit" else "Refund pending review",
                style = Typography.bodySmall,
                color = SlateSoft
            )
        }

        if (cancellable != null) {
            TextButton(
                onClick = { onCancel(cancellable.id) },
                enabled = !submitting,
                colors = ButtonDefaults.textButtonColors(contentColor = Slate),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = if (submitting) "CANCELLING…" else "CANCEL REFUND REQUEST",
                    style = Typography.labelMedium
                )
            }
        }
    }
}

// ── Status banner (success / error feedback after a mutation) ─────────────────

@Composable
fun RefundStatusBanner(message: String, isError: Boolean) {
    Surface(
        color = if (isError) ErrorRed.copy(alpha = 0.12f) else SuccessGreen.copy(alpha = 0.12f),
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Text(
            text = message,
            style = Typography.bodySmall,
            color = if (isError) ErrorRed else SuccessGreen,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp)
        )
    }
}

// ── Refund history timeline (port of website refund-timeline.tsx) ─────────────

@Composable
fun RefundTimeline(disputes: List<RunnerDisputeDto>) {
    if (disputes.isEmpty()) return
    val ordered = disputes.sortedByDescending { it.openedAt }

    Card(
        colors = CardDefaults.cardColors(containerColor = BoneDeep),
        border = BorderStroke(1.dp, Line),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column {
            Text(
                text = "REFUND HISTORY",
                style = Typography.labelSmall,
                color = Slate,
                modifier = Modifier.padding(16.dp)
            )
            ordered.forEachIndexed { index, dispute ->
                if (index > 0) Divider(color = Line)
                RefundTimelineRow(dispute)
            }
        }
    }
}

@Composable
private fun RefundTimelineRow(dispute: RunnerDisputeDto) {
    val badge = statusBadge(dispute.status)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(8.dp)
                        .clip(RoundedCornerShape(percent = 100))
                        .background(badge.color)
                )
                Spacer(Modifier.width(8.dp))
                Text(badge.label, style = Typography.labelSmall, color = badge.color)
                if (dispute.status == "resolved" && dispute.refundAmount != null) {
                    Spacer(Modifier.width(8.dp))
                    Text(
                        text = String.format("· ₱%,.2f refunded", dispute.refundAmount),
                        style = Typography.labelSmall,
                        color = Ink
                    )
                }
            }
            Text(
                text = "PHOTO ${dispute.photoId.take(8).uppercase()}",
                style = Typography.labelSmall,
                color = Slate
            )
        }

        // Lifecycle lines
        TimelineLine("Filed", formatDisputeTimestamp(dispute.openedAt))
        if (dispute.status == "escalated") {
            TimelineLine("Escalated for review", null)
        }
        if (dispute.resolvedAt != null && (dispute.status == "resolved" || dispute.status == "denied")) {
            TimelineLine(
                if (dispute.status == "resolved") "Resolved" else "Denied",
                formatDisputeTimestamp(dispute.resolvedAt)
            )
        }
        if (dispute.withdrawnAt != null && dispute.status == "withdrawn") {
            TimelineLine("You cancelled this request", formatDisputeTimestamp(dispute.withdrawnAt))
        }

        // Reason + notes
        Text(
            text = "Reason · ${refundReasonLabel(dispute.reason)}",
            style = Typography.bodySmall,
            color = InkSoft
        )
        if (dispute.note.isNotBlank()) {
            Text(
                text = "“${dispute.note}”",
                style = Typography.bodySmall,
                color = InkSoft,
                fontWeight = FontWeight.Light
            )
        }
        if (!dispute.resolutionNote.isNullOrBlank() &&
            (dispute.status == "resolved" || dispute.status == "denied")
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp)
            ) {
                Text("ADMIN NOTE", style = Typography.labelSmall, color = Slate)
                Spacer(Modifier.height(2.dp))
                Text(dispute.resolutionNote, style = Typography.bodySmall, color = Ink)
            }
        }
    }
}

@Composable
private fun TimelineLine(label: String, timestamp: String?) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(label, style = Typography.bodySmall, color = SlateSoft)
        if (timestamp != null) {
            Text(timestamp, style = Typography.bodySmall, color = Slate)
        }
    }
}

// ── Request-a-refund dialog (port of website refund-modal.tsx) ────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RefundRequestDialog(
    order: OrderDetailDto,
    submitting: Boolean,
    onDismiss: () -> Unit,
    onSubmit: (photoIds: List<String>, reason: String, note: String) -> Unit
) {
    var selected by remember { mutableStateOf(setOf<String>()) }
    var reason by remember { mutableStateOf<String?>(null) }
    var note by remember { mutableStateOf("") }
    var reasonExpanded by remember { mutableStateOf(false) }

    val canSubmit = selected.isNotEmpty() && reason != null && !submitting

    Dialog(
        onDismissRequest = { if (!submitting) onDismiss() },
        properties = DialogProperties(usePlatformDefaultWidth = false)
    ) {
        Surface(
            color = Bone,
            shape = RoundedCornerShape(20.dp),
            modifier = Modifier
                .fillMaxWidth(0.94f)
                .fillMaxHeight(0.9f)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                // Scrollable body
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .verticalScroll(rememberScrollState())
                        .padding(24.dp),
                    verticalArrangement = Arrangement.spacedBy(20.dp)
                ) {
                    Column {
                        Text("REQUEST A REFUND", style = Typography.labelSmall, color = Slate)
                        Spacer(Modifier.height(4.dp))
                        Text(
                            text = String.format("₱%,.2f", order.total) +
                                " paid for ${order.photoIds.size} photo${if (order.photoIds.size == 1) "" else "s"}" +
                                (order.eventName?.let { " from $it" } ?: "") + ".",
                            style = Typography.bodyMedium,
                            color = InkSoft
                        )
                    }

                    // Photo picker
                    Column {
                        Text("PICK THE PHOTOS TO REFUND", style = Typography.labelSmall, color = Slate)
                        Spacer(Modifier.height(8.dp))
                        Card(
                            colors = CardDefaults.cardColors(containerColor = BoneDeep),
                            border = BorderStroke(1.dp, Line),
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column {
                                order.photos.forEachIndexed { index, photo ->
                                    val lockedDispute = order.disputes.firstOrNull {
                                        it.photoId == photo.id &&
                                            it.status != "denied" &&
                                            it.status != "withdrawn"
                                    }
                                    val isLocked = lockedDispute != null
                                    val isChecked = selected.contains(photo.id)
                                    if (index > 0) Divider(color = Line)
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(horizontal = 12.dp, vertical = 10.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Checkbox(
                                            checked = isChecked,
                                            enabled = !isLocked && !submitting,
                                            onCheckedChange = {
                                                selected = if (isChecked) selected - photo.id else selected + photo.id
                                            },
                                            colors = CheckboxDefaults.colors(checkedColor = Fresh)
                                        )
                                        Box(
                                            modifier = Modifier
                                                .size(40.dp)
                                                .clip(RoundedCornerShape(6.dp))
                                                .background(Line)
                                        ) {
                                            if (photo.thumbnailUrl != null) {
                                                AsyncImage(
                                                    model = photo.thumbnailUrl,
                                                    contentDescription = null,
                                                    modifier = Modifier.fillMaxSize()
                                                )
                                            }
                                        }
                                        Spacer(Modifier.width(12.dp))
                                        Column(modifier = Modifier.weight(1f)) {
                                            Text(
                                                text = "Bib ${photo.bib ?: "—"}",
                                                style = Typography.bodyMedium,
                                                color = if (isLocked) SlateSoft else Ink
                                            )
                                            Text(photo.time, style = Typography.bodySmall, color = SlateSoft)
                                        }
                                        if (isLocked && lockedDispute != null) {
                                            Text(
                                                text = "Refund ${refundChipLabel(lockedDispute.status)}",
                                                style = Typography.labelSmall,
                                                color = Slate
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Reason dropdown
                    Column {
                        Text("REASON", style = Typography.labelSmall, color = Slate)
                        Spacer(Modifier.height(8.dp))
                        ExposedDropdownMenuBox(
                            expanded = reasonExpanded,
                            onExpandedChange = { if (!submitting) reasonExpanded = !reasonExpanded }
                        ) {
                            OutlinedTextField(
                                value = reason?.let { refundReasonLabel(it) } ?: "Choose a reason…",
                                onValueChange = {},
                                readOnly = true,
                                trailingIcon = {
                                    ExposedDropdownMenuDefaults.TrailingIcon(expanded = reasonExpanded)
                                },
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = Line,
                                    focusedTextColor = Ink,
                                    unfocusedTextColor = if (reason == null) SlateSoft else Ink
                                ),
                                shape = RoundedCornerShape(12.dp),
                                modifier = Modifier
                                    .menuAnchor()
                                    .fillMaxWidth()
                            )
                            ExposedDropdownMenu(
                                expanded = reasonExpanded,
                                onDismissRequest = { reasonExpanded = false }
                            ) {
                                REFUND_REASONS.forEach { (code, label) ->
                                    DropdownMenuItem(
                                        text = { Text(label, color = Ink) },
                                        onClick = {
                                            reason = code
                                            reasonExpanded = false
                                        }
                                    )
                                }
                            }
                        }
                    }

                    // Note field
                    Column {
                        Text("NOTE (OPTIONAL)", style = Typography.labelSmall, color = Slate)
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = note,
                            onValueChange = { if (it.length <= 500) note = it },
                            enabled = !submitting,
                            placeholder = { Text("Tell us anything that'll help us review faster.", color = SlateSoft) },
                            minLines = 3,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = Line,
                                focusedTextColor = Ink,
                                unfocusedTextColor = Ink
                            ),
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth()
                        )
                        Text(
                            text = "${note.length} / 500",
                            style = Typography.labelSmall,
                            color = SlateSoft,
                            textAlign = TextAlign.End,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }

                // Sticky footer actions
                Divider(color = Line)
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 24.dp, vertical = 16.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.End),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(
                        onClick = onDismiss,
                        enabled = !submitting,
                        colors = ButtonDefaults.textButtonColors(contentColor = Slate)
                    ) {
                        Text("CANCEL", style = Typography.labelMedium)
                    }
                    Button(
                        onClick = {
                            val r = reason ?: return@Button
                            onSubmit(selected.toList(), r, note.trim())
                        },
                        enabled = canSubmit,
                        colors = ButtonDefaults.buttonColors(containerColor = Ink, contentColor = Bone),
                        shape = RoundedCornerShape(percent = 100)
                    ) {
                        Text(if (submitting) "SENDING…" else "SEND REQUEST", style = Typography.labelMedium)
                    }
                }
            }
        }
    }
}
