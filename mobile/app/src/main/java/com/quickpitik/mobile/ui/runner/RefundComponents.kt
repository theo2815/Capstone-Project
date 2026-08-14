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
import java.util.Locale

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

// ── Refund policy copy (verbatim port of website src/lib/refund-policy.ts) ───
// One source of truth for the words. Surfaced pre-purchase from the event
// cockpit; the website also reuses it inside the request modal.

private const val REFUND_PROCESSING_DAYS = 3
private const val REFUND_ELIGIBILITY_DAYS = 30

val REFUND_POLICY_BULLETS: List<Pair<String, String>> = listOf(
    "Eligibility" to
        "Request a refund within $REFUND_ELIGIBILITY_DAYS days of your purchase. After that the order is final.",
    "Accepted reasons" to
        "Wrong runner in the photo, photo quality too low to use, order paid but never delivered, or you were charged twice for the same order. Anything else, pick Other and tell us what happened.",
    "Review time" to
        "We review every request within $REFUND_PROCESSING_DAYS business days. You'll see the status update on this receipt — no email needed.",
    "Where the money goes" to
        "Approved refunds return to your original payment method. GCash and Maya land within 24 hours; cards take 5–7 business days depending on the bank.",
    "What we don't refund" to
        "Photos you've already downloaded and kept past the eligibility window, change-of-mind after 30 days, or photos that match your bib and selfie correctly.",
)

// Read-only policy disclosure — port of the website's RefundModal mode="policy",
// reached from the "Refund Policy →" kicker on the event cockpit. Numbered
// rules with a left hairline, matching web's border-l list.
@Composable
fun RefundPolicyDialog(onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = Bone,
        title = {
            Text(
                text = "Refund policy",
                color = Ink,
                fontWeight = FontWeight.Bold,
            )
        },
        text = {
            Column(
                modifier = Modifier.verticalScroll(rememberScrollState()),
            ) {
                REFUND_POLICY_BULLETS.forEachIndexed { index, (kicker, body) ->
                    Row(modifier = Modifier.padding(bottom = 20.dp)) {
                        // Left hairline rule, web's `border-l border-line pl-4`.
                        Box(
                            modifier = Modifier
                                .width(1.dp)
                                .fillMaxHeight()
                                .background(Line),
                        )
                        Column(modifier = Modifier.padding(start = 16.dp)) {
                            Kicker(
                                text = "${(index + 1).toString().padStart(2, '0')} · $kicker",
                                color = Slate,
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = body,
                                color = InkSoft,
                                style = Typography.bodyMedium,
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text(text = "GOT IT", color = Ink, style = Typography.labelMedium)
            }
        },
    )
}

// Photo IDs the runner can still dispute. Excludes any photo already attached to
// an open / escalated / resolved dispute. Denied + withdrawn leave it eligible.
fun disputablePhotoIds(order: OrderDetailDto): List<String> {
    val blocked = order.disputes
        .filter { it.status != "denied" && it.status != "withdrawn" }
        .map { it.photoId }
        .toSet()
    return order.photoIds.filter { it !in blocked }
}

// Full port of website getOrderRefundStatus — kind plus the counts/amount/note
// the chip needs. Takes the raw pieces rather than an order type so it serves
// both OrderListItemDto (list rows) and OrderDetailDto (expanded receipt).
data class RefundRollup(
    val kind: String,          // none | pending | partial | approved | rejected
    val refundAmount: Double,
    val pendingCount: Int,
    val approvedCount: Int,
    val rejectedCount: Int,
    val rejectedNote: String?,
    val totalDisputed: Int,
)

fun computeRefundRollup(photoCount: Int, disputes: List<RunnerDisputeDto>): RefundRollup {
    val visible = disputes.filter { it.status != "withdrawn" }
    val pending = visible.filter { it.status == "open" || it.status == "escalated" }
    val approved = visible.filter { it.status == "resolved" && it.refundAmount != null }
    val rejected = visible.filter { it.status == "denied" }
    val kind = when {
        visible.isEmpty() -> "none"
        pending.isNotEmpty() -> if (visible.size < photoCount) "partial" else "pending"
        approved.isNotEmpty() && rejected.isEmpty() -> "approved"
        rejected.isNotEmpty() && approved.isEmpty() -> "rejected"
        else -> "partial"
    }
    return RefundRollup(
        kind = kind,
        refundAmount = approved.sumOf { it.refundAmount ?: 0.0 },
        pendingCount = pending.size,
        approvedCount = approved.size,
        rejectedCount = rejected.size,
        // Prefer admin's resolution note over the runner's own submission note —
        // when the chip says "declined" the runner cares why admin said no.
        rejectedNote = rejected.firstOrNull()?.let { it.resolutionNote ?: it.note },
        totalDisputed = visible.size,
    )
}

// Rollup mirrors website getOrderRefundStatus.kind: none|pending|partial|approved|rejected.
fun orderRefundKind(order: OrderDetailDto): String =
    computeRefundRollup(order.photoIds.size, order.disputes).kind

// Refund-state chip for an order row. Renders nothing when there's no dispute,
// so an ordinary receipt stays clean. Port of the website's order-row status
// line: "Refund pending · N of M" / "in review" / "approved · ₱x" / "declined"
// with the admin's note underneath.
@Composable
fun RefundStatusChip(rollup: RefundRollup, photoCount: Int) {
    if (rollup.kind == "none") return

    val (tone, label) = when (rollup.kind) {
        "pending" -> WarningOrange to "Refund pending · ${rollup.totalDisputed} of $photoCount"
        "partial" -> WarningOrange to "Refund in review · ${rollup.totalDisputed} of $photoCount"
        "approved" -> Fresh to "Refund approved · ₱${formatPeso(rollup.refundAmount)}"
        "rejected" -> ErrorRed to "Refund declined"
        else -> Slate to "Refund updated"
    }

    Column(modifier = Modifier.padding(top = 8.dp)) {
        Box(
            modifier = Modifier
                .clip(BadgeShape)
                .background(tone.copy(alpha = 0.12f))
                .padding(horizontal = 8.dp, vertical = 4.dp),
        ) {
            Text(text = label, color = tone, style = Typography.labelMedium)
        }
        rollup.rejectedNote?.takeIf { it.isNotBlank() }?.let { note ->
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = note, color = SlateSoft, style = Typography.bodySmall)
        }
    }
}

// Whole-peso formatting with thousands separators, matching the website's
// toLocaleString with no decimals on order totals.
private fun formatPeso(amount: Double): String =
    String.format(Locale.US, "%,.0f", amount)

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
    var showCancelConfirm by remember { mutableStateOf(false) }

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
                // Confirm first — withdrawing is destructive (the dispute row is
                // hard-deleted) and the button sits directly under the request
                // CTA, so a mis-tap is easy. Mirrors the website's
                // confirm({danger:true}) in handleCancelRequest.
                onClick = { showCancelConfirm = true },
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

    if (showCancelConfirm && cancellable != null) {
        AlertDialog(
            onDismissRequest = { showCancelConfirm = false },
            containerColor = Bone,
            title = {
                Text(text = "Cancel refund request?", color = Ink, fontWeight = FontWeight.Bold)
            },
            text = {
                Text(
                    text = "You can submit a new request for these photos later if you change your mind.",
                    color = Slate,
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    showCancelConfirm = false
                    onCancel(cancellable.id)
                }) {
                    Text(text = "CANCEL REQUEST", color = ErrorRed, style = Typography.labelMedium)
                }
            },
            dismissButton = {
                TextButton(onClick = { showCancelConfirm = false }) {
                    Text(text = "KEEP REQUEST", color = Ink, style = Typography.labelMedium)
                }
            },
        )
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
        shape = QpCardShape,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column {
            Box(modifier = Modifier.padding(16.dp)) {
                Kicker("Refund history")
            }
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
                                        Text(
                                            text = "Photo ${index + 1}",
                                            style = Typography.bodyMedium,
                                            color = if (isLocked) SlateSoft else Ink,
                                            modifier = Modifier.weight(1f),
                                        )
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
                        Text("Cancel", style = Typography.labelMedium)
                    }
                    PrimaryCta(
                        text = "Send request",
                        onClick = {
                            val r = reason ?: return@PrimaryCta
                            onSubmit(selected.toList(), r, note.trim())
                        },
                        enabled = canSubmit,
                        loading = submitting,
                    )
                }
            }
        }
    }
}
