package com.quickpitik.mobile.ui.common

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.ReportRequest
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.apiCall
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.FieldShape
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.Typography
import kotlinx.coroutines.launch

/** What a report is about (AA-R9 routes). */
sealed interface ReportTarget {
    data class Event(val id: String) : ReportTarget
    data class Photographer(val handle: String) : ReportTarget
}

/** The backend's note rule (ReportRequest: 10-500 characters), on the trimmed note it is sent. */
internal fun canSendReport(note: String): Boolean = note.trim().length in 10..500

/**
 * Website report-dialog parity (AA-R17a): a "Report" link for signed-in users
 * only, a free-text note (no categories, owner question P2-Q5), and the same
 * thanks whether or not this account had already reported the target. A
 * failure (a 429 included) shows the server's message and keeps the note.
 */
@Composable
fun ReportAction(target: ReportTarget, modifier: Modifier = Modifier) {
    // Read live, like rememberIsTrueRunner: sign-in does not recreate the composition.
    val token = SessionManager.getInstance(LocalContext.current).getAccessToken() ?: return
    var open by remember { mutableStateOf(false) }
    TextButton(onClick = { open = true }, modifier = modifier.heightIn(min = 48.dp)) {
        Text("Report", style = Typography.labelMedium, color = Slate)
    }
    if (open) ReportDialog(token, target, onDismiss = { open = false })
}

@Composable
private fun ReportDialog(token: String, target: ReportTarget, onDismiss: () -> Unit) {
    val scope = rememberCoroutineScope()
    var note by remember { mutableStateOf("") }
    var sending by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var thanked by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = { if (!sending) onDismiss() },
        containerColor = Bone,
        title = {
            Text(
                if (target is ReportTarget.Event) "Report this event" else "Report this photographer",
                style = Typography.titleMedium,
                color = Ink,
            )
        },
        text = {
            if (thanked) {
                Text("Thanks, an admin will look at it.", style = Typography.bodyMedium, color = InkSoft)
            } else {
                Column {
                    OutlinedTextField(
                        value = note,
                        onValueChange = { if (it.length <= 500) note = it },
                        enabled = !sending,
                        placeholder = { Text("Tell us what is wrong (at least 10 characters).", color = SlateSoft) },
                        minLines = 3,
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = Fresh,
                            unfocusedBorderColor = Line,
                            focusedTextColor = Ink,
                            unfocusedTextColor = Ink,
                        ),
                        shape = FieldShape,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Text(
                        text = "${note.length} / 500",
                        style = Typography.labelSmall,
                        color = SlateSoft,
                        textAlign = TextAlign.End,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    error?.let {
                        Spacer(Modifier.height(8.dp))
                        Text(it, style = Typography.bodySmall, color = ErrorRed)
                    }
                }
            }
        },
        confirmButton = {
            if (thanked) {
                TextButton(onClick = onDismiss) { Text("Close", style = Typography.labelMedium, color = Ink) }
            } else {
                TextButton(
                    enabled = canSendReport(note) && !sending,
                    onClick = {
                        sending = true
                        error = null
                        scope.launch {
                            sendReport(token, target, note.trim())
                                .onSuccess { thanked = true }
                                .onFailure { error = it.message }
                            sending = false
                        }
                    },
                ) {
                    Text("Send report", style = Typography.labelMedium, color = Ink)
                }
            }
        },
        dismissButton = {
            if (!thanked) {
                TextButton(onClick = onDismiss, enabled = !sending) {
                    Text("Cancel", style = Typography.labelMedium, color = Slate)
                }
            }
        },
    )
}

private suspend fun sendReport(token: String, target: ReportTarget, note: String) =
    apiCall("Couldn't send the report. Try again.") {
        val api = RetrofitClient.apiService
        when (target) {
            is ReportTarget.Event -> api.reportEvent("Bearer $token", target.id, ReportRequest(note))
            is ReportTarget.Photographer -> api.reportPhotographer("Bearer $token", target.handle, ReportRequest(note))
        }
    }
