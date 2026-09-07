package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.layout.*
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import com.quickpitik.mobile.ui.theme.*
import androidx.compose.ui.unit.dp

// Event coupon editor (V47) — lives on the event, not Settings. Only for a
// covered PAID event. code (UPPERCASE 4-16), percentOff 1-50 (or 100 for a free
// giveaway on an event you created), active toggle, optional usage limit. The
// backend enforces coverage/eligibility; this only mirrors the shape.
// ponytail: 50 mirrors the backend default COUPON_MAX_PERCENT (mobile has no
// platform-fees fetch to read a raised cap); the sheet only opens for an
// owned/created event, so 100 is always the valid giveaway here. No expiresAt
// editor — the loaded value is preserved on save so a web-set expiry survives.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun EventCouponSheet(
    eventId: String,
    eventName: String,
    viewModel: PhotographerDashboardViewModel,
    onDismiss: () -> Unit,
) {
    val state by viewModel.couponState.collectAsState()
    val actionMessage by viewModel.couponActionMessage.collectAsState()
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    LaunchedEffect(eventId) { viewModel.loadCoupon(eventId) }
    DisposableEffect(Unit) { onDispose { viewModel.clearCouponMessage() } }

    val loaded = (state as? CouponUiState.Loaded)?.coupon
    var code by remember(loaded) { mutableStateOf(loaded?.code ?: "") }
    var percent by remember(loaded) { mutableStateOf(loaded?.percentOff?.toString() ?: "") }
    var active by remember(loaded) { mutableStateOf(loaded?.active ?: true) }
    var usageLimit by remember(loaded) { mutableStateOf(loaded?.usageLimit?.toString() ?: "") }

    val codeValid = Regex("^[A-Z0-9]{4,16}$").matches(code)
    val pct = percent.trim().toIntOrNull()
    val pctValid = pct != null && (pct in 1..50 || pct == 100)
    val canSave = codeValid && pctValid

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Bone,
        contentColor = Ink,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .imePadding()
                .navigationBarsPadding()
                .padding(horizontal = 24.dp, vertical = 8.dp),
        ) {
            Kicker("Event coupon")
            Spacer(Modifier.height(6.dp))
            Text(eventName, style = Typography.titleMedium, fontWeight = FontWeight.SemiBold, color = Ink)
            Spacer(Modifier.height(20.dp))

            when (state) {
                is CouponUiState.Loading -> Box(
                    modifier = Modifier.fillMaxWidth().padding(24.dp),
                    contentAlignment = Alignment.Center,
                ) { CircularProgressIndicator(color = Fresh, strokeWidth = 2.dp) }

                is CouponUiState.Error -> {
                    Text((state as CouponUiState.Error).message, style = Typography.bodyMedium, color = ErrorRed)
                    Spacer(Modifier.height(12.dp))
                    GhostCta(text = "Retry", onClick = { viewModel.loadCoupon(eventId) }, modifier = Modifier.fillMaxWidth())
                }

                is CouponUiState.Loaded -> {
                    CouponField("Code", code, { code = it.uppercase() }, "SAVE20")
                    if (code.isNotBlank() && !codeValid) FieldError("4-16 letters/numbers, e.g. SAVE20.")
                    Spacer(Modifier.height(16.dp))
                    CouponField("Percent off · 1-50 (100 = giveaway)", percent, { percent = it.filter { c -> c.isDigit() } }, "20")
                    if (percent.isNotBlank() && !pctValid) FieldError("Enter 1-50, or 100 for a free giveaway.")
                    Spacer(Modifier.height(16.dp))
                    CouponField("Usage limit · optional", usageLimit, { usageLimit = it.filter { c -> c.isDigit() } }, "No limit")

                    Spacer(Modifier.height(16.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Switch(
                            checked = active,
                            onCheckedChange = { active = it },
                            colors = SwitchDefaults.colors(checkedTrackColor = Ink, checkedThumbColor = SurfaceWhite),
                        )
                        Spacer(Modifier.width(12.dp))
                        Text(if (active) "Active" else "Paused", style = Typography.bodyMedium, color = Ink)
                    }

                    actionMessage?.let {
                        Spacer(Modifier.height(12.dp))
                        Text(it, style = Typography.bodyMedium, color = ErrorRed)
                    }

                    Spacer(Modifier.height(20.dp))
                    PrimaryCta(
                        text = if (loaded == null) "Create coupon" else "Save coupon",
                        onClick = {
                            viewModel.saveCoupon(
                                eventId = eventId,
                                code = code,
                                percentOff = pct ?: 0,
                                active = active,
                                // Preserve any expiry set elsewhere (e.g. the website) — this
                                // editor has no expiresAt field, and the PUT is a full upsert,
                                // so sending null would silently clear it.
                                expiresAt = loaded?.expiresAt,
                                usageLimit = usageLimit.trim().toIntOrNull(),
                            )
                        },
                        enabled = canSave,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    if (loaded != null) {
                        Spacer(Modifier.height(6.dp))
                        TextButton(onClick = { viewModel.deleteCoupon(eventId) }, modifier = Modifier.fillMaxWidth()) {
                            Text("Remove coupon", style = Typography.bodyMedium, color = ErrorRed)
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                }
            }
        }
    }
}

@Composable
private fun CouponField(label: String, value: String, onValueChange: (String) -> Unit, placeholder: String) {
    Kicker(label)
    Spacer(Modifier.height(8.dp))
    TextField(
        value = value,
        onValueChange = onValueChange,
        placeholder = { Text(placeholder, color = SlateSoft) },
        singleLine = true,
        colors = TextFieldDefaults.colors(
            focusedContainerColor = BoneDeep,
            unfocusedContainerColor = BoneDeep,
            focusedIndicatorColor = Fresh,
            unfocusedIndicatorColor = Color.Transparent,
            focusedTextColor = Ink,
            unfocusedTextColor = InkSoft,
        ),
        shape = FieldShape,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun FieldError(text: String) {
    Spacer(Modifier.height(6.dp))
    Text(text, style = Typography.bodySmall, color = ErrorRed)
}
