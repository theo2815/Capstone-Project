package com.quickpitik.mobile.ui.theme

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalHapticFeedback

// Semantic haptic vocabulary. The Compose BOM in use (2023.08) exposes only
// LongPress + TextHandleMove — newer `HapticFeedbackType.Confirm/Reject` land
// in Compose UI 1.7. Map our intents onto the two we have so call sites read
// as "confirm this" / "remove this" instead of leaking the API name.
enum class QpHaptic { CONFIRM, REMOVE }

@Composable
fun rememberQpHaptic(): (QpHaptic) -> Unit {
    val haptics = LocalHapticFeedback.current
    return remember(haptics) {
        { type ->
            when (type) {
                QpHaptic.CONFIRM -> haptics.performHapticFeedback(HapticFeedbackType.LongPress)
                QpHaptic.REMOVE -> haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove)
            }
        }
    }
}
