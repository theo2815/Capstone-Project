package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.ui.theme.*

/**
 * Standard sticky top bar across all runner tabs (Browse, Profile, Orders, Settings).
 * Includes the page kicker, QuickPitik title, optional trailing actions (e.g. inbox bell),
 * and the user avatar circle with a logout dropdown.
 */
@Composable
fun RunnerTopBar(
    kicker: String,
    title: String = "QuickPitik",
    onLogout: () -> Unit = {},
    modifier: Modifier = Modifier,
    trailingContent: (@Composable () -> Unit)? = null
) {
    val context = LocalContext.current
    val sessionManager = remember { SessionManager.getInstance(context) }
    val userName = remember { sessionManager.getUserName() ?: "Runner" }
    var menuExpanded by remember { mutableStateOf(false) }

    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(48.dp)
            .padding(horizontal = 24.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            Text(
                text = kicker,
                style = Typography.labelMedium,
                color = Slate
            )
            Text(
                text = title,
                style = Typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = Ink
            )
        }

        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            trailingContent?.invoke()

            Box {
                Box(
                    modifier = Modifier
                        .size(40.dp)
                        .clip(CircleShape)
                        .background(BoneDeep)
                        .clickable { menuExpanded = true },
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = userName.take(1).uppercase(),
                        color = Ink,
                        style = Typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                }
                DropdownMenu(
                    expanded = menuExpanded,
                    onDismissRequest = { menuExpanded = false },
                    modifier = Modifier.background(BoneDeep)
                ) {
                    DropdownMenuItem(
                        text = { Text("Log out", color = ErrorRed) },
                        onClick = {
                            menuExpanded = false
                            onLogout()
                        }
                    )
                }
            }
        }
    }
}
