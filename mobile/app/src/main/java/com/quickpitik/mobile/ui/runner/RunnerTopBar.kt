package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.local.ViewMode
import com.quickpitik.mobile.data.local.isPhotographerRole
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.BrandLogo
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.Typography

/**
 * Standard sticky top bar across all runner tabs (Browse, Profile, Orders, Settings).
 * Includes the page kicker, QuickPitik logo, optional trailing actions (e.g. inbox bell),
 * and the user avatar circle with a logout dropdown.
 */
@Composable
fun RunnerTopBar(
    kicker: String,
    userName: String? = null,
    avatarUrl: String? = null,
    onLogout: () -> Unit = {},
    modifier: Modifier = Modifier,
    trailingContent: (@Composable () -> Unit)? = null
) {
    val context = LocalContext.current
    val sessionManager = remember { SessionManager.getInstance(context) }
    val displayName = userName ?: sessionManager.getUserName() ?: "Runner"
    val resolvedAvatarUrl = RetrofitClient.resolveImageUrl(
        avatarUrl ?: sessionManager.getAvatarUrl()
    )
    var menuExpanded by remember { mutableStateOf(false) }

    Row(
        modifier = modifier
            .fillMaxWidth()
            .defaultMinSize(minHeight = 48.dp)
            .padding(horizontal = 24.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column {
            BrandLogo(compact = true)
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = kicker,
                style = Typography.labelMedium,
                color = Slate
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
                    // Real avatar when one exists (SessionManager caches it;
                    // the photographer top bar already did this) — initials
                    // only as the fallback.
                    if (!resolvedAvatarUrl.isNullOrBlank()) {
                        AsyncImage(
                            model = resolvedAvatarUrl,
                            contentDescription = "Profile menu",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize().clip(CircleShape),
                        )
                    } else {
                        Text(
                            text = displayName.take(1).uppercase(),
                            color = Ink,
                            style = Typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
                DropdownMenu(
                    expanded = menuExpanded,
                    onDismissRequest = { menuExpanded = false },
                    modifier = Modifier.background(BoneDeep)
                ) {
                    // Only a PHOTOGRAPHER browsing in runner view sees this —
                    // the true role never changes while switched (web
                    // useViewModeStore parity). The bar reads the role itself
                    // and signals via ViewMode so six host screens don't each
                    // thread a callback; MainActivity performs the navigation.
                    if (isPhotographerRole(sessionManager.getUserRole())) {
                        DropdownMenuItem(
                            text = { Text("Switch to photographer", color = Ink) },
                            onClick = {
                                menuExpanded = false
                                ViewMode.requestSwitchToPhotographer()
                            }
                        )
                    }
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
