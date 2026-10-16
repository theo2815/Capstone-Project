package com.quickpitik.mobile.ui.auth

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Slate

@Composable
fun ConfirmEmailChangeScreen(
    viewModel: AuthViewModel,
    token: String?,
    onNavigateToLogin: () -> Unit,
) {
    val context = LocalContext.current
    // Same one-shot rule as VerifyEmailScreen. On success the ViewModel has
    // already signed this phone out if its own account changed.
    val key = token.orEmpty()
    LaunchedEffect(key) { viewModel.confirmEmailChange(key) }
    val links by viewModel.emailLinks.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bone)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        when (val state = links[key]) {
            null -> {
                CircularProgressIndicator(color = Ink)
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Confirming your new email...",
                    color = Slate,
                    fontSize = 16.sp
                )
            }
            is EmailLinkState.Success -> {
                // Still signed in: the link changed another account (or the check
                // got no answer), so this phone was not signed out.
                val signedIn = remember { SessionManager.getInstance(context).getAccessToken() != null }
                Icon(
                    imageVector = Icons.Default.CheckCircle,
                    contentDescription = "Success",
                    tint = Color(0xFF10B981),
                    modifier = Modifier.size(64.dp)
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Email Change Confirmed!",
                    color = Ink,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = if (signedIn) "Email change confirmed. This phone is still signed in." else state.message?.takeIf { it.isNotBlank() }
                        ?: "Email updated successfully! Please sign in again with your new address.",
                    color = Slate,
                    fontSize = 16.sp,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                Button(
                    onClick = onNavigateToLogin,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Ink, contentColor = Bone),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(if (signedIn) "Go back" else "Sign In Again", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                }
            }
            is EmailLinkState.Error -> {
                // Signed in, the exit goes back to the screen the link opened over.
                val signedIn = remember { SessionManager.getInstance(context).getAccessToken() != null }
                Icon(
                    imageVector = Icons.Default.Warning,
                    contentDescription = "Error",
                    tint = Color(0xFFEF4444),
                    modifier = Modifier.size(64.dp)
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Confirmation Failed",
                    color = Ink,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = state.message,
                    color = Slate,
                    fontSize = 16.sp,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                Button(
                    onClick = onNavigateToLogin,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Ink, contentColor = Bone),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(if (signedIn) "Go back" else "Return to Login", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}
