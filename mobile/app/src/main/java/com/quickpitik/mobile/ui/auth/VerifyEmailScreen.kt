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
import com.quickpitik.mobile.data.local.isPhotographerRole
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Slate

@Composable
fun VerifyEmailScreen(
    viewModel: AuthViewModel,
    token: String?,
    onNavigateToLogin: () -> Unit,
    onNavigateToDashboard: (Boolean) -> Unit,
) {
    val context = LocalContext.current
    // Asking on every entry is safe: the token is one-shot server-side, and
    // AuthViewModel sends it once and keeps the answer across rotation and
    // process death (the website guards the same hazard with a useRef).
    val key = token.orEmpty()
    LaunchedEffect(key) { viewModel.verifyEmail(key) }
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
                    text = "Verifying your email...",
                    color = Slate,
                    fontSize = 16.sp
                )
            }
            is EmailLinkState.Success -> {
                Icon(
                    imageVector = Icons.Default.CheckCircle,
                    contentDescription = "Success",
                    tint = Color(0xFF10B981),
                    modifier = Modifier.size(64.dp)
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Email Verified!",
                    color = Ink,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Your email address has been successfully verified. You can now use all features of your account.",
                    color = Slate,
                    fontSize = 16.sp,
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(32.dp))
                Button(
                    onClick = {
                        val session = SessionManager.getInstance(context)
                        val role = session.getUserRole()
                        if (session.getAccessToken() != null && role != null) {
                            onNavigateToDashboard(isPhotographerRole(role))
                        } else {
                            onNavigateToLogin()
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(50.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Ink, contentColor = Bone),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text("Continue", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
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
                    text = "Verification Failed",
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
