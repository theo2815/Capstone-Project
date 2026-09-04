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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.local.SessionManager
import com.quickpitik.mobile.data.remote.EmailChangeConfirmRequest
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Slate

enum class ConfirmEmailChangeState {
    Loading, Success, Error
}

@Composable
fun ConfirmEmailChangeScreen(
    token: String?,
    onNavigateToLogin: () -> Unit,
) {
    val context = LocalContext.current
    // rememberSaveable + requestFired: the token is one-shot server-side.
    // Plain remember would reset on rotation/process death, re-fire the
    // LaunchedEffect, and turn a real success into "Confirmation Failed".
    var uiState by rememberSaveable { mutableStateOf(ConfirmEmailChangeState.Loading) }
    var errorMessage by rememberSaveable { mutableStateOf("") }
    var successMessage by rememberSaveable { mutableStateOf("Email updated successfully! Please sign in again with your new address.") }
    var requestFired by rememberSaveable { mutableStateOf(false) }

    LaunchedEffect(token) {
        if (requestFired) return@LaunchedEffect
        requestFired = true
        if (token.isNullOrBlank()) {
            errorMessage = "Invalid or missing email confirmation token."
            uiState = ConfirmEmailChangeState.Error
            return@LaunchedEffect
        }
        try {
            val response = RetrofitClient.apiService.confirmEmailChange(EmailChangeConfirmRequest(token))
            if (response.success) {
                // Clear active local session to force re-login with the new email
                SessionManager.getInstance(context).clearSession()
                response.data?.message?.let { msg ->
                    if (msg.isNotBlank()) successMessage = msg
                }
                uiState = ConfirmEmailChangeState.Success
            } else {
                errorMessage = response.error ?: "Failed to confirm email change."
                uiState = ConfirmEmailChangeState.Error
            }
        } catch (e: Exception) {
            errorMessage = e.localizedMessage ?: "A network error occurred."
            uiState = ConfirmEmailChangeState.Error
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bone)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        when (uiState) {
            ConfirmEmailChangeState.Loading -> {
                CircularProgressIndicator(color = Ink)
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = "Confirming your new email...",
                    color = Slate,
                    fontSize = 16.sp
                )
            }
            ConfirmEmailChangeState.Success -> {
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
                    text = successMessage,
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
                    Text("Sign In Again", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                }
            }
            ConfirmEmailChangeState.Error -> {
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
                    text = errorMessage,
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
                    Text("Return to Login", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}
