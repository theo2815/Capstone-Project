package com.quickpitik.mobile.ui.auth

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LoginScreen(
    viewModel: AuthViewModel,
    onNavigateToRegister: () -> Unit,
    onLoginSuccess: (isPhotographer: Boolean) -> Unit
) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var isPhotographerMode by remember { mutableStateOf(false) }

    val authState by viewModel.authState.collectAsState()

    // Handle single-time navigation routing on success state
    LaunchedEffect(authState) {
        if (authState is AuthState.Success) {
            val user = (authState as AuthState.Success).response.user
            onLoginSuccess(user.role == "PHOTOGRAPHER")
            viewModel.resetState()
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Bone
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp)
                .statusBarsPadding()
                .navigationBarsPadding(),
            verticalArrangement = Arrangement.Center
        ) {
            // Eyebrow kicker
            Text(
                text = "LOG IN",
                style = Typography.labelLarge,
                color = Slate
            )
            Spacer(modifier = Modifier.height(12.dp))

            // Display Title
            Text(
                text = buildAnnotatedString {
                    append("Welcome\n")
                    withStyle(style = SpanStyle(color = Fresh)) {
                        append("back.")
                    }
                },
                style = Typography.displayLarge,
                color = Ink
            )
            Spacer(modifier = Modifier.height(12.dp))

            // Subtitle
            Text(
                text = "Continue to your photos.",
                style = Typography.bodyLarge,
                color = InkSoft
            )
            Spacer(modifier = Modifier.height(32.dp))

            // Role selector row (useful to toggle testing defaults)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = { isPhotographerMode = false },
                    enabled = authState !is AuthState.Loading,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (!isPhotographerMode) Ink else BoneDeep,
                        contentColor = if (!isPhotographerMode) Bone else Ink
                    ),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Runner Mode", style = Typography.labelMedium)
                }

                Button(
                    onClick = { isPhotographerMode = true },
                    enabled = authState !is AuthState.Loading,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isPhotographerMode) Ink else BoneDeep,
                        contentColor = if (isPhotographerMode) Bone else Ink
                    ),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Camera Mode", style = Typography.labelMedium)
                }
            }
            Spacer(modifier = Modifier.height(28.dp))

            // Fields
            Column(
                verticalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                // Email Field
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = "EMAIL",
                        style = Typography.labelMedium,
                        color = Slate
                    )
                    TextField(
                        value = email,
                        onValueChange = { email = it },
                        enabled = authState !is AuthState.Loading,
                        placeholder = { Text("you@example.com", color = SlateSoft) },
                        singleLine = true,
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = Color.Transparent,
                            unfocusedContainerColor = Color.Transparent,
                            disabledContainerColor = Color.Transparent,
                            focusedIndicatorColor = Fresh,
                            unfocusedIndicatorColor = Line,
                            focusedTextColor = Ink,
                            unfocusedTextColor = InkSoft
                        ),
                        modifier = Modifier.fillMaxWidth()
                    )
                }

                // Password Field
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = "PASSWORD",
                        style = Typography.labelMedium,
                        color = Slate
                    )
                    TextField(
                        value = password,
                        onValueChange = { password = it },
                        enabled = authState !is AuthState.Loading,
                        placeholder = { Text("Your password", color = SlateSoft) },
                        singleLine = true,
                        visualTransformation = PasswordVisualTransformation(),
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = Color.Transparent,
                            unfocusedContainerColor = Color.Transparent,
                            disabledContainerColor = Color.Transparent,
                            focusedIndicatorColor = Fresh,
                            unfocusedIndicatorColor = Line,
                            focusedTextColor = Ink,
                            unfocusedTextColor = InkSoft
                        ),
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            }
            Spacer(modifier = Modifier.height(20.dp))

            // Dynamic Error Message Text
            if (authState is AuthState.Error) {
                Text(
                    text = (authState as AuthState.Error).message,
                    color = Color.Red,
                    style = Typography.bodyMedium,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(16.dp))
            }

            // Action Button
            Button(
                onClick = { viewModel.login(email.trim(), password) },
                enabled = authState !is AuthState.Loading && email.isNotBlank() && password.length >= 8,
                shape = RoundedCornerShape(percent = 100),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Fresh,
                    contentColor = Bone
                ),
                contentPadding = PaddingValues(vertical = 16.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                if (authState is AuthState.Loading) {
                    CircularProgressIndicator(
                        color = Bone,
                        modifier = Modifier.size(24.dp)
                    )
                } else {
                    Text(
                        text = "LOG IN →",
                        style = Typography.labelLarge
                    )
                }
            }
            Spacer(modifier = Modifier.height(24.dp))

            // Bottom Redirect
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "NEW HERE? ",
                    style = Typography.labelMedium,
                    color = Slate
                )
                Text(
                    text = "CREATE ACCOUNT →",
                    style = Typography.labelMedium,
                    color = Ink,
                    modifier = Modifier.clickable { 
                        if (authState !is AuthState.Loading) {
                            viewModel.resetState()
                            onNavigateToRegister()
                        }
                    }
                )
            }
        }
    }
}
