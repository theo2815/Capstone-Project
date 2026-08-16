package com.quickpitik.mobile.ui.auth

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.foundation.background
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.animation.core.*
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LoginScreen(
    viewModel: AuthViewModel,
    onNavigateToRegister: () -> Unit,
    onNavigateToForgotPassword: () -> Unit,
    onLoginSuccess: (isPhotographer: Boolean) -> Unit
) {
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }

    val authState by viewModel.authState.collectAsState()

    // Handle single-time navigation routing on success state
    LaunchedEffect(authState) {
        if (authState is AuthState.Success) {
            val user = (authState as AuthState.Success).response.user
            onLoginSuccess(user.role.contains("PHOTO", ignoreCase = true))
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
                .navigationBarsPadding()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.Top
        ) {
            Spacer(modifier = Modifier.height(40.dp))

            // Logo Row
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.padding(bottom = 24.dp)
            ) {
                Box(
                    modifier = Modifier.size(28.dp),
                    contentAlignment = Alignment.Center
                ) {
                    androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                        drawCircle(
                            color = Ink,
                            radius = size.minDimension / 2f,
                            style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.5.dp.toPx())
                        )
                        drawCircle(
                            color = Fresh,
                            radius = size.minDimension / 5.6f
                        )
                    }
                }
                Text(
                    text = "QuickPitik",
                    style = Typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                    fontSize = 18.sp
                )
            }

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
                        visualTransformation = if (passwordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                        trailingIcon = {
                            Text(
                                text = if (passwordVisible) "HIDE" else "SHOW",
                                color = Slate,
                                style = Typography.labelMedium,
                                modifier = Modifier
                                    .clickable { passwordVisible = !passwordVisible }
                                    .padding(end = 12.dp)
                            )
                        },
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
                enabled = authState !is AuthState.Loading,
                shape = RoundedCornerShape(percent = 100),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Fresh,
                    contentColor = Bone,
                    disabledContainerColor = Fresh.copy(alpha = 0.8f),
                    disabledContentColor = Bone
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
                        text = "LOG IN",
                        style = Typography.labelLarge
                    )
                }
            }
            Spacer(modifier = Modifier.height(16.dp))

            // Recovery link. Website places this in the same slot — directly
            // under the submit button, above the "New here?" divider
            // (components/auth/login-form.tsx). Box gives the text a 48dp
            // touch target.
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .clickable(enabled = authState !is AuthState.Loading) {
                        viewModel.resetPasswordResetState()
                        onNavigateToForgotPassword()
                    },
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "FORGOT PASSWORD?",
                    style = Typography.labelMedium,
                    color = Slate
                )
            }
            Spacer(modifier = Modifier.height(8.dp))

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
                    text = "CREATE ACCOUNT",
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

            // Debug-only backend switcher. Renders nothing in a release build.
            // Sits last so the editorial top of the screen is untouched, and
            // lives on Login specifically because that is where every physical-
            // device test starts — and where no WebSocket is open yet, so a
            // host change takes effect with no restart. See DevServerRow.
            Spacer(modifier = Modifier.height(24.dp))
            DevServerRow()
            Spacer(modifier = Modifier.height(24.dp))

            if (authState is AuthState.Loading) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Bone),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        val infiniteTransition = rememberInfiniteTransition(label = "pulse")
                        val scale by infiniteTransition.animateFloat(
                            initialValue = 0.9f,
                            targetValue = 1.1f,
                            animationSpec = infiniteRepeatable(
                                animation = tween(1000, easing = LinearEasing),
                                repeatMode = RepeatMode.Reverse
                            ),
                            label = "scale"
                        )
                        
                        Box(
                            modifier = Modifier
                                .size(80.dp)
                                .graphicsLayer(scaleX = scale, scaleY = scale),
                            contentAlignment = Alignment.Center
                        ) {
                            androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                                drawCircle(
                                    color = Ink,
                                    radius = size.minDimension / 2f,
                                    style = androidx.compose.ui.graphics.drawscope.Stroke(width = 4.dp.toPx())
                                )
                                drawCircle(
                                    color = Fresh,
                                    radius = size.minDimension / 5.6f
                                )
                            }
                        }
                        
                        Spacer(modifier = Modifier.height(24.dp))
                        
                        Text(
                            text = "QuickPitik",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Ink,
                            fontSize = 22.sp
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Securing your connection...",
                            style = Typography.bodyMedium,
                            color = SlateSoft
                        )
                    }
                }
            }
        }
    }
}
