package com.quickpitik.mobile.ui.auth

import androidx.compose.foundation.BorderStroke
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
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RegisterScreen(
    onNavigateToLogin: () -> Unit,
    onRegisterSuccess: (isPhotographer: Boolean) -> Unit
) {
    var name by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var isPhotographer by remember { mutableStateOf(false) }

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
            verticalArrangement = Arrangement.Center
        ) {
            Spacer(modifier = Modifier.height(24.dp))
            
            // Eyebrow kicker
            Text(
                text = "CREATE ACCOUNT",
                style = Typography.labelLarge,
                color = Slate
            )
            Spacer(modifier = Modifier.height(12.dp))

            // Display Title
            Text(
                text = buildAnnotatedString {
                    append("Join\n")
                    withStyle(style = SpanStyle(color = Fresh)) {
                        append("QuickPitik.")
                    }
                },
                style = Typography.displayLarge,
                color = Ink
            )
            Spacer(modifier = Modifier.height(24.dp))

            // Role selection buttons (Matches custom web layout buttons)
            Text(
                text = "ACCOUNT TYPE",
                style = Typography.labelMedium,
                color = Slate
            )
            Spacer(modifier = Modifier.height(8.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Runner Option
                Card(
                    onClick = { isPhotographer = false },
                    border = BorderStroke(
                        width = 1.5.dp,
                        color = if (!isPhotographer) Ink else Line
                    ),
                    colors = CardDefaults.cardColors(
                        containerColor = if (!isPhotographer) BoneDeep else Bone
                    ),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("01", style = Typography.labelMedium, color = SlateSoft)
                            if (!isPhotographer) {
                                Surface(
                                    shape = RoundedCornerShape(percent = 100),
                                    color = Fresh,
                                    modifier = Modifier.size(6.dp)
                                ) {}
                            }
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("I run", style = Typography.titleMedium, color = Ink)
                        Text("Find your photos", style = Typography.bodyMedium, color = SlateSoft)
                    }
                }

                // Photographer Option
                Card(
                    onClick = { isPhotographer = true },
                    border = BorderStroke(
                        width = 1.5.dp,
                        color = if (isPhotographer) Ink else Line
                    ),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isPhotographer) BoneDeep else Bone
                    ),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("02", style = Typography.labelMedium, color = SlateSoft)
                            if (isPhotographer) {
                                Surface(
                                    shape = RoundedCornerShape(percent = 100),
                                    color = Fresh,
                                    modifier = Modifier.size(6.dp)
                                ) {}
                            }
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Text("I shoot", style = Typography.titleMedium, color = Ink)
                        Text("Sell your photos", style = Typography.bodyMedium, color = SlateSoft)
                    }
                }
            }
            Spacer(modifier = Modifier.height(32.dp))

            // Inputs
            Column(
                verticalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                // Name Input
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = "FULL NAME",
                        style = Typography.labelMedium,
                        color = Slate
                    )
                    TextField(
                        value = name,
                        onValueChange = { name = it },
                        placeholder = { Text("Juan dela Cruz", color = SlateSoft) },
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

                // Email Input
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = "EMAIL",
                        style = Typography.labelMedium,
                        color = Slate
                    )
                    TextField(
                        value = email,
                        onValueChange = { email = it },
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

                // Password Input
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = buildAnnotatedString {
                            append("PASSWORD")
                            withStyle(style = SpanStyle(color = SlateSoft)) {
                                append("  min. 8 char")
                            }
                        },
                        style = Typography.labelMedium,
                        color = Slate
                    )
                    TextField(
                        value = password,
                        onValueChange = { password = it },
                        placeholder = { Text("••••••••", color = SlateSoft) },
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
            Spacer(modifier = Modifier.height(40.dp))

            // CTA Button
            Button(
                onClick = { onRegisterSuccess(isPhotographer) },
                shape = RoundedCornerShape(percent = 100),
                colors = ButtonDefaults.buttonColors(
                    containerColor = Fresh,
                    contentColor = Bone
                ),
                contentPadding = PaddingValues(vertical = 16.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "CREATE ACCOUNT →",
                    style = Typography.labelLarge
                )
            }
            Spacer(modifier = Modifier.height(24.dp))

            // Redirect Footer
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "ALREADY HAVE AN ACCOUNT? ",
                    style = Typography.labelMedium,
                    color = Slate
                )
                Text(
                    text = "SIGN IN →",
                    style = Typography.labelMedium,
                    color = Ink,
                    modifier = Modifier.clickable { onNavigateToLogin() }
                )
            }
            
            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}
