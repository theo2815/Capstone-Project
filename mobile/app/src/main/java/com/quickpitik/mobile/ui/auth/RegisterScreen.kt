package com.quickpitik.mobile.ui.auth

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.local.isPhotographerRole
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.BrandLogo
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.Typography

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RegisterScreen(
    viewModel: AuthViewModel,
    onNavigateToLogin: () -> Unit,
    onRegisterSuccess: (isPhotographer: Boolean) -> Unit
) {
    var name by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }
    var passwordVisible by remember { mutableStateOf(false) }
    var confirmPasswordVisible by remember { mutableStateOf(false) }
    var validationError by remember { mutableStateOf<String?>(null) }
    var isPhotographer by remember { mutableStateOf(false) }

    val authState by viewModel.authState.collectAsState()

    // Handle single-time navigation routing on success state
    LaunchedEffect(authState) {
        if (authState is AuthState.Success) {
            val user = (authState as AuthState.Success).response.user
            onRegisterSuccess(isPhotographerRole(user.role))
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

            BrandLogo(modifier = Modifier.padding(bottom = 24.dp))

            // Eyebrow kicker
            Text(
                text = "CREATE ACCOUNT",
                style = Typography.labelLarge,
                color = Slate
            )
            Spacer(modifier = Modifier.height(12.dp))

            // Hero title — Anton uppercase (Finish Line). Uppercased in the
            // literals because HeroText can't carry the two-tone span.
            Text(
                text = buildAnnotatedString {
                    append("JOIN\n")
                    withStyle(style = SpanStyle(color = Fresh)) {
                        append("QUICKPITIK.")
                    }
                },
                style = Typography.displayLarge,
                color = Ink
            )
            Spacer(modifier = Modifier.height(24.dp))

            // "Continue with Google" + divider — website parity: the block
            // sits above the role pivot on /register. A brand-new Google
            // account picks its role in the sheet below, not from the cards —
            // the toggle's default must never silently decide a permanent
            // choice. Renders nothing when no client ID is compiled in.
            GoogleSignInRow(
                enabled = authState !is AuthState.Loading,
                onIdToken = viewModel::googleLogin,
                onError = viewModel::showError,
            )

            // Role selection buttons
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
                    onClick = { 
                        if (authState !is AuthState.Loading) {
                            isPhotographer = false 
                        }
                    },
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
                    onClick = {
                        if (authState !is AuthState.Loading) {
                            isPhotographer = true 
                        }
                    },
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
                        enabled = authState !is AuthState.Loading,
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
                        onValueChange = { 
                            password = it 
                            if (validationError != null) validationError = null
                        },
                        enabled = authState !is AuthState.Loading,
                        placeholder = { Text("••••••••", color = SlateSoft) },
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

                // Re-enter Password Input
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = "RE-ENTER PASSWORD",
                        style = Typography.labelMedium,
                        color = Slate
                    )
                    TextField(
                        value = confirmPassword,
                        onValueChange = { 
                            confirmPassword = it 
                            if (validationError != null) validationError = null
                        },
                        enabled = authState !is AuthState.Loading,
                        placeholder = { Text("••••••••", color = SlateSoft) },
                        singleLine = true,
                        visualTransformation = if (confirmPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                        trailingIcon = {
                            Text(
                                text = if (confirmPasswordVisible) "HIDE" else "SHOW",
                                color = Slate,
                                style = Typography.labelMedium,
                                modifier = Modifier
                                    .clickable { confirmPasswordVisible = !confirmPasswordVisible }
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

            // Dynamic Error Message Text (Client validation or Server Error)
            val displayedError = validationError ?: (authState as? AuthState.Error)?.message
            if (displayedError != null) {
                Text(
                    text = displayedError,
                    color = ErrorRed,
                    style = Typography.bodyMedium,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(16.dp))
            }

            // CTA Button
            Button(
                onClick = {
                    // Shared validators, not inline rules: the hand-rolled
                    // `length < 8` check silently accepted a >72-byte password
                    // that bcrypt truncates — the reset screen enforced the
                    // ceiling while register didn't. One rulebook for both.
                    val emailProblem = validateEmail(email)
                    val passwordProblem = validateNewPassword(password)
                    when {
                        emailProblem != null -> validationError = emailProblem
                        passwordProblem != null -> validationError = passwordProblem
                        password != confirmPassword -> {
                            validationError = "Passwords do not match"
                        }
                        else -> {
                            validationError = null
                            viewModel.register(name.trim(), email.trim(), password, isPhotographer)
                        }
                    }
                },
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
                        text = "CREATE ACCOUNT",
                        style = Typography.labelLarge
                    )
                }
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
                    text = "SIGN IN",
                    style = Typography.labelMedium,
                    color = Ink,
                    modifier = Modifier.clickable { 
                        if (authState !is AuthState.Loading) {
                            viewModel.resetState()
                            onNavigateToLogin()
                        }
                    }
                )
            }
            
            Spacer(modifier = Modifier.height(24.dp))
        }

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
                    
                    BrandLogo(
                        modifier = Modifier
                            .graphicsLayer(scaleX = scale, scaleY = scale)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Creating your account...",
                        style = Typography.bodyMedium,
                        color = SlateSoft
                    )
                }
            }

            // Brand-new Google account — role pick before it exists.
            if (authState is AuthState.GoogleRoleRequired) {
                GoogleRoleSheet(
                    onPick = viewModel::completeGoogleSignup,
                    onDismiss = viewModel::cancelGoogleSignup,
                )
            }
        }
    }
}
