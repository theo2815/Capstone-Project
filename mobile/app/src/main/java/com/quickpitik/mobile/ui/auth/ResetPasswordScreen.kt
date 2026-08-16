package com.quickpitik.mobile.ui.auth

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.ui.theme.*

// Mobile counterpart to website /(auth)/reset-password
// (components/auth/reset-password-form.tsx).
//
// The web page reads its token from `?token=` because the reset email links
// straight to it. EmailService hardcodes that link to the website origin and
// the backend is frozen (Build Mandate rule 2), so this screen cannot receive
// a token the same way — it takes one PASTED instead, and the web's dead-end
// "missing-token" state collapses into an empty required field. Two states
// here vs the web's three; everything else is a faithful port.
//
// In dev this is the smoother path anyway: EmailService runs in devMode and
// prints the whole link to the backend console, so the code is right there.

@Composable
fun ResetPasswordScreen(
    viewModel: AuthViewModel,
    onNavigateToLogin: () -> Unit,
) {
    val resetState by viewModel.passwordResetState.collectAsState()

    var token by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }
    var tokenError by remember { mutableStateOf<String?>(null) }
    var passwordError by remember { mutableStateOf<String?>(null) }
    var confirmError by remember { mutableStateOf<String?>(null) }

    // Arriving from the forgot screen, passwordResetState is still Success from
    // the send — clear it so this screen opens on its form, not its receipt.
    LaunchedEffect(Unit) { viewModel.resetPasswordResetState() }

    val done = resetState is PasswordResetState.Success
    val loading = resetState is PasswordResetState.Loading

    // Any keystroke clears a stale server error; the user is already fixing it.
    val clearSubmitError = {
        if (resetState is PasswordResetState.Error) viewModel.resetPasswordResetState()
    }

    AuthScreenScaffold {
        AnimatedContent(
            targetState = done,
            transitionSpec = { fadeIn(tween(200)) togetherWith fadeOut(tween(200)) },
            label = "reset-password-state",
        ) { isDone ->
            Column {
                if (isDone) {
                    ResetDoneView(onSignIn = onNavigateToLogin)
                } else {
                    ResetFormView(
                        token = token,
                        onTokenChange = {
                            token = it
                            if (tokenError != null) tokenError = null
                            clearSubmitError()
                        },
                        tokenError = tokenError,
                        password = password,
                        onPasswordChange = {
                            password = it
                            if (passwordError != null) passwordError = null
                            clearSubmitError()
                        },
                        passwordError = passwordError,
                        confirmPassword = confirmPassword,
                        onConfirmChange = {
                            confirmPassword = it
                            if (confirmError != null) confirmError = null
                            clearSubmitError()
                        },
                        confirmError = confirmError,
                        submitError = (resetState as? PasswordResetState.Error)?.message,
                        loading = loading,
                        onSubmit = {
                            val tokenErr = if (token.isBlank()) {
                                "Paste the reset code from your email."
                            } else null
                            val passErr = validateNewPassword(password)
                            val confirmErr = when {
                                confirmPassword.isEmpty() -> "Please confirm your new password."
                                password != confirmPassword -> "Passwords don't match."
                                else -> null
                            }
                            tokenError = tokenErr
                            passwordError = passErr
                            confirmError = confirmErr
                            if (tokenErr == null && passErr == null && confirmErr == null) {
                                viewModel.confirmPasswordReset(token, password)
                            }
                        },
                        onBackToLogin = onNavigateToLogin,
                    )
                }
            }
        }
    }
}

@Composable
private fun ResetFormView(
    token: String,
    onTokenChange: (String) -> Unit,
    tokenError: String?,
    password: String,
    onPasswordChange: (String) -> Unit,
    passwordError: String?,
    confirmPassword: String,
    onConfirmChange: (String) -> Unit,
    confirmError: String?,
    submitError: String?,
    loading: Boolean,
    onSubmit: () -> Unit,
    onBackToLogin: () -> Unit,
) {
    AuthKicker(text = "Reset access")
    Spacer(modifier = Modifier.height(12.dp))

    AuthHeadline(lead = "Set a new", accent = "password.")
    Spacer(modifier = Modifier.height(12.dp))

    Text(
        text = "Pick a strong one. You'll use it next time you sign in.",
        style = Typography.bodyLarge,
        color = InkSoft,
    )
    Spacer(modifier = Modifier.height(32.dp))

    Column(verticalArrangement = Arrangement.spacedBy(20.dp)) {
        AuthField(
            label = "Reset code",
            value = token,
            onValueChange = onTokenChange,
            placeholder = "Paste from your reset link",
            labelSuffix = "from your email",
            enabled = !loading,
            error = tokenError,
        )
        AuthField(
            label = "New password",
            value = password,
            onValueChange = onPasswordChange,
            placeholder = "••••••••",
            labelSuffix = "min. $PASSWORD_MIN characters",
            keyboardType = KeyboardType.Password,
            isPassword = true,
            enabled = !loading,
            error = passwordError,
        )
        AuthField(
            label = "Confirm new password",
            value = confirmPassword,
            onValueChange = onConfirmChange,
            placeholder = "••••••••",
            keyboardType = KeyboardType.Password,
            imeAction = ImeAction.Done,
            isPassword = true,
            enabled = !loading,
            error = confirmError,
        )
    }

    if (submitError != null) {
        Spacer(modifier = Modifier.height(16.dp))
        AuthSubmitError(submitError)
    }
    Spacer(modifier = Modifier.height(24.dp))

    PrimaryCta(
        text = "Reset password →",
        onClick = onSubmit,
        loading = loading,
        modifier = Modifier.fillMaxWidth(),
    )
    Spacer(modifier = Modifier.height(24.dp))

    BackToSignIn(onClick = onBackToLogin, enabled = !loading)
}

@Composable
private fun ResetDoneView(onSignIn: () -> Unit) {
    AuthKicker(text = "Reset access", suffix = "Done")
    Spacer(modifier = Modifier.height(12.dp))

    AuthHeadline(lead = "Password", accent = "reset.")
    Spacer(modifier = Modifier.height(12.dp))

    Text(
        text = "Sign in with your new password to continue to your photos.",
        style = Typography.bodyLarge,
        color = InkSoft,
    )
    Spacer(modifier = Modifier.height(16.dp))

    // confirmReset revokes every refresh token on the account, so any other
    // signed-in device drops to the login screen on its next call. Say so
    // rather than letting it look like a bug.
    Text(
        text = "Any other devices signed in to this account have been signed out.",
        style = Typography.bodyMedium,
        color = Slate,
    )
    Spacer(modifier = Modifier.height(24.dp))

    PrimaryCta(
        text = "Sign in →",
        onClick = onSignIn,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Preview(showBackground = true)
@Composable
private fun ResetFormPreview() {
    AuthScreenScaffold {
        ResetFormView(
            token = "",
            onTokenChange = {},
            tokenError = null,
            password = "",
            onPasswordChange = {},
            passwordError = null,
            confirmPassword = "",
            onConfirmChange = {},
            confirmError = null,
            submitError = null,
            loading = false,
            onSubmit = {},
            onBackToLogin = {},
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun ResetFormErrorPreview() {
    AuthScreenScaffold {
        ResetFormView(
            token = "3vJqk9…",
            onTokenChange = {},
            tokenError = null,
            password = "short",
            onPasswordChange = {},
            passwordError = "Password must be at least 8 characters.",
            confirmPassword = "shorter",
            onConfirmChange = {},
            confirmError = "Passwords don't match.",
            submitError = "Could not reset your password. The link may have expired — request a new one.",
            loading = false,
            onSubmit = {},
            onBackToLogin = {},
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun ResetDonePreview() {
    AuthScreenScaffold {
        ResetDoneView(onSignIn = {})
    }
}
