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

// Mobile counterpart to website /(auth)/forgot-password
// (components/auth/forgot-password-form.tsx). Two states, same as the web:
// the request form, and the "we sent it" confirmation.
//
// One deliberate divergence, forced by a backend the Build Mandate freezes:
// EmailService builds the reset link against the WEBSITE origin
// ($frontendOrigin/reset-password?token=…), so the link can never open this
// app. The Sent state therefore points the user at ResetPasswordScreen, where
// the code is pasted by hand. See the module ADR for the full rationale.

@Composable
fun ForgotPasswordScreen(
    viewModel: AuthViewModel,
    onNavigateToLogin: () -> Unit,
    onNavigateToReset: () -> Unit,
) {
    val resetState by viewModel.passwordResetState.collectAsState()

    var email by remember { mutableStateOf("") }
    var submittedEmail by remember { mutableStateOf("") }
    var emailError by remember { mutableStateOf<String?>(null) }

    // The two recovery screens share one hoisted AuthViewModel, so a stale
    // Success from a previous visit would render this screen already-Sent.
    LaunchedEffect(Unit) { viewModel.resetPasswordResetState() }

    val sent = resetState is PasswordResetState.Success
    val loading = resetState is PasswordResetState.Loading

    AuthScreenScaffold {
        AnimatedContent(
            targetState = sent,
            transitionSpec = { fadeIn(tween(200)) togetherWith fadeOut(tween(200)) },
            label = "forgot-password-state",
        ) { isSent ->
            Column {
                if (isSent) {
                    ForgotSentView(
                        submittedEmail = submittedEmail,
                        onTryAnother = {
                            email = ""
                            emailError = null
                            viewModel.resetPasswordResetState()
                        },
                        onHaveCode = onNavigateToReset,
                        onBackToLogin = onNavigateToLogin,
                    )
                } else {
                    ForgotFormView(
                        email = email,
                        onEmailChange = {
                            email = it
                            if (emailError != null) emailError = null
                            if (resetState is PasswordResetState.Error) {
                                viewModel.resetPasswordResetState()
                            }
                        },
                        emailError = emailError,
                        submitError = (resetState as? PasswordResetState.Error)?.message,
                        loading = loading,
                        onSubmit = {
                            val fieldError = validateEmail(email)
                            emailError = fieldError
                            if (fieldError == null) {
                                submittedEmail = email.trim()
                                viewModel.requestPasswordReset(email)
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
private fun ForgotFormView(
    email: String,
    onEmailChange: (String) -> Unit,
    emailError: String?,
    submitError: String?,
    loading: Boolean,
    onSubmit: () -> Unit,
    onBackToLogin: () -> Unit,
) {
    AuthKicker(text = "Reset access")
    Spacer(modifier = Modifier.height(12.dp))

    AuthHeadline(lead = "Forgot", accent = "your password?")
    Spacer(modifier = Modifier.height(12.dp))

    Text(
        text = "We'll send a reset link to your inbox.",
        style = Typography.bodyLarge,
        color = InkSoft,
    )
    Spacer(modifier = Modifier.height(32.dp))

    AuthField(
        label = "Email",
        value = email,
        onValueChange = onEmailChange,
        placeholder = "you@example.com",
        keyboardType = KeyboardType.Email,
        imeAction = ImeAction.Done,
        enabled = !loading,
        error = emailError,
    )

    if (submitError != null) {
        Spacer(modifier = Modifier.height(16.dp))
        AuthSubmitError(submitError)
    }
    Spacer(modifier = Modifier.height(24.dp))

    PrimaryCta(
        text = "Send reset link →",
        onClick = onSubmit,
        loading = loading,
        modifier = Modifier.fillMaxWidth(),
    )
    Spacer(modifier = Modifier.height(24.dp))

    BackToSignIn(onClick = onBackToLogin, enabled = !loading)
}

@Composable
private fun ForgotSentView(
    submittedEmail: String,
    onTryAnother: () -> Unit,
    onHaveCode: () -> Unit,
    onBackToLogin: () -> Unit,
) {
    AuthKicker(text = "Reset access", suffix = "Sent")
    Spacer(modifier = Modifier.height(12.dp))

    AuthHeadline(lead = "Check your", accent = "inbox.")
    Spacer(modifier = Modifier.height(12.dp))

    // Deliberately phrased as "if that address is registered" — the backend is
    // anti-enumeration silent and returns the same 200 for unknown emails, so
    // confirming delivery outright would be a lie (and an enumeration oracle).
    Text(
        text = "If that address is registered, a reset link is on its way to:",
        style = Typography.bodyLarge,
        color = InkSoft,
    )
    Spacer(modifier = Modifier.height(16.dp))

    QpCard(modifier = Modifier.fillMaxWidth()) {
        Kicker(text = "Sent to", color = Slate)
        Spacer(modifier = Modifier.height(6.dp))
        Text(text = submittedEmail, style = Typography.bodyMedium, color = Ink)
    }
    Spacer(modifier = Modifier.height(16.dp))

    Text(
        text = "Didn't arrive? Check spam, or try a different address. " +
            "The link expires in 15 minutes.",
        style = Typography.bodyMedium,
        color = Slate,
    )
    Spacer(modifier = Modifier.height(24.dp))

    // The emailed link opens the website, not this app — so the mobile path is
    // to copy the code out of it and paste it on the next screen.
    PrimaryCta(
        text = "I have my code →",
        onClick = onHaveCode,
        modifier = Modifier.fillMaxWidth(),
    )
    Spacer(modifier = Modifier.height(12.dp))

    GhostCta(
        text = "Try another email",
        onClick = onTryAnother,
        modifier = Modifier.fillMaxWidth(),
    )
    Spacer(modifier = Modifier.height(24.dp))

    BackToSignIn(onClick = onBackToLogin)
}

@Preview(showBackground = true)
@Composable
private fun ForgotFormPreview() {
    AuthScreenScaffold {
        ForgotFormView(
            email = "runner@example.com",
            onEmailChange = {},
            emailError = null,
            submitError = null,
            loading = false,
            onSubmit = {},
            onBackToLogin = {},
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun ForgotFormErrorPreview() {
    AuthScreenScaffold {
        ForgotFormView(
            email = "not-an-email",
            onEmailChange = {},
            emailError = "Use a valid email address.",
            submitError = "Couldn't reach QuickPitik — check your connection.",
            loading = false,
            onSubmit = {},
            onBackToLogin = {},
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun ForgotSentPreview() {
    AuthScreenScaffold {
        ForgotSentView(
            submittedEmail = "runner@example.com",
            onTryAnother = {},
            onHaveCode = {},
            onBackToLogin = {},
        )
    }
}
