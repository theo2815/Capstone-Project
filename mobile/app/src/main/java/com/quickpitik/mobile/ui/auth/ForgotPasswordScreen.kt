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
import kotlinx.coroutines.delay

// Mobile counterpart to website /(auth)/forgot-password
// (components/auth/forgot-password-form.tsx). The whole OTP reset flow lives
// on this one screen, same steps as the web: request a 6-digit code, verify
// it, set the new password. The old ResetPasswordScreen (hand-pasted link
// token — a workaround for reset mail that could only link to the website
// origin) is gone: the mailed code is native input on every client now.
//
// The continuation token from verify lives in AuthViewModel memory only; this
// screen never sees it.

private enum class ResetStep { Email, Code, Password, Done }

// Which call is in flight, so the shared PasswordResetState.Success can be
// routed: a resend's Success restarts the cooldown, a verify's advances the
// step. Only one call is ever in flight at a time.
private enum class PendingAction { None, Request, Resend, Verify, Confirm }

private const val RESEND_COOLDOWN_SECONDS = 60

@Composable
fun ForgotPasswordScreen(
    viewModel: AuthViewModel,
    onNavigateToLogin: () -> Unit,
) {
    val resetState by viewModel.passwordResetState.collectAsState()

    var step by remember { mutableStateOf(ResetStep.Email) }
    var pending by remember { mutableStateOf(PendingAction.None) }
    var email by remember { mutableStateOf("") }
    var submittedEmail by remember { mutableStateOf("") }
    var code by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }
    var emailError by remember { mutableStateOf<String?>(null) }
    var codeError by remember { mutableStateOf<String?>(null) }
    var passwordError by remember { mutableStateOf<String?>(null) }
    var confirmError by remember { mutableStateOf<String?>(null) }
    var cooldown by remember { mutableIntStateOf(0) }

    // A stale Success from a previous visit would advance the flow on entry.
    LaunchedEffect(Unit) { viewModel.resetPasswordResetState() }

    LaunchedEffect(resetState) {
        if (resetState is PasswordResetState.Success) {
            when (pending) {
                PendingAction.Request -> {
                    step = ResetStep.Code
                    cooldown = RESEND_COOLDOWN_SECONDS
                }
                PendingAction.Resend -> {
                    code = ""
                    cooldown = RESEND_COOLDOWN_SECONDS
                }
                PendingAction.Verify -> step = ResetStep.Password
                PendingAction.Confirm -> step = ResetStep.Done
                PendingAction.None -> {}
            }
            pending = PendingAction.None
            viewModel.resetPasswordResetState()
        }
    }

    LaunchedEffect(cooldown) {
        if (cooldown > 0) {
            delay(1000)
            cooldown -= 1
        }
    }

    val loading = resetState is PasswordResetState.Loading
    val submitError = (resetState as? PasswordResetState.Error)?.message

    // Any keystroke clears a stale server error; the user is already fixing it.
    val clearSubmitError = {
        if (resetState is PasswordResetState.Error) viewModel.resetPasswordResetState()
    }

    fun startOver() {
        step = ResetStep.Email
        email = ""
        submittedEmail = ""
        code = ""
        password = ""
        confirmPassword = ""
        emailError = null
        codeError = null
        passwordError = null
        confirmError = null
        cooldown = 0
        viewModel.resetPasswordResetState()
    }

    AuthScreenScaffold {
        AnimatedContent(
            targetState = step,
            transitionSpec = { fadeIn(tween(200)) togetherWith fadeOut(tween(200)) },
            label = "forgot-password-step",
        ) { current ->
            Column {
                when (current) {
                    ResetStep.Email -> EmailStepView(
                        email = email,
                        onEmailChange = {
                            email = it
                            if (emailError != null) emailError = null
                            clearSubmitError()
                        },
                        emailError = emailError,
                        submitError = submitError,
                        loading = loading,
                        onSubmit = {
                            val fieldError = validateEmail(email)
                            emailError = fieldError
                            if (fieldError == null) {
                                submittedEmail = email.trim()
                                pending = PendingAction.Request
                                viewModel.requestPasswordReset(email)
                            }
                        },
                        onBackToLogin = onNavigateToLogin,
                    )

                    ResetStep.Code -> CodeStepView(
                        submittedEmail = submittedEmail,
                        code = code,
                        onCodeChange = {
                            code = it.filter(Char::isDigit).take(6)
                            if (codeError != null) codeError = null
                            clearSubmitError()
                        },
                        codeError = codeError,
                        submitError = submitError,
                        loading = loading,
                        cooldown = cooldown,
                        onSubmit = {
                            val fieldError = validateResetCode(code)
                            codeError = fieldError
                            if (fieldError == null) {
                                pending = PendingAction.Verify
                                viewModel.verifyResetOtp(submittedEmail, code)
                            }
                        },
                        onResend = {
                            codeError = null
                            pending = PendingAction.Resend
                            viewModel.requestPasswordReset(submittedEmail)
                        },
                        onTryAnother = ::startOver,
                    )

                    ResetStep.Password -> PasswordStepView(
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
                        submitError = submitError,
                        loading = loading,
                        onSubmit = {
                            val passErr = validateNewPassword(password)
                            val confirmErr = when {
                                confirmPassword.isEmpty() -> "Please confirm your new password."
                                password != confirmPassword -> "Passwords don't match."
                                else -> null
                            }
                            passwordError = passErr
                            confirmError = confirmErr
                            if (passErr == null && confirmErr == null) {
                                pending = PendingAction.Confirm
                                viewModel.confirmPasswordReset(password)
                            }
                        },
                        onStartOver = ::startOver,
                    )

                    ResetStep.Done -> DoneStepView(onSignIn = onNavigateToLogin)
                }
            }
        }
    }
}

@Composable
private fun EmailStepView(
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
        text = "We'll email you a 6-digit code.",
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
        text = "Send code →",
        onClick = onSubmit,
        loading = loading,
        modifier = Modifier.fillMaxWidth(),
    )
    Spacer(modifier = Modifier.height(24.dp))

    BackToSignIn(onClick = onBackToLogin, enabled = !loading)
}

@Composable
private fun CodeStepView(
    submittedEmail: String,
    code: String,
    onCodeChange: (String) -> Unit,
    codeError: String?,
    submitError: String?,
    loading: Boolean,
    cooldown: Int,
    onSubmit: () -> Unit,
    onResend: () -> Unit,
    onTryAnother: () -> Unit,
) {
    AuthKicker(text = "Reset access", suffix = "Sent")
    Spacer(modifier = Modifier.height(12.dp))

    AuthHeadline(lead = "Enter your", accent = "code.")
    Spacer(modifier = Modifier.height(12.dp))

    // Deliberately phrased as "if that address is registered" — the backend is
    // anti-enumeration silent and returns the same 200 for unknown emails, so
    // confirming delivery outright would be a lie (and an enumeration oracle).
    Text(
        text = "If that address is registered, a 6-digit code is on its way. " +
            "It expires in 10 minutes.",
        style = Typography.bodyLarge,
        color = InkSoft,
    )
    Spacer(modifier = Modifier.height(16.dp))

    QpCard(modifier = Modifier.fillMaxWidth()) {
        Kicker(text = "Sent to", color = Slate)
        Spacer(modifier = Modifier.height(6.dp))
        Text(text = submittedEmail, style = Typography.bodyMedium, color = Ink)
    }
    Spacer(modifier = Modifier.height(24.dp))

    AuthField(
        label = "Code",
        value = code,
        onValueChange = onCodeChange,
        placeholder = "000000",
        labelSuffix = "from your email",
        keyboardType = KeyboardType.Number,
        imeAction = ImeAction.Done,
        enabled = !loading,
        error = codeError,
    )

    if (submitError != null) {
        Spacer(modifier = Modifier.height(16.dp))
        AuthSubmitError(submitError)
    }
    Spacer(modifier = Modifier.height(24.dp))

    PrimaryCta(
        text = "Verify code →",
        onClick = onSubmit,
        loading = loading,
        modifier = Modifier.fillMaxWidth(),
    )
    Spacer(modifier = Modifier.height(12.dp))

    GhostCta(
        text = if (cooldown > 0) "Resend code in ${cooldown}s" else "Resend code",
        onClick = onResend,
        enabled = cooldown == 0 && !loading,
        modifier = Modifier.fillMaxWidth(),
    )
    Spacer(modifier = Modifier.height(12.dp))

    GhostCta(
        text = "Try another email",
        onClick = onTryAnother,
        enabled = !loading,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun PasswordStepView(
    password: String,
    onPasswordChange: (String) -> Unit,
    passwordError: String?,
    confirmPassword: String,
    onConfirmChange: (String) -> Unit,
    confirmError: String?,
    submitError: String?,
    loading: Boolean,
    onSubmit: () -> Unit,
    onStartOver: () -> Unit,
) {
    AuthKicker(text = "Reset access", suffix = "Verified")
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
    Spacer(modifier = Modifier.height(12.dp))

    // The continuation token is 15-minute one-shot; when it dies the only way
    // forward is a fresh code.
    GhostCta(
        text = "Start over",
        onClick = onStartOver,
        enabled = !loading,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun DoneStepView(onSignIn: () -> Unit) {
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
private fun EmailStepPreview() {
    AuthScreenScaffold {
        EmailStepView(
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
private fun CodeStepErrorPreview() {
    AuthScreenScaffold {
        CodeStepView(
            submittedEmail = "runner@example.com",
            code = "123456",
            onCodeChange = {},
            codeError = null,
            submitError = "That code is invalid or has expired",
            loading = false,
            cooldown = 42,
            onSubmit = {},
            onResend = {},
            onTryAnother = {},
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun DoneStepPreview() {
    AuthScreenScaffold {
        DoneStepView(onSignIn = {})
    }
}
