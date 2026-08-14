package com.quickpitik.mobile.ui.auth

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Divider
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.ui.theme.*

// Shared chrome for the auth-recovery screens (ForgotPasswordScreen +
// ResetPasswordScreen), the mobile counterpart to the website's <AuthShell>.
// Extracted because both screens need identical inset/IME handling and getting
// it right once beats getting it right twice.
//
// LoginScreen/RegisterScreen predate these and keep their own inline chrome —
// migrating them is not part of this change.

/**
 * Bone page + logo lockup + safe-area and keyboard insets.
 *
 * `imePadding()` sits on the scrolling Column (not the fields) so the focused
 * input lifts above the keyboard instead of being covered by it — these are
 * form screens, and they are only ever used with the keyboard up.
 */
@Composable
fun AuthScreenScaffold(content: @Composable ColumnScope.() -> Unit) {
    Surface(modifier = Modifier.fillMaxSize(), color = Bone) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp)
                .statusBarsPadding()
                .navigationBarsPadding()
                .imePadding()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.Top,
        ) {
            Spacer(modifier = Modifier.height(40.dp))
            AuthLogoLockup()
            content()
            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

/** Ring-and-dot mark + wordmark. Mirrors the lockup in LoginScreen. */
@Composable
private fun AuthLogoLockup() {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier.padding(bottom = 24.dp),
    ) {
        Box(modifier = Modifier.size(28.dp), contentAlignment = Alignment.Center) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                drawCircle(
                    color = Ink,
                    radius = size.minDimension / 2f,
                    style = Stroke(width = 1.5.dp.toPx()),
                )
                drawCircle(color = Fresh, radius = size.minDimension / 5.6f)
            }
        }
        Text(
            text = "QuickPitik",
            style = Typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = Ink,
            fontSize = 18.sp,
        )
    }
}

/**
 * Two-line display headline with the second line in Fresh — the website's
 * `<h1>Forgot<br/><span className="text-fresh">your password?</span></h1>`.
 *
 * This is the one place Fresh appears besides the PrimaryCta. Both the website
 * auth pages and the existing LoginScreen/RegisterScreen do the same: the
 * accent reads as typography, not as a second competing button.
 */
@Composable
fun AuthHeadline(lead: String, accent: String) {
    Text(
        text = buildAnnotatedString {
            append("$lead\n")
            withStyle(style = SpanStyle(color = Fresh)) { append(accent) }
        },
        style = Typography.displayLarge,
        color = Ink,
    )
}

/**
 * Mono kicker with an optional tone-coloured suffix — the website's
 * `Reset access · Sent` / `· Done` pattern. [suffix] is rendered inside the
 * same uppercase Kicker so the two halves share a baseline.
 */
@Composable
fun AuthKicker(text: String, suffix: String? = null, suffixColor: androidx.compose.ui.graphics.Color = Fresh) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Kicker(text = text, color = Slate)
        if (suffix != null) {
            Spacer(modifier = Modifier.width(8.dp))
            Kicker(text = "· $suffix", color = suffixColor)
        }
    }
}

/**
 * Labelled input. Same tokens and shape as the photographer settings sheet's
 * field (BoneDeep-free, Line border, [FieldShape], ErrorRed on invalid), plus
 * the SHOW/HIDE password toggle the recovery flow needs.
 *
 * [labelSuffix] renders normal-case next to the uppercase kicker — the
 * website's "New password · min. 8 characters" hint.
 */
@Composable
fun AuthField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    placeholder: String = "",
    labelSuffix: String? = null,
    keyboardType: KeyboardType = KeyboardType.Text,
    imeAction: ImeAction = ImeAction.Next,
    isPassword: Boolean = false,
    enabled: Boolean = true,
    error: String? = null,
) {
    var revealed by remember { mutableStateOf(false) }
    Column(modifier = modifier) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Kicker(text = label, color = Slate)
            if (labelSuffix != null) {
                Spacer(modifier = Modifier.width(8.dp))
                Text(text = labelSuffix, style = Typography.bodySmall, color = SlateSoft)
            }
        }
        Spacer(modifier = Modifier.height(6.dp))
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            enabled = enabled,
            placeholder = { Text(placeholder, color = SlateSoft, style = Typography.bodyMedium) },
            singleLine = true,
            visualTransformation = if (isPassword && !revealed) {
                PasswordVisualTransformation()
            } else {
                VisualTransformation.None
            },
            trailingIcon = if (isPassword) {
                {
                    // Text toggle rather than an icon — matches LoginScreen and
                    // avoids pulling in material-icons-extended.
                    Box(
                        modifier = Modifier
                            .heightIn(min = 48.dp)
                            .clickable(enabled = enabled) { revealed = !revealed }
                            .padding(horizontal = 12.dp),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = if (revealed) "HIDE" else "SHOW",
                            style = Typography.labelMedium,
                            color = Slate,
                        )
                    }
                }
            } else null,
            keyboardOptions = KeyboardOptions(keyboardType = keyboardType, imeAction = imeAction),
            isError = error != null,
            shape = FieldShape,
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Ink,
                unfocusedTextColor = Ink,
                focusedBorderColor = Ink,
                unfocusedBorderColor = Line,
                errorBorderColor = ErrorRed,
                cursorColor = Ink,
                focusedContainerColor = Bone,
                unfocusedContainerColor = Bone,
                disabledContainerColor = Bone,
            ),
            textStyle = Typography.bodyMedium,
            modifier = Modifier.fillMaxWidth(),
        )
        if (error != null) {
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = error, style = Typography.bodySmall, color = ErrorRed)
        }
    }
}

/** Submit-level error line — the website's standalone `<FieldError>` under the form. */
@Composable
fun AuthSubmitError(message: String) {
    Text(
        text = message,
        style = Typography.bodyMedium,
        color = ErrorRed,
        modifier = Modifier.fillMaxWidth(),
    )
}

/**
 * Hairline + centered "← Back to sign in". Sized to a 48dp touch target; the
 * glyph is a plain left arrow (U+2190), which the mono kicker font does carry —
 * only the RIGHT arrow is missing from Funnel Sans, which is why the CTAs go
 * through ArrowLabel/CtaContent instead.
 */
@Composable
fun BackToSignIn(onClick: () -> Unit, enabled: Boolean = true) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Divider(color = Line, thickness = 1.dp)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 48.dp)
                .clickable(enabled = enabled, onClick = onClick),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = "← BACK TO SIGN IN",
                style = Typography.labelMedium,
                color = Slate,
            )
        }
    }
}
