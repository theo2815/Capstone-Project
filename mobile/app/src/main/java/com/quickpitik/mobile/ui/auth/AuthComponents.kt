package com.quickpitik.mobile.ui.auth

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.credentials.CredentialManager
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import androidx.credentials.exceptions.GetCredentialCancellationException
import androidx.credentials.exceptions.GetCredentialException
import androidx.credentials.exceptions.NoCredentialException
import com.google.android.libraries.identity.googleid.GetSignInWithGoogleOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import com.quickpitik.mobile.BuildConfig
import com.quickpitik.mobile.R
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.*
import kotlinx.coroutines.launch

// Shared chrome for the auth-recovery flow (ForgotPasswordScreen, which since
// the OTP cutover carries every step), the mobile counterpart to the website's
// <AuthShell>. Extracted so inset/IME handling is got right once.
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
    // Anton hero (Finish Line) — uppercased here because HeroText can't carry
    // the two-tone span.
    Text(
        text = buildAnnotatedString {
            append("${lead.uppercase()}\n")
            withStyle(style = SpanStyle(color = Fresh)) { append(accent.uppercase()) }
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

/**
 * Debug-only backend switcher: a quiet collapsed row that opens a sheet.
 *
 * Exists because every physical-device protocol in `mobile/tasks.md` opens with
 * "set RetrofitClient.BASE_URL to the laptop's Wi-Fi IPv4" — which used to mean
 * a Kotlin edit, a recompile and a reinstall. During a camera shoot the phone's
 * USB-C port is occupied by the body, so there is no cable to push that build
 * over. Now it is typed once, here, and persisted.
 *
 * Renders NOTHING outside a debug build, and [RetrofitClient.setBaseUrl] is
 * itself gated, so no shipped APK carries a server field.
 *
 * No Fresh in the collapsed row on purpose: the Login viewport already spends
 * its accent three times over (logo dot, the word "back.", the LOG IN button).
 * The sheet is its own viewport and gets exactly one, on the confirm CTA.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DevServerRow() {
    if (!BuildConfig.DEBUG) return

    var sheetOpen by remember { mutableStateOf(false) }
    // Re-read on each open rather than hoisting: RetrofitClient is the owner,
    // and a stale copy here would show the wrong origin after a reset.
    var current by remember { mutableStateOf(RetrofitClient.BASE_URL) }

    Column(modifier = Modifier.fillMaxWidth()) {
        Divider(color = Line, thickness = 1.dp)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 48.dp)
                .clickable {
                    current = RetrofitClient.BASE_URL
                    sheetOpen = true
                },
            contentAlignment = Alignment.Center,
        ) {
            Kicker(text = "SERVER · ${current.toDisplayOrigin()}", color = SlateSoft)
        }
    }

    if (sheetOpen) {
        DevServerSheet(
            initialValue = current,
            onDismiss = { sheetOpen = false },
            onApplied = {
                current = RetrofitClient.BASE_URL
                sheetOpen = false
            },
        )
    }
}

/** "http://192.168.1.232:8080/" → "192.168.1.232:8080" — the half worth reading. */
private fun String.toDisplayOrigin(): String =
    removePrefix("https://").removePrefix("http://").trimEnd('/')

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun DevServerSheet(
    initialValue: String,
    onDismiss: () -> Unit,
    onApplied: () -> Unit,
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = Bone,
        dragHandle = null,
    ) {
        DevServerSheetContent(
            initialValue = initialValue,
            // Returns false when the value never parsed: nothing changed, and
            // the sheet stays open showing why.
            onApply = { typed ->
                val ok = RetrofitClient.setBaseUrl(typed)
                if (ok) onApplied()
                ok
            },
            onReset = {
                RetrofitClient.resetBaseUrl()
                onApplied()
            },
        )
    }
}

/**
 * Split out from the sheet so the states are previewable — a ModalBottomSheet
 * can't render inside @Preview.
 */
@Composable
private fun DevServerSheetContent(
    initialValue: String,
    onApply: (String) -> Boolean,
    onReset: () -> Unit,
) {
    var typed by remember { mutableStateOf(initialValue.toDisplayOrigin()) }
    var error by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .navigationBarsPadding()
            .imePadding()
            .verticalScroll(rememberScrollState()),
    ) {
        Spacer(modifier = Modifier.height(8.dp))
        AuthKicker(text = "Debug", suffix = "Backend", suffixColor = Slate)
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "Point this build at a different backend.",
            style = Typography.bodyLarge,
            color = Ink,
        )
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "Run ipconfig on the laptop and enter its Wi-Fi IPv4. " +
                "Debug builds only — this never ships.",
            style = Typography.bodySmall,
            color = SlateSoft,
        )
        Spacer(modifier = Modifier.height(20.dp))

        AuthField(
            label = "Address",
            value = typed,
            onValueChange = {
                typed = it
                error = null
            },
            placeholder = "192.168.1.232:8080",
            labelSuffix = "http:// is assumed",
            keyboardType = KeyboardType.Uri,
            imeAction = ImeAction.Done,
            error = error,
        )
        Spacer(modifier = Modifier.height(24.dp))

        PrimaryCta(
            text = "Use this server",
            onClick = {
                // setBaseUrl is the only validator — duplicating the rule here
                // would give two answers that could drift apart.
                if (!onApply(typed)) {
                    error = "That isn't a valid address. Try 192.168.1.232:8080."
                }
            },
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(12.dp))
        GhostCta(
            text = "Reset to default",
            onClick = onReset,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(modifier = Modifier.height(24.dp))
    }
}

@Preview(name = "Dev server sheet", showBackground = true, backgroundColor = 0xFFF7F6F2)
@Composable
private fun DevServerSheetPreview() {
    QuickPitikMobileTheme {
        Surface(color = Bone) {
            DevServerSheetContent(
                initialValue = "http://192.168.1.232:8080/",
                onApply = { true },
                onReset = {},
            )
        }
    }
}

@Preview(name = "Dev server row", showBackground = true, backgroundColor = 0xFFF7F6F2)
@Composable
private fun DevServerRowPreview() {
    QuickPitikMobileTheme {
        Surface(color = Bone) {
            Column(modifier = Modifier.padding(24.dp)) {
                Divider(color = Line, thickness = 1.dp)
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Kicker(text = "SERVER · 192.168.1.232:8080", color = SlateSoft)
                }
            }
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

// ---------------------------------------------------------------------------
// "Continue with Google" (2026-08-29) — Credential Manager + role sheet
// ---------------------------------------------------------------------------

/**
 * GhostCta-idiom Google button (Ink outline, no fill — Fresh stays reserved
 * for the screen's primary CTA) followed by the "or with email" divider,
 * mirroring the website's `<GoogleButton/> + <AuthDivider/>` block. Renders
 * nothing when no web client ID is compiled in (QP_GOOGLE_SERVER_CLIENT_ID in
 * gradle.properties) — website button and backend endpoint go dark off the
 * same value, so the three surfaces stay symmetric.
 *
 * The getCredential call must anchor to the ACTIVITY — which LocalContext is
 * inside MainActivity's composition; an Application context cannot show the
 * account dialog. Only the extracted ID token leaves this composable; the
 * exchange itself is AuthViewModel.googleLogin's job.
 */
@Composable
fun GoogleSignInRow(
    enabled: Boolean,
    onIdToken: (String) -> Unit,
    onError: (String) -> Unit,
) {
    if (BuildConfig.GOOGLE_SERVER_CLIENT_ID.isBlank()) return

    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    Column {
        OutlinedButton(
            onClick = {
                scope.launch {
                    try {
                        val option = GetSignInWithGoogleOption
                            .Builder(BuildConfig.GOOGLE_SERVER_CLIENT_ID)
                            .build()
                        val result = CredentialManager.create(context).getCredential(
                            context,
                            GetCredentialRequest(listOf(option)),
                        )
                        val credential = result.credential
                        if (
                            credential is CustomCredential &&
                            credential.type == GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL
                        ) {
                            onIdToken(GoogleIdTokenCredential.createFrom(credential.data).idToken)
                        } else {
                            onError("Google sign-in unavailable. Use your email instead.")
                        }
                    } catch (e: GetCredentialCancellationException) {
                        // The user closed the account picker — not an error.
                    } catch (e: NoCredentialException) {
                        onError("No Google account on this device.")
                    } catch (e: GetCredentialException) {
                        // Play-less devices/emulators land here as provider-
                        // configuration failures, as does a missing Android
                        // OAuth client in Google Cloud (see gradle.properties).
                        onError("Google sign-in unavailable. Use your email instead.")
                    }
                }
            },
            enabled = enabled,
            shape = PillShape,
            border = BorderStroke(1.dp, Ink),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
            modifier = Modifier
                .fillMaxWidth()
                .height(48.dp),
        ) {
            Image(
                painter = painterResource(R.drawable.ic_google_g),
                contentDescription = null,
                modifier = Modifier.size(18.dp),
            )
            Spacer(modifier = Modifier.width(10.dp))
            Text(text = "CONTINUE WITH GOOGLE", style = Typography.labelLarge)
        }

        Spacer(modifier = Modifier.height(20.dp))

        Row(verticalAlignment = Alignment.CenterVertically) {
            Divider(modifier = Modifier.weight(1f), color = Line)
            Text(
                text = "OR WITH EMAIL",
                style = Typography.labelMedium,
                color = SlateSoft,
                modifier = Modifier.padding(horizontal = 12.dp),
            )
            Divider(modifier = Modifier.weight(1f), color = Line)
        }

        Spacer(modifier = Modifier.height(28.dp))
    }
}

/**
 * Role pick for a brand-new Google account (AuthState.GoogleRoleRequired —
 * the backend answered 422 ROLE_REQUIRED because Google supplies no role).
 * A bottom sheet, per the mobile-design rule that contextual choices ride
 * sheets, not dialogs. Mirrors the website's /onboarding gate: the choice is
 * permanent, so it is asked explicitly, never defaulted from a toggle the
 * user may not have noticed.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GoogleRoleSheet(
    onPick: (isPhotographer: Boolean) -> Unit,
    onDismiss: () -> Unit,
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = Bone,
    ) {
        GoogleRoleSheetContent(onPick = onPick)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun GoogleRoleSheetContent(onPick: (isPhotographer: Boolean) -> Unit) {
    var isPhotographer by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .padding(horizontal = 24.dp)
            .padding(bottom = 24.dp)
            .navigationBarsPadding(),
    ) {
        Text(
            text = "ONE LAST THING",
            style = Typography.labelLarge,
            color = Slate,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "How will you use QuickPitik?",
            style = Typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = Ink,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "This choice is permanent.",
            style = Typography.bodyMedium,
            color = SlateSoft,
        )
        Spacer(modifier = Modifier.height(20.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            GoogleRoleCard(
                index = "01",
                title = "I run",
                subtitle = "Find your photos",
                selected = !isPhotographer,
                onClick = { isPhotographer = false },
                modifier = Modifier.weight(1f),
            )
            GoogleRoleCard(
                index = "02",
                title = "I shoot",
                subtitle = "Sell your photos",
                selected = isPhotographer,
                onClick = { isPhotographer = true },
                modifier = Modifier.weight(1f),
            )
        }
        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = { onPick(isPhotographer) },
            shape = PillShape,
            colors = ButtonDefaults.buttonColors(
                containerColor = Fresh,
                contentColor = Bone,
            ),
            contentPadding = PaddingValues(vertical = 16.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(text = "CONTINUE", style = Typography.labelLarge)
        }
    }
}

// The RegisterScreen role-card recipe (numbered corner, 6dp Fresh dot when
// selected, BoneDeep fill + Ink border), reused for the Google role pick.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun GoogleRoleCard(
    index: String,
    title: String,
    subtitle: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(
        onClick = onClick,
        border = BorderStroke(
            width = 1.5.dp,
            color = if (selected) Ink else Line,
        ),
        colors = CardDefaults.cardColors(
            containerColor = if (selected) BoneDeep else Bone,
        ),
        shape = RoundedCornerShape(16.dp),
        modifier = modifier,
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(index, style = Typography.labelMedium, color = SlateSoft)
                if (selected) {
                    Surface(
                        shape = PillShape,
                        color = Fresh,
                        modifier = Modifier.size(6.dp),
                    ) {}
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(title, style = Typography.titleMedium, color = Ink)
            Text(subtitle, style = Typography.bodyMedium, color = SlateSoft)
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun GoogleRoleSheetContentPreview() {
    Surface(color = Bone) {
        GoogleRoleSheetContent(onPick = {})
    }
}
