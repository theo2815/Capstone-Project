package com.quickpitik.mobile.ui.common

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.browser.customtabs.CustomTabColorSchemeParams
import androidx.browser.customtabs.CustomTabsIntent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.selection.toggleable
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.LinkAnnotation
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLinkStyles
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withLink
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.quickpitik.mobile.data.local.FaceConsentPrefs
import com.quickpitik.mobile.data.local.faceNoticeNeeded
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.Typography

// The legal text lives only on the website; the app opens it (plan D1).
const val WEBSITE_ORIGIN = "https://quickpitik.com"
const val PRIVACY_URL = "$WEBSITE_ORIGIN/privacy"
const val TERMS_URL = "$WEBSITE_ORIGIN/terms"
const val SUPPORT_EMAIL = "quickpitiksupport@gmail.com"

const val DELETION_NOTICE =
    "Need to permanently delete your account? Email $SUPPORT_EMAIL and our team will process your request within 7 business days."

// Sign-up copy, same words as the website's LEGAL_COPY (PT-0a deck).
const val SIGNUP_TICK_ERROR = "Please tick the box to agree before you continue."
const val FACE_NOTICE =
    "To find your photos, we send this selfie to Amazon Rekognition (run by AWS in Singapore) to compare the face in it with faces in this event's photos. " +
        "If you only search, we don't keep the selfie; selfies you save to your profile stay until you delete them. "
const val FACE_TICK_LABEL = "I agree to face matching with this selfie. If it shows my child, I am their parent or guardian."

/** Above the pay button, for everyone (website checkoutLine). */
@Composable
internal fun CheckoutTermsLine(modifier: Modifier = Modifier) {
    Text(
        oneLink("By paying, you agree to the ", "Terms of Service (including refunds)", "$TERMS_URL#refunds", "."),
        style = Typography.bodySmall,
        color = Slate,
        modifier = modifier,
    )
}

/**
 * Privacy plan D2: shows the face-matching notice until it is ticked once on
 * this device. Returns true once ticked; selfie actions stay disabled until then.
 */
@Composable
internal fun faceNoticeAccepted(): Boolean {
    val context = LocalContext.current
    // shortcut: each surface reads the tick when it opens, so a notice already on screen elsewhere stays until redrawn; share one state if that confuses testers.
    var needed by remember { mutableStateOf(faceNoticeNeeded(FaceConsentPrefs(context).acceptedVersion)) }
    if (needed) {
        QpCard(modifier = Modifier.fillMaxWidth()) {
            Text(
                oneLink(FACE_NOTICE, "How face matching works", "$PRIVACY_URL#face-matching", ""),
                style = Typography.bodySmall,
                color = InkSoft,
            )
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .toggleable(value = false, role = Role.Checkbox) {
                        FaceConsentPrefs(context).accept()
                        needed = false
                    },
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Checkbox(checked = false, onCheckedChange = null, colors = CheckboxDefaults.colors(checkedColor = Ink, checkmarkColor = Bone))
                Text(FACE_TICK_LABEL, style = Typography.bodySmall, color = Ink)
            }
        }
        Spacer(Modifier.height(12.dp))
    }
    return !needed
}

@Composable
private fun oneLink(before: String, label: String, url: String, after: String): AnnotatedString {
    val context = LocalContext.current
    val style = TextLinkStyles(SpanStyle(color = Ink, textDecoration = TextDecoration.Underline))
    return buildAnnotatedString {
        append(before)
        withLink(LinkAnnotation.Clickable(label, style) { openInCustomTab(context, url) }) { append(label) }
        append(after)
    }
}

// "<before>Terms of Service<middle>Privacy Policy<after>", both links open in a Custom Tab.
@Composable
private fun legalLinks(before: String, middle: String, after: String): AnnotatedString {
    val context = LocalContext.current
    val style = TextLinkStyles(SpanStyle(color = Ink, textDecoration = TextDecoration.Underline))
    return buildAnnotatedString {
        append(before)
        withLink(LinkAnnotation.Clickable("terms", style) { openInCustomTab(context, TERMS_URL) }) { append("Terms of Service") }
        append(middle)
        withLink(LinkAnnotation.Clickable("privacy", style) { openInCustomTab(context, PRIVACY_URL) }) { append("Privacy Policy") }
        append(after)
    }
}

/** The sign-up agreement tick (register and the Google role sheet). */
@Composable
internal fun SignupAgreementRow(checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 48.dp)
            .toggleable(value = checked, role = Role.Checkbox, onValueChange = onCheckedChange),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Checkbox(
            checked = checked,
            onCheckedChange = null,
            colors = CheckboxDefaults.colors(checkedColor = Ink, checkmarkColor = Bone),
        )
        Text(
            legalLinks("I am 18 or older, I agree to the ", " and I have read the ", "."),
            style = Typography.bodySmall,
            color = Slate,
        )
    }
}

/** Under the Google button on Register only; the binding tick is in the role sheet. */
@Composable
internal fun GoogleSignupLine() {
    Text(
        legalLinks("By continuing with Google, you agree to the ", " and ", ". You'll confirm this on the next step."),
        style = Typography.bodySmall,
        color = Slate,
    )
}

fun openInCustomTab(context: Context, url: String) {
    runCatching {
        try {
            CustomTabsIntent.Builder()
                .setShowTitle(true)
                .setDefaultColorSchemeParams(
                    CustomTabColorSchemeParams.Builder().setToolbarColor(Bone.toArgb()).build()
                )
                .build()
                .launchUrl(context, Uri.parse(url))
        } catch (e: ActivityNotFoundException) {
            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
        }
    }
}

// Encoded so an address like "a+b&c@x.com" reaches the body intact.
fun deletionRequestUri(accountEmail: String?): String {
    val body = "Please delete my QuickPitik account.\n\nAccount email: ${accountEmail.orEmpty()}"
    return "mailto:$SUPPORT_EMAIL?subject=${Uri.encode("Account deletion request")}&body=${Uri.encode(body)}"
}

fun requestAccountDeletion(context: Context, accountEmail: String?) {
    try {
        context.startActivity(Intent(Intent.ACTION_SENDTO, Uri.parse(deletionRequestUri(accountEmail))))
    } catch (_: ActivityNotFoundException) {
        Toast.makeText(context, "No email app found. Email $SUPPORT_EMAIL to delete your account.", Toast.LENGTH_LONG).show()
    }
}

@Composable
internal fun PolicyLinkRow(label: String, action: String, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 6.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(label, style = Typography.bodyMedium, color = Ink, fontWeight = FontWeight.Medium)
        ArrowLabel(action, color = Slate, style = Typography.labelMedium)
    }
}

@Composable
internal fun SupportEmailRow() {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 6.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text("Support email", style = Typography.bodyMedium, color = Ink, fontWeight = FontWeight.Medium)
        Text(SUPPORT_EMAIL, style = Typography.bodyMedium, color = Slate)
    }
}
