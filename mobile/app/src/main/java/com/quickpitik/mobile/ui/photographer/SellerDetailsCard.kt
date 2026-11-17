package com.quickpitik.mobile.ui.photographer

import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.quickpitik.mobile.ui.common.WEBSITE_ORIGIN
import com.quickpitik.mobile.ui.common.MANILA
import com.quickpitik.mobile.ui.common.openInCustomTab
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.Typography
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeParseException
import java.util.Locale

/** What the seller-details card shows; null means no card (verified, unknown, or the read failed). */
sealed class SellerDetailsCard {
    data object Missing : SellerDetailsCard()
    data class Rejected(val reason: String?) : SellerDetailsCard()
    data class ManualReview(val paused: Boolean, val dueDate: String?) : SellerDetailsCard()
}

internal fun sellerDetailsCard(
    reviewState: String?,
    reason: String?,
    uploadsAllowed: Boolean? = null,
    dueDate: String? = null,
    today: LocalDate = LocalDate.now(MANILA),
): SellerDetailsCard? = when (reviewState) {
    "missing" -> SellerDetailsCard.Missing
    // The card adds its own ". " after the reason, so an admin's trailing period would double it.
    "rejected" -> SellerDetailsCard.Rejected(reason?.trim()?.trimEnd('.')?.trim()?.ifEmpty { null })
    "manual_review" -> SellerDetailsCard.ManualReview(
        paused = uploadsAllowed == false,
        dueDate = if (uploadsAllowed == false) null else try {
            dueDate?.takeIf { !LocalDate.parse(it).isBefore(today) }
        } catch (_: DateTimeParseException) { null },
    )
    else -> null
}

/** R-18: the gate date as "Nov 17, 2026"; text that is not an ISO date is shown as sent. */
internal fun sellerDueDateLabel(iso: String): String = try {
    LocalDate.parse(iso).format(DateTimeFormatter.ofPattern("MMM d, yyyy", Locale.US))
} catch (_: DateTimeParseException) { iso }

// SC-3: the form lives on the website (plan § 3.9), so the card only explains and links there.
@Composable
fun SellerDetailsCardView(card: SellerDetailsCard, onReturn: () -> Unit, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    // Back from the website (Custom Tab): read the status again, like FloatingCart's payment check.
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        var wasAway = false
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_STOP -> wasAway = true
                Lifecycle.Event.ON_RESUME -> if (wasAway) { wasAway = false; onReturn() }
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }
    QpCard(modifier = modifier.fillMaxWidth()) {
        Kicker("Seller details", color = if (card is SellerDetailsCard.Rejected) ErrorRed else Ink)
        Text(
            when (card) {
                SellerDetailsCard.Missing ->
                    "Before you can upload, add your ID, address and phone number in your photographer settings on quickpitik.com."
                is SellerDetailsCard.Rejected ->
                    "Your seller details were not accepted: ${card.reason ?: "no reason given"}. Send them again on quickpitik.com."
                is SellerDetailsCard.ManualReview -> when {
                    card.paused -> "An admin is checking your seller details. Uploads resume when they approve."
                    card.dueDate != null -> "Before ${sellerDueDateLabel(card.dueDate)}, take the live selfie check on quickpitik.com to keep uploading."
                    else -> "An admin is checking your seller details. You can upload meanwhile."
                }
            },
            color = Ink, style = Typography.bodyMedium,
        )
        GhostCta(
            "Open settings on quickpitik.com",
            onClick = { openInCustomTab(context, "$WEBSITE_ORIGIN/dashboard/settings#seller-details") },
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun SellerDetailsMissingPreview() { SellerDetailsCardView(SellerDetailsCard.Missing, onReturn = {}) }

@Preview(showBackground = true)
@Composable
private fun SellerDetailsRejectedPreview() { SellerDetailsCardView(SellerDetailsCard.Rejected("The ID photo is blurry."), onReturn = {}) }

@Preview(showBackground = true)
@Composable
private fun SellerDetailsManualReviewPreview() { SellerDetailsCardView(SellerDetailsCard.ManualReview(true, null), onReturn = {}) }
