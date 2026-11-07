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
import com.quickpitik.mobile.ui.common.openInCustomTab
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.Typography

/** What the seller-details card shows; null means no card (sent, approved, or the read failed). */
sealed class SellerDetailsCard {
    data object Missing : SellerDetailsCard()
    data class Rejected(val reason: String?) : SellerDetailsCard()
}

internal fun sellerDetailsCard(status: String?, reason: String?): SellerDetailsCard? = when (status) {
    "missing" -> SellerDetailsCard.Missing
    "rejected" -> SellerDetailsCard.Rejected(reason)
    else -> null
}

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
