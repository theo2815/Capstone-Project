package com.quickpitik.mobile.ui.runner

import android.content.ActivityNotFoundException
import android.net.Uri
import androidx.browser.customtabs.CustomTabColorSchemeParams
import androidx.browser.customtabs.CustomTabsIntent
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.quickpitik.mobile.data.remote.OrderResponse
import com.quickpitik.mobile.ui.auth.validateEmail
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.FieldShape
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.SurfaceWhite
import com.quickpitik.mobile.ui.theme.Typography
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull

internal fun isSafeCheckoutUrl(url: String): Boolean = url.toHttpUrlOrNull()?.isHttps == true

// Mobile mirror of website `CheckoutModal`. Opens from the CartSheet's
// "Continue to checkout" CTA. Once the BE returns the PayMongo redirect URL
// the sheet flips to a blocking redirect overlay and auto-launches the
// gateway — no in-sheet buttons to tap. If the user comes back without
// completing PayMongo (no deep link fires), Activity.ON_RESUME routes them
// to /orders so the pending order is visible instead of leaving them stuck
// on the overlay.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CheckoutSheet(
    cartViewModel: CartViewModel,
    onDismiss: () -> Unit,
    onCheckoutSuccess: (OrderResponse?) -> Unit,
) {
    val cartItems by cartViewModel.cartItems.collectAsState()
    val total by cartViewModel.cartTotal.collectAsState()
    val couponPreview by cartViewModel.couponPreview.collectAsState()
    val couponError by cartViewModel.couponError.collectAsState()
    val couponLoading by cartViewModel.couponLoading.collectAsState()
    val discount = couponPreview?.discountTotal ?: 0.0
    val net = (total - discount).coerceAtLeast(0.0)
    val ownedPhotoIds by cartViewModel.ownedPhotoIds.collectAsState()
    val ownedItems = cartItems.filter { ownedPhotoIds.contains(it.photoId) }
    val checkoutState by cartViewModel.checkoutState.collectAsState()
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    var emailInput by remember { mutableStateOf(cartViewModel.getLoggedInUserEmail()) }
    var selectedPaymentMethod by remember { mutableStateOf("GCASH") }
    var couponInput by remember { mutableStateOf("") }
    // Same gate as the website's checkout identify step (EMAIL_REGEX): the
    // receipt + download links go to this address, so a typo is a lost order.
    val emailError = validateEmail(emailInput)

    val scrollState = rememberScrollState()
    val context = LocalContext.current
    val uriHandler = LocalUriHandler.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val isLoading = checkoutState is CheckoutState.Loading
    val redirectingSuccess =
        (checkoutState as? CheckoutState.Success)?.order?.redirectUrl != null

    // Load the auto-applied photographer coupons for the current cart so the
    // total shown is what the runner is actually charged, not the list price.
    LaunchedEffect(Unit) { cartViewModel.previewCoupon(null) }

    // Refresh owned-photo data on open so already-purchased flags are current.
    LaunchedEffect(Unit) { cartViewModel.fetchOrders() }

    // Settled with no PayMongo handoff — a ₱0/100%-coupon giveaway or an
    // already-paid replay returns FULFILLED/PAID with no redirect. Route to the
    // receipt (like the PayMongo deep-link return) so the cart clears and the
    // downloads show. Keyed on status, NOT redirectUrl (a legacy hosted-checkout
    // field): a PENDING order must never be misread as settled.
    LaunchedEffect(checkoutState) {
        val order = (checkoutState as? CheckoutState.Success)?.order ?: return@LaunchedEffect
        if (order.status.uppercase() in setOf("PAID", "FULFILLED")) {
            onCheckoutSuccess(order)
            cartViewModel.resetCheckoutState()
        }
    }

    // Auto-launch PayMongo the moment the BE returns the redirect URL — the
    // user shouldn't have to tap anything. Custom Tabs rather than a plain
    // ACTION_VIEW (which is what LocalUriHandler fires internally): a bare
    // ACTION_VIEW can raise an app-chooser in the middle of paying, and the
    // chosen app may not honour the quickpitik:// return deep link. A Custom
    // Tab hands off to the user's default browser directly, keeps our theme,
    // and returns to this Activity — so the ON_RESUME bail-back below still
    // fires. Falls back to the generic handler on devices with no CCT
    // provider; both paths are wrapped because either can throw when the
    // device has no browser at all.
    LaunchedEffect(redirectingSuccess) {
        if (!redirectingSuccess) return@LaunchedEffect
        val url = (checkoutState as? CheckoutState.Success)?.order?.redirectUrl
            ?: return@LaunchedEffect
        if (!isSafeCheckoutUrl(url)) {
            cartViewModel.setCheckoutError("Checkout returned an unsafe payment link. Please try again.")
            return@LaunchedEffect
        }
        try {
            try {
                CustomTabsIntent.Builder()
                    .setShowTitle(true)
                    .setDefaultColorSchemeParams(
                        CustomTabColorSchemeParams.Builder()
                            .setToolbarColor(Bone.toArgb())
                            .build()
                    )
                    .build()
                    .launchUrl(context, Uri.parse(url))
            } catch (e: ActivityNotFoundException) {
                uriHandler.openUri(url)
            }
            cartViewModel.markPaymongoLaunched()
        } catch (e: Exception) {
            cartViewModel.setCheckoutError(
                "Couldn't open PayMongo. Make sure a browser is installed and try again."
            )
        }
    }

    // When the user comes back to the app after we handed off to PayMongo, if
    // a deep link fired they're already on OrderReturnScreen (this composable
    // is unmounted). If no deep link fired they bailed before paying — close
    // the sheet and drop them on /orders so the pending order is visible.
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME && cartViewModel.paymongoLaunched.value) {
                cartViewModel.clearPaymongoLaunched()
                cartViewModel.closeCheckoutSheet()
                cartViewModel.resetCheckoutState()
                // Reconcile the local cart with the server. The BE empties the
                // cart at order creation, so this returns an empty cart and the
                // floating pill clears — covering the case where no deep link
                // fired (manual return), which the OrderReturnScreen poll path
                // would otherwise have been the only thing to clear.
                cartViewModel.fetchCart()
                onCheckoutSuccess(null)
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    ModalBottomSheet(
        // Block all dismissal paths during the PayMongo handoff so the user
        // can't tap-outside / drag-down themselves out of the redirect overlay.
        onDismissRequest = {
            if (!redirectingSuccess) onDismiss()
        },
        sheetState = sheetState,
        containerColor = Bone,
        contentColor = Ink,
    ) {
        if (redirectingSuccess) {
            RedirectingOverlay()
            return@ModalBottomSheet
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .imePadding()
                .navigationBarsPadding(),
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 16.dp),
                verticalAlignment = Alignment.Top,
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Kicker("Secure checkout")
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "Confirm and pay.",
                        style = Typography.headlineSmall,
                        color = Ink,
                    )
                }
                if (isLoading) {
                    TextButton(onClick = {
                        cartViewModel.resetCheckoutState()
                        onDismiss()
                    }) {
                        Text(
                            text = "Cancel",
                            style = Typography.labelMedium,
                            color = Slate,
                        )
                    }
                }
            }
            Divider(color = Line)

            Column(
                modifier = Modifier
                    .weight(1f, fill = false)
                    .heightIn(max = 600.dp)
                    .verticalScroll(scrollState)
                    .padding(horizontal = 24.dp, vertical = 20.dp),
            ) {
                Kicker("Recipient email")
                Spacer(modifier = Modifier.height(8.dp))
                TextField(
                    value = emailInput,
                    onValueChange = { emailInput = it },
                    placeholder = { Text("Email for receipt + downloads", color = SlateSoft) },
                    singleLine = true,
                    enabled = !isLoading,
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = BoneDeep,
                        unfocusedContainerColor = BoneDeep,
                        focusedIndicatorColor = Fresh,
                        unfocusedIndicatorColor = Color.Transparent,
                        focusedTextColor = Ink,
                        unfocusedTextColor = InkSoft,
                    ),
                    shape = FieldShape,
                    modifier = Modifier.fillMaxWidth(),
                )
                if (emailInput.isNotBlank() && emailError != null) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = emailError,
                        style = Typography.bodySmall,
                        color = ErrorRed,
                    )
                }

                Spacer(modifier = Modifier.height(20.dp))
                Kicker("Payment provider")
                Spacer(modifier = Modifier.height(8.dp))

                PaymentOptionCard(
                    title = "GCash e-wallet",
                    subtitle = "Pay via GCash app redirection",
                    selected = selectedPaymentMethod == "GCASH",
                    onSelect = { selectedPaymentMethod = "GCASH" },
                )
                Spacer(modifier = Modifier.height(10.dp))
                PaymentOptionCard(
                    title = "Maya e-wallet",
                    subtitle = "Pay via Maya app redirection",
                    selected = selectedPaymentMethod == "MAYA",
                    onSelect = { selectedPaymentMethod = "MAYA" },
                )
                Spacer(modifier = Modifier.height(10.dp))
                // No card fields here, deliberately: PayMongo's hosted checkout
                // collects the card. Mobile once rendered number/MM-YY/CVV
                // fields that were captured and never transmitted — dead UI
                // that implied we take PAN data. The website never collects
                // card details either.
                PaymentOptionCard(
                    title = "Credit / debit card",
                    subtitle = "Visa, Mastercard, JCB — entered securely on PayMongo",
                    selected = selectedPaymentMethod == "CARD",
                    onSelect = { selectedPaymentMethod = "CARD" },
                )

                if (checkoutState is CheckoutState.Error) {
                    Spacer(modifier = Modifier.height(16.dp))
                    Surface(
                        color = ErrorRed.copy(alpha = 0.1f),
                        border = BorderStroke(1.dp, ErrorRed),
                        shape = FieldShape,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            text = (checkoutState as CheckoutState.Error).message,
                            color = ErrorRed,
                            style = Typography.bodyMedium,
                            modifier = Modifier.padding(16.dp),
                        )
                    }
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Coupon code (V47/V50). Auto photographer coupons are already
                // priced into the total below; this field is the optional typed
                // override. Deliberately not a Fresh element — the pay CTA is the
                // one accent per the Mobile Design skill.
                Kicker("Coupon code")
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TextField(
                        value = couponInput,
                        onValueChange = { couponInput = it.uppercase() },
                        placeholder = { Text("Optional", color = SlateSoft) },
                        singleLine = true,
                        enabled = !isLoading,
                        colors = TextFieldDefaults.colors(
                            focusedContainerColor = BoneDeep,
                            unfocusedContainerColor = BoneDeep,
                            focusedIndicatorColor = Fresh,
                            unfocusedIndicatorColor = Color.Transparent,
                            focusedTextColor = Ink,
                            unfocusedTextColor = InkSoft,
                        ),
                        shape = FieldShape,
                        modifier = Modifier.weight(1f),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    TextButton(
                        onClick = { cartViewModel.previewCoupon(couponInput) },
                        enabled = !couponLoading && couponInput.isNotBlank() && !isLoading,
                    ) {
                        Text(
                            text = if (couponLoading) "…" else "Apply",
                            style = Typography.labelLarge,
                            color = Ink,
                        )
                    }
                }
                couponError?.let { err ->
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = err,
                        style = Typography.bodySmall,
                        color = ErrorRed,
                    )
                }

                Spacer(modifier = Modifier.height(20.dp))

                if (ownedItems.isNotEmpty()) {
                    Surface(
                        color = BoneDeep,
                        border = BorderStroke(1.dp, Line),
                        shape = FieldShape,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Column(modifier = Modifier.padding(16.dp)) {
                            Text(
                                text = "You already own ${ownedItems.size} of these ${
                                    if (ownedItems.size == 1) "photo" else "photos"
                                }.",
                                style = Typography.titleSmall,
                                color = Ink,
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Re-download them free from Orders. Remove them here to check out the rest.",
                                style = Typography.bodySmall,
                                color = SlateSoft,
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            TextButton(
                                onClick = {
                                    ownedItems.forEach { cartViewModel.removeFromCart(it.photoId) }
                                },
                                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 2.dp),
                            ) {
                                Text(
                                    text = "✕ Remove already-purchased" +
                                        if (ownedItems.size > 1) " (${ownedItems.size})" else "",
                                    style = Typography.labelMedium,
                                    color = Ink,
                                )
                            }
                        }
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                }

                QpCard {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        Text("Photos", style = Typography.bodyMedium, color = Slate)
                        Text(
                            text = "${cartItems.size}",
                            style = NumeralStyle.copy(fontSize = 14.sp),
                            color = Ink,
                        )
                    }
                    if (discount > 0.0) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                        ) {
                            Text(
                                text = couponPreview?.code?.let { "Coupon $it" } ?: "Coupon discount",
                                style = Typography.bodyMedium,
                                color = Slate,
                            )
                            Text(
                                text = "−₱${"%,.2f".format(discount)}",
                                style = NumeralStyle.copy(fontSize = 14.sp),
                                color = Slate,
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Bottom,
                    ) {
                        Kicker("Total")
                        Text(
                            text = "₱${"%,.2f".format(net)}",
                            style = NumeralStyle.copy(fontSize = 28.sp),
                            color = Ink,
                        )
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    PrimaryCta(
                        text = "Authorize payment & pay",
                        onClick = { cartViewModel.checkout(emailInput, selectedPaymentMethod) },
                        modifier = Modifier.fillMaxWidth(),
                        loading = isLoading,
                        enabled = emailError == null && ownedItems.isEmpty(),
                    )
                }
            }
        }
    }
}

// Full-bleed blocking surface while the OS hands the PayMongo URL to a
// browser. Sized large enough to fill the sheet without flicker between
// "redirect URL arrived" and "browser actually opens" (Custom Tabs/intent
// resolution can take a beat). No CTAs — user can't escape into a stale
// half-checkout. They come back via deep link (→ OrderReturnScreen) or via
// the lifecycle observer in CheckoutSheet (→ /orders fallback).
@Composable
private fun RedirectingOverlay() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 420.dp)
            .background(Bone)
            .navigationBarsPadding()
            .padding(horizontal = 32.dp, vertical = 48.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        CircularProgressIndicator(
            color = Fresh,
            strokeWidth = 2.dp,
            modifier = Modifier.size(48.dp),
        )
        Spacer(modifier = Modifier.height(28.dp))
        Kicker("Hold on")
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Redirecting to PayMongo…",
            style = Typography.titleLarge,
            color = Ink,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = "Don't close the app — your browser is about to open. We'll bring you back automatically after you pay.",
            style = Typography.bodyMedium,
            color = SlateSoft,
            textAlign = TextAlign.Center,
        )
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PaymentOptionCard(
    title: String,
    subtitle: String,
    selected: Boolean,
    onSelect: () -> Unit,
) {
    Card(
        onClick = onSelect,
        border = BorderStroke(1.5.dp, if (selected) Ink else Line),
        colors = CardDefaults.cardColors(containerColor = if (selected) BoneDeep else SurfaceWhite),
        shape = QpCardShape,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(
                    selected = selected,
                    onClick = onSelect,
                    colors = RadioButtonDefaults.colors(selectedColor = Fresh),
                )
                Spacer(modifier = Modifier.width(8.dp))
                Column {
                    Text(title, style = Typography.titleSmall, color = Ink)
                    Text(subtitle, style = Typography.bodySmall, color = SlateSoft)
                }
            }
        }
    }
}
