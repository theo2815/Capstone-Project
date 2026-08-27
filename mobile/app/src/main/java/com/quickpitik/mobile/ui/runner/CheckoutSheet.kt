package com.quickpitik.mobile.ui.runner

import android.content.ActivityNotFoundException
import android.net.Uri
import androidx.browser.customtabs.CustomTabColorSchemeParams
import androidx.browser.customtabs.CustomTabsIntent
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
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
import com.quickpitik.mobile.ui.auth.validateEmail
import com.quickpitik.mobile.ui.theme.*
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
    onCheckoutSuccess: () -> Unit,
) {
    val cartItems by cartViewModel.cartItems.collectAsState()
    val total by cartViewModel.cartTotal.collectAsState()
    val checkoutState by cartViewModel.checkoutState.collectAsState()
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)

    var emailInput by remember { mutableStateOf(cartViewModel.getLoggedInUserEmail()) }
    var selectedPaymentMethod by remember { mutableStateOf("GCASH") }
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

    // Empty-cart settlement (no PayMongo handoff) — fire success immediately.
    LaunchedEffect(checkoutState) {
        val state = checkoutState
        if (state is CheckoutState.Success && state.order.redirectUrl == null) {
            onCheckoutSuccess()
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
                onCheckoutSuccess()
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
                    Spacer(modifier = Modifier.height(8.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Bottom,
                    ) {
                        Kicker("Total")
                        Text(
                            text = "₱${"%,.2f".format(total)}",
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
                        enabled = emailError == null,
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
