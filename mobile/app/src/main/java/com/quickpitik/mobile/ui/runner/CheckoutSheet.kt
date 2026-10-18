package com.quickpitik.mobile.ui.runner

import android.content.ActivityNotFoundException
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.activity.compose.BackHandler
import androidx.browser.customtabs.CustomTabColorSchemeParams
import androidx.browser.customtabs.CustomTabsIntent
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.SheetValue
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.data.download.PhotoDownloader
import com.quickpitik.mobile.ui.auth.validateEmail
import com.quickpitik.mobile.ui.common.formatManila
import com.quickpitik.mobile.ui.common.formatOrderRef
import com.quickpitik.mobile.ui.common.peso
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.FieldShape
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
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
import com.quickpitik.mobile.ui.theme.WarningOrange
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import java.util.Base64

internal fun isSafeCheckoutUrl(url: String): Boolean = url.toHttpUrlOrNull()?.isHttps == true

/** PayMongo's QR (a base64 PNG data URI) as image bytes; null if it isn't one. */
internal fun qrImageBytes(dataUri: String): ByteArray? {
    // A QR PNG is a few KB; anything this big is not one (and would stall the decode).
    if (dataUri.length > 512 * 1024) return null
    val comma = dataUri.indexOf(',')
    if (comma < 0) return null
    val header = dataUri.substring(0, comma)
    if (!header.equals("data:image/png;base64", ignoreCase = true)) return null
    return runCatching { Base64.getMimeDecoder().decode(dataUri.substring(comma + 1)) }.getOrNull()
}

internal enum class QrWaitMode { AWAITING, CONFIRMING, SLOW }

// After "I've paid", the copy turns to "taking longer than usual" after a
// minute — banks usually report back within 10–30 s (website SLOW_AFTER_MS).
internal fun qrWaitMode(paidClaimedAt: Long?, now: Long): QrWaitMode = when {
    paidClaimedAt == null -> QrWaitMode.AWAITING
    now - paidClaimedAt < 60_000 -> QrWaitMode.CONFIRMING
    else -> QrWaitMode.SLOW
}

internal fun formatElapsed(ms: Long): String {
    val s = (ms / 1000).coerceAtLeast(0)
    return "${s / 60}:${(s % 60).toString().padStart(2, '0')}"
}

private const val MANUAL_CHECK_COOLDOWN_MS = 20_000L
private const val EXPIRY_WARN_MS = 5 * 60_000L

private fun clock(iso: String): String = formatManila(iso, "h:mm a") ?: "its expiry"

// Mobile mirror of website `CheckoutModal`: review & pay, then the QR Ph step.
// Opens from the CartSheet's "Continue to checkout". QR Ph is the only payment
// method (owner decision 2026-10-17). FloatingCart routes a paid order to the
// receipt, also when it is paid after the sheet was closed.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CheckoutSheet(
    cartViewModel: CartViewModel,
    onDismiss: () -> Unit,
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

    val qr = (checkoutState as? CheckoutState.AwaitingQr)?.payment
    // A live QR may already have been paid, so leaving is a question, never a
    // slip (website requestClose). Cancelling locks the sheet until PayMongo answers.
    val qrLive = qr != null && !qr.unreachable
    // Locked while the QR is being made or cancelled, as on the website: the
    // backend's answer is on its way and decides what the runner sees next.
    val isLoading = checkoutState is CheckoutState.Loading
    val isLocked = isLoading || qr?.cancelling == true
    var confirmLeave by remember { mutableStateOf(false) }
    var confirmCancel by remember { mutableStateOf(false) }
    val askBeforeClosing by rememberUpdatedState(qrLive)
    val locked by rememberUpdatedState(isLocked)
    val sheetState = rememberModalBottomSheetState(
        skipPartiallyExpanded = true,
        confirmValueChange = { value ->
            when {
                value != SheetValue.Hidden -> true
                locked -> false
                askBeforeClosing -> { confirmLeave = true; false }
                else -> true
            }
        },
    )
    val scope = rememberCoroutineScope()
    val requestClose = {
        when {
            isLocked -> Unit
            qrLive -> confirmLeave = true
            else -> onDismiss()
        }
    }

    var emailInput by remember { mutableStateOf(cartViewModel.getLoggedInUserEmail()) }
    var couponInput by remember { mutableStateOf("") }
    // Same gate as the website's checkout identify step (EMAIL_REGEX): the
    // receipt + download links go to this address, so a typo is a lost order.
    val emailError = validateEmail(emailInput)

    val scrollState = rememberScrollState()

    // Load the auto-applied photographer coupons for the current cart so the
    // total shown is what the runner is actually charged, not the list price.
    LaunchedEffect(Unit) { cartViewModel.previewCoupon(null) }

    // Refresh owned-photo data on open so already-purchased flags are current.
    LaunchedEffect(Unit) { cartViewModel.fetchOrders() }

    // A question asked about one QR never carries over to the next one.
    LaunchedEffect(qr?.order?.id) {
        confirmLeave = false
        confirmCancel = false
    }

    ModalBottomSheet(
        onDismissRequest = requestClose,
        sheetState = sheetState,
        containerColor = Bone,
        contentColor = Ink,
    ) {
        BackHandler(enabled = qrLive || isLocked) { requestClose() }

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
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Kicker("Checkout")
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = when {
                            qr?.unreachable == true -> "Check your inbox."
                            qr != null && qr.paidClaimedAt != null -> "Confirming your payment."
                            qr != null -> "Scan to pay."
                            isLoading && net == 0.0 -> "Placing your order…"
                            isLoading -> "Creating your QR code…"
                            net == 0.0 -> "Review your order"
                            else -> "Review & pay"
                        },
                        style = Typography.headlineSmall,
                        color = Ink,
                    )
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
                when {
                    qr?.unreachable == true -> UnreachableNotice(
                        email = qr.email,
                        orderId = qr.order.id,
                        onDone = {
                            cartViewModel.resetCheckoutState()
                            onDismiss()
                        },
                    )
                    qr != null -> QrPaymentStep(
                        payment = qr,
                        onPaid = { cartViewModel.claimQrPaid() },
                        onCancel = { confirmCancel = true },
                        onLeaveForEmail = onDismiss,
                    )
                    else -> PayStep(
                        checkoutState = checkoutState,
                        emailInput = emailInput,
                        onEmailChange = { emailInput = it },
                        emailError = emailError,
                        couponInput = couponInput,
                        onCouponChange = { couponInput = it.uppercase() },
                        couponLoading = couponLoading,
                        couponError = couponError,
                        onApplyCoupon = { cartViewModel.previewCoupon(couponInput) },
                        couponLabel = couponPreview?.code?.let { "Coupon $it" } ?: "Coupon discount",
                        ownedCount = ownedItems.size,
                        onRemoveOwned = { ownedItems.forEach { cartViewModel.removeFromCart(it.photoId) } },
                        photoCount = cartItems.size,
                        discount = discount,
                        net = net,
                        isLoading = isLoading,
                        onPay = { cartViewModel.checkout(emailInput) },
                    )
                }
            }
        }
    }

    if (confirmLeave && qr != null) {
        AlertDialog(
            onDismissRequest = { confirmLeave = false },
            containerColor = Bone,
            title = { Text("Leave checkout?") },
            text = {
                Text(
                    "Your QR code stays valid until ${clock(qr.qr.expiresAt)}. If you've already paid, " +
                        "your photos and receipt are safe — we'll email them to ${qr.email}. " +
                        "You can come back to this payment from the cart."
                )
            },
            confirmButton = {
                GhostCta("Leave", onClick = {
                    confirmLeave = false
                    onDismiss()
                })
            },
            dismissButton = {
                GhostCta("Stay", onClick = {
                    confirmLeave = false
                    scope.launch { sheetState.show() }
                })
            },
        )
    }

    if (confirmCancel && qr != null) {
        AlertDialog(
            onDismissRequest = { confirmCancel = false },
            containerColor = Bone,
            title = { Text("Cancel this payment?") },
            text = {
                Text(
                    "The QR code stops being tied to this order and nothing is charged. " +
                        "Already paid? Choose Keep waiting — we'll confirm it in a moment."
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    confirmCancel = false
                    cartViewModel.cancelQr()
                }) { Text("Cancel payment", style = Typography.labelLarge, color = ErrorRed) }
            },
            dismissButton = { GhostCta("Keep waiting", onClick = { confirmCancel = false }) },
        )
    }
}

@Composable
private fun PayStep(
    checkoutState: CheckoutState,
    emailInput: String,
    onEmailChange: (String) -> Unit,
    emailError: String?,
    couponInput: String,
    onCouponChange: (String) -> Unit,
    couponLoading: Boolean,
    couponError: String?,
    onApplyCoupon: () -> Unit,
    couponLabel: String,
    ownedCount: Int,
    onRemoveOwned: () -> Unit,
    photoCount: Int,
    discount: Double,
    net: Double,
    isLoading: Boolean,
    onPay: () -> Unit,
) {
    // Why the runner is back here — the first thing they read (website PaymentNotice).
    when (checkoutState) {
        is CheckoutState.Ended -> {
            val (tone, title, body) = when (checkoutState.outcome) {
                QrOutcome.EXPIRED -> Triple(NoticeTone.NEUTRAL, "Your QR code expired", "No payment was detected before it ran out. Generate a new code to pay.")
                QrOutcome.FAILED -> Triple(NoticeTone.ERROR, "Payment didn't go through", "Your bank or e-wallet didn't complete it. Try again, or use a different app.")
                QrOutcome.CANCELLED -> Triple(NoticeTone.NEUTRAL, "Payment cancelled", "Your cart is unchanged. Generate a new code whenever you're ready.")
            }
            PaymentNotice(tone = tone, kicker = "Nothing was charged", title = title) { NoticeBody(body) }
            Spacer(modifier = Modifier.height(20.dp))
        }
        is CheckoutState.Error -> {
            PaymentNotice(tone = NoticeTone.ERROR, kicker = "Try again", title = "Couldn't create a QR code") {
                NoticeBody(checkoutState.message)
            }
            Spacer(modifier = Modifier.height(20.dp))
        }
        else -> Unit
    }

    Kicker("Recipient email")
    Spacer(modifier = Modifier.height(8.dp))
    TextField(
        value = emailInput,
        onValueChange = onEmailChange,
        placeholder = { Text("Email for receipt + downloads", color = SlateSoft) },
        singleLine = true,
        enabled = !isLoading,
        colors = fieldColors(),
        shape = FieldShape,
        modifier = Modifier.fillMaxWidth(),
    )
    if (emailInput.isNotBlank() && emailError != null) {
        Spacer(modifier = Modifier.height(6.dp))
        Text(text = emailError, style = Typography.bodySmall, color = ErrorRed)
    }

    // A ₱0 order (100% giveaway) has nothing to pay with.
    if (net > 0.0) {
        Spacer(modifier = Modifier.height(20.dp))
        Kicker("Payment method")
        Spacer(modifier = Modifier.height(8.dp))
        QpCard {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(36.dp)
                        .background(Ink, CircleShape),
                    contentAlignment = Alignment.Center,
                ) {
                    Text("QR", style = NumeralStyle.copy(fontSize = 13.sp), color = SurfaceWhite)
                }
                Spacer(modifier = Modifier.width(14.dp))
                Column {
                    Text("QR Ph", style = Typography.titleSmall, color = Ink)
                    Text(
                        "Pay from any participating bank or e-wallet app that can scan QR Ph codes.",
                        style = Typography.bodySmall,
                        color = SlateSoft,
                    )
                }
            }
        }
    }

    Spacer(modifier = Modifier.height(24.dp))

    // Coupon code (V47/V50). Auto photographer coupons are already priced
    // into the total below; this field is the optional typed override.
    // Deliberately not a Fresh element — the pay CTA is the one accent.
    Kicker("Coupon code")
    Spacer(modifier = Modifier.height(8.dp))
    Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        TextField(
            value = couponInput,
            onValueChange = onCouponChange,
            placeholder = { Text("Optional", color = SlateSoft) },
            singleLine = true,
            enabled = !isLoading,
            colors = fieldColors(),
            shape = FieldShape,
            modifier = Modifier.weight(1f),
        )
        Spacer(modifier = Modifier.width(8.dp))
        TextButton(
            onClick = onApplyCoupon,
            enabled = !couponLoading && couponInput.isNotBlank() && !isLoading,
        ) {
            Text(text = if (couponLoading) "…" else "Apply", style = Typography.labelLarge, color = Ink)
        }
    }
    couponError?.let { err ->
        Spacer(modifier = Modifier.height(6.dp))
        Text(text = err, style = Typography.bodySmall, color = ErrorRed)
    }

    Spacer(modifier = Modifier.height(20.dp))

    if (ownedCount > 0) {
        Surface(
            color = BoneDeep,
            border = BorderStroke(1.dp, Line),
            shape = FieldShape,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "You already own $ownedCount of these ${if (ownedCount == 1) "photo" else "photos"}.",
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
                    onClick = onRemoveOwned,
                    contentPadding = PaddingValues(horizontal = 4.dp, vertical = 2.dp),
                ) {
                    Text(
                        text = "✕ Remove already-purchased" + if (ownedCount > 1) " ($ownedCount)" else "",
                        style = Typography.labelMedium,
                        color = Ink,
                    )
                }
            }
        }
        Spacer(modifier = Modifier.height(16.dp))
    }

    QpCard {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Photos", style = Typography.bodyMedium, color = Slate)
            Text(text = "$photoCount", style = NumeralStyle.copy(fontSize = 14.sp), color = Ink)
        }
        if (discount > 0.0) {
            Spacer(modifier = Modifier.height(8.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(text = couponLabel, style = Typography.bodyMedium, color = Slate)
                Text(text = "−${peso(discount)}", style = NumeralStyle.copy(fontSize = 14.sp), color = Slate)
            }
        }
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.Bottom,
        ) {
            Kicker("Total")
            Text(text = peso(net), style = NumeralStyle.copy(fontSize = 28.sp), color = Ink)
        }
        Spacer(modifier = Modifier.height(16.dp))
        PrimaryCta(
            text = when {
                net == 0.0 -> "Complete order →"
                checkoutState is CheckoutState.Ended -> "Generate a new QR ${peso(net)} →"
                else -> "Generate QR to pay ${peso(net)} →"
            },
            onClick = onPay,
            modifier = Modifier.fillMaxWidth(),
            loading = isLoading,
            enabled = emailError == null && ownedCount == 0,
        )
        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text = when {
                ownedCount > 0 -> "Remove already-purchased photos to continue"
                net == 0.0 -> "Nothing to pay · Watermark removed on download"
                else -> "Nothing is charged until you scan · Watermark removed on download"
            },
            style = Typography.bodySmall,
            color = SlateSoft,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun ColumnScope.QrPaymentStep(
    payment: QrCheckout,
    onPaid: () -> Unit,
    onCancel: () -> Unit,
    onLeaveForEmail: () -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    // 1 s clock for the countdown, the elapsed kicker and the check cooldown.
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) {
        while (true) {
            now = System.currentTimeMillis()
            delay(1_000)
        }
    }
    val qrBytes = remember(payment.qr.imageUrl) { qrImageBytes(payment.qr.imageUrl) }
    val qrBitmap = remember(qrBytes) {
        qrBytes?.let { runCatching { BitmapFactory.decodeByteArray(it, 0, it.size)?.asImageBitmap() }.getOrNull() }
    }
    var saveMessage by remember { mutableStateOf<String?>(null) }

    val total = payment.order.checkoutTotalAmount ?: payment.order.totalAmount
    val ref = formatOrderRef(payment.order.id)
    val mode = qrWaitMode(payment.paidClaimedAt, now)
    val elapsed = payment.paidClaimedAt?.let { now - it } ?: 0L
    val cooldownLeft = (payment.lastManualCheckAt + MANUAL_CHECK_COOLDOWN_MS - now).coerceAtLeast(0)
    val expiresAt = parseInstantMillis(payment.qr.expiresAt)
    val remaining = expiresAt?.let { it - now }

    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.fillMaxWidth()) {
        Kicker("Amount due · ${peso(total)}")
        Spacer(modifier = Modifier.height(12.dp))
        Box(
            modifier = Modifier
                .background(SurfaceWhite, QpCardShape)
                .border(1.dp, Line, QpCardShape)
                .padding(12.dp),
            contentAlignment = Alignment.Center,
        ) {
            if (qrBitmap != null) {
                Image(
                    bitmap = qrBitmap,
                    contentDescription = "QR Ph payment code for ${peso(total)}",
                    modifier = Modifier.size(248.dp),
                )
            } else {
                Text(
                    "We couldn't show this QR code. Cancel the payment and generate a new one.",
                    style = Typography.bodySmall,
                    color = SlateSoft,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .size(248.dp)
                        .padding(24.dp),
                )
            }
        }
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = (if (remaining == null) "" else if (remaining > 0) "Expires in ${formatElapsed(remaining)} · " else "Expired ${clock(payment.qr.expiresAt)} · ") + "Ref $ref",
            style = NumeralStyle.copy(fontSize = 12.sp),
            color = if (remaining != null && remaining < EXPIRY_WARN_MS) WarningOrange else SlateSoft,
        )
    }

    payment.qr.testUrl?.let { testUrl ->
        Spacer(modifier = Modifier.height(20.dp))
        TestModePanel(testUrl)
    }

    Spacer(modifier = Modifier.height(20.dp))
    GhostCta(
        text = "Save QR code",
        enabled = qrBitmap != null,
        onClick = {
            val bytes = qrBytes ?: return@GhostCta
            scope.launch {
                saveMessage = when (
                    val result = PhotoDownloader.saveImageToGallery(context, bytes, "quickpitik-qrph-${payment.order.id}.png")
                ) {
                    is PhotoDownloader.Result.Saved -> "Saved to Pictures/QuickPitik. Open it from your bank or e-wallet app's Scan QR screen."
                    is PhotoDownloader.Result.Failed -> result.message
                }
            }
        },
        modifier = Modifier.fillMaxWidth(),
    )
    saveMessage?.let {
        Spacer(modifier = Modifier.height(6.dp))
        Text(it, style = Typography.bodySmall, color = Slate)
    }

    Spacer(modifier = Modifier.height(20.dp))
    PaymentNotice(
        tone = if (mode == QrWaitMode.SLOW) NoticeTone.SLOW else NoticeTone.NEUTRAL,
        kicker = when (mode) {
            QrWaitMode.AWAITING -> null
            QrWaitMode.CONFIRMING -> "Confirming · ${formatElapsed(elapsed)}"
            QrWaitMode.SLOW -> "Still confirming · ${formatElapsed(elapsed)}"
        },
        title = when (mode) {
            QrWaitMode.AWAITING -> "Waiting for your payment"
            QrWaitMode.CONFIRMING -> "Confirming your payment…"
            QrWaitMode.SLOW -> "Taking longer than usual"
        },
    ) {
        when (mode) {
            QrWaitMode.AWAITING -> NoticeBody(
                "Pay with your bank or e-wallet app. We check automatically every few seconds — no need to refresh."
            )
            QrWaitMode.CONFIRMING -> {
                NoticeBody(
                    "Banks usually take 10–30 seconds to report back. Don't pay again — your ${peso(total)} is tied to Ref $ref."
                )
                Spacer(modifier = Modifier.height(6.dp))
                NoticeBody(
                    "Safe to leave — we'll pick up where you left off, and your receipt goes to ${payment.email} either way."
                )
            }
            QrWaitMode.SLOW -> NoticeBody(
                "Your payment is still safe. If it went through, this screen will confirm it and we'll email your " +
                    "download links to ${payment.email} — even if you leave. Please don't pay a second time."
            )
        }
    }

    if (mode == QrWaitMode.AWAITING) {
        Spacer(modifier = Modifier.height(16.dp))
        QpCard {
            Kicker("How to pay")
            Spacer(modifier = Modifier.height(10.dp))
            listOf(
                "Tap Save QR code, or show this screen to another phone.",
                "In a participating bank or e-wallet app, choose Scan QR, then upload the saved image from your gallery.",
                "Confirm the exact amount in your app, then tap I've paid below. We detect the payment automatically either way.",
            ).forEachIndexed { i, step ->
                Row(modifier = Modifier.padding(vertical = 4.dp)) {
                    Text("0${i + 1}", style = NumeralStyle.copy(fontSize = 13.sp), color = Ink)
                    Spacer(modifier = Modifier.width(12.dp))
                    Text(step, style = Typography.bodySmall, color = InkSoft)
                }
            }
        }
    }

    payment.checkError?.let {
        Spacer(modifier = Modifier.height(12.dp))
        Text(it, style = Typography.bodySmall, color = ErrorRed)
    }

    Spacer(modifier = Modifier.height(20.dp))
    if (mode == QrWaitMode.AWAITING) {
        PrimaryCta(
            text = "I've paid",
            onClick = onPaid,
            modifier = Modifier.fillMaxWidth(),
            loading = payment.checking,
        )
    } else {
        GhostCta(
            text = when {
                payment.checking -> "Checking…"
                cooldownLeft > 0 -> "Checking again in ${(cooldownLeft + 999) / 1000}s"
                else -> "Check again"
            },
            onClick = onPaid,
            enabled = !payment.checking && cooldownLeft == 0L,
            modifier = Modifier.fillMaxWidth(),
        )
    }

    if (mode == QrWaitMode.SLOW) {
        TextButton(onClick = onLeaveForEmail, modifier = Modifier.align(Alignment.CenterHorizontally)) {
            Text("I'll wait for the email →", style = Typography.labelLarge, color = Ink)
        }
    }

    // The way out. Quiet on purpose — under the payment actions, never beside
    // them — but always there, so a pending screen is never a trap.
    if (!payment.checking) {
        TextButton(
            onClick = onCancel,
            enabled = !payment.cancelling,
            modifier = Modifier.align(Alignment.CenterHorizontally),
        ) {
            Text(
                text = if (payment.cancelling) "Cancelling…" else "Cancel payment",
                style = Typography.labelLarge,
                color = Slate,
            )
        }
    }
}

// Test mode only: the backend sends testUrl solely on an sk_test_ key. The QR
// itself is still a *real* QR Ph code — scanning it moves real money.
@Composable
private fun TestModePanel(testUrl: String) {
    val context = LocalContext.current
    val uriHandler = LocalUriHandler.current
    QpCard {
        Kicker("Test mode · PayMongo sandbox")
        Spacer(modifier = Modifier.height(6.dp))
        NoticeBody(
            "Don't scan this code with a real app — it would charge you. Use the simulator to authorize " +
                "or fail this payment, then come back and tap I've paid."
        )
        if (isSafeCheckoutUrl(testUrl)) {
            Spacer(modifier = Modifier.height(10.dp))
            GhostCta(
                text = "Open PayMongo simulator ↗",
                onClick = {
                    runCatching {
                        try {
                            CustomTabsIntent.Builder()
                                .setShowTitle(true)
                                .setDefaultColorSchemeParams(
                                    CustomTabColorSchemeParams.Builder().setToolbarColor(Bone.toArgb()).build()
                                )
                                .build()
                                .launchUrl(context, Uri.parse(testUrl))
                        } catch (e: ActivityNotFoundException) {
                            uriHandler.openUri(testUrl)
                        }
                    }
                },
            )
        }
    }
}

// The guest pass ran out or the order is gone: checking is over, but the
// backend still emails the receipt when the payment settles.
@Composable
private fun UnreachableNotice(email: String, orderId: String, onDone: () -> Unit) {
    QpCard {
        Text("We can't check this payment from here any more.", style = Typography.titleSmall, color = Ink)
        Spacer(modifier = Modifier.height(4.dp))
        NoticeBody(
            "If you paid, your receipt and download links are on their way to $email. " +
                "If you didn't, nothing was charged and you can start a fresh checkout."
        )
        Spacer(modifier = Modifier.height(12.dp))
        Kicker("Ref · ${formatOrderRef(orderId)}")
    }
    Spacer(modifier = Modifier.height(20.dp))
    GhostCta(text = "Done", onClick = onDone, modifier = Modifier.fillMaxWidth())
}

private enum class NoticeTone(val border: Color, val dot: Color) {
    NEUTRAL(Line, SlateSoft),
    ERROR(ErrorRed, ErrorRed),
    SLOW(WarningOrange, WarningOrange),
}

@Composable
private fun PaymentNotice(
    tone: NoticeTone,
    kicker: String?,
    title: String,
    content: @Composable () -> Unit,
) {
    Surface(
        color = BoneDeep,
        border = BorderStroke(1.dp, tone.border),
        shape = FieldShape,
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(modifier = Modifier.padding(16.dp)) {
            Box(
                modifier = Modifier
                    .padding(top = 7.dp)
                    .size(8.dp)
                    .background(tone.dot, CircleShape)
            )
            Spacer(modifier = Modifier.width(12.dp))
            Column {
                kicker?.let {
                    Kicker(it)
                    Spacer(modifier = Modifier.height(4.dp))
                }
                Text(title, style = Typography.titleSmall, color = Ink)
                Spacer(modifier = Modifier.height(4.dp))
                content()
            }
        }
    }
}

@Composable
private fun NoticeBody(text: String) {
    Text(text, style = Typography.bodySmall, color = InkSoft)
}

@Composable
private fun fieldColors() = TextFieldDefaults.colors(
    focusedContainerColor = BoneDeep,
    unfocusedContainerColor = BoneDeep,
    focusedIndicatorColor = Fresh,
    unfocusedIndicatorColor = Color.Transparent,
    focusedTextColor = Ink,
    unfocusedTextColor = InkSoft,
)
