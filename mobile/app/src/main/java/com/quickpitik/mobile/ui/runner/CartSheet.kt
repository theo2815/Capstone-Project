package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Divider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.*

// Mobile mirror of website `CartModal`. Opens as a full-height
// ModalBottomSheet from the FloatingCart pill. Mirrors the website's header
// copy ("Your cart" + "X photos."), watermark explainer, and the
// continue-to-checkout + keep-browsing + clear-cart action triad.
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CartSheet(
    cartViewModel: CartViewModel,
    onDismiss: () -> Unit,
    onContinueToCheckout: () -> Unit,
    onBrowseEvents: () -> Unit,
) {
    val items by cartViewModel.cartItems.collectAsState()
    val total by cartViewModel.cartTotal.collectAsState()
    val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)
    val hapticFire = rememberQpHaptic()
    val itemCount = items.size
    var previewIndex by remember { mutableStateOf<Int?>(null) }
    var showClearConfirm by remember { mutableStateOf(false) }
    val snackbarHostState = remember { SnackbarHostState() }
    var snackbarMessage by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(snackbarMessage) {
        snackbarMessage?.let { msg ->
            snackbarHostState.showSnackbar(msg)
            snackbarMessage = null
        }
    }

    // If a removal shrinks the list past the previewed slot, clamp it.
    // When the last item is removed, the preview closes itself.
    LaunchedEffect(items.size) {
        previewIndex = previewIndex?.let { idx ->
            when {
                items.isEmpty() -> null
                idx >= items.size -> items.size - 1
                else -> idx
            }
        }
    }

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = sheetState,
        containerColor = Bone,
        contentColor = Ink,
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .navigationBarsPadding(),
        ) {
            // ── Header ──────────────────────────────────────────────
            Column(modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp)) {
                Kicker("Your cart")
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = when (itemCount) {
                        0 -> "Empty for now."
                        1 -> "1 photo."
                        else -> "$itemCount photos."
                    },
                    style = Typography.headlineSmall,
                    color = Ink,
                )
            }

            Divider(color = Line)

            if (itemCount == 0) {
                EmptyCartBody(onBrowse = {
                    onDismiss()
                    onBrowseEvents()
                })
            } else {
                LazyColumn(
                    modifier = Modifier
                        .weight(1f, fill = false)
                        .heightIn(max = 480.dp),
                ) {
                    items(items, key = { it.photoId }) { item ->
                        val idx = items.indexOf(item)
                        CartSheetRow(
                            item = item,
                            onPreview = { previewIndex = idx },
                            onRemove = {
                                hapticFire(QpHaptic.REMOVE)
                                cartViewModel.removeFromCart(item.photoId)
                            },
                        )
                        Divider(color = Line)
                    }
                }

                // ── Footer ──────────────────────────────────────────
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(BoneDeep)
                        .padding(horizontal = 24.dp, vertical = 20.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Bottom,
                    ) {
                        Kicker("Subtotal")
                        Text(
                            text = "₱${"%,.2f".format(total)}",
                            style = NumeralStyle.copy(fontSize = 32.sp),
                            color = Ink,
                        )
                    }
                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        text = "Watermarked previews are free. Pay once, download forever.",
                        style = Typography.bodySmall,
                        color = SlateSoft,
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    PrimaryCta(
                        text = "Continue to checkout →",
                        onClick = onContinueToCheckout,
                        modifier = Modifier.fillMaxWidth(),
                    )
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        TextButton(
                            onClick = {
                                onDismiss()
                                onBrowseEvents()
                            },
                        ) {
                            Text(
                                text = "← Keep browsing",
                                style = Typography.labelMedium,
                                color = Slate,
                            )
                        }
                        TextButton(
                            onClick = { showClearConfirm = true },
                        ) {
                            Text(
                                text = "Clear cart",
                                style = Typography.labelMedium,
                                color = SlateSoft,
                            )
                        }
                    }
                }
            }

            // Snackbar surface for "Cart cleared." feedback. Anchored above
            // the navigation bar so it doesn't sit under the system gesture
            // pill on phones with edge-to-edge nav.
            SnackbarHost(
                hostState = snackbarHostState,
                modifier = Modifier
                    .align(Alignment.CenterHorizontally)
                    .padding(bottom = 16.dp),
            )
        }
    }

    // Photo preview lightbox — sits on top of the bottom sheet as a Dialog.
    // Tapping a row sets previewIndex; the LaunchedEffect above clamps/closes
    // it if the cart shrinks underneath the preview.
    val activePreviewIndex = previewIndex
    if (activePreviewIndex != null && items.isNotEmpty()) {
        val safeIndex = activePreviewIndex.coerceIn(0, items.size - 1)
        PhotoPreview(
            photos = items.map { it.toPreviewData() },
            currentIndex = safeIndex,
            isInCart = { true },
            onClose = { previewIndex = null },
            onIndexChange = { newIndex -> previewIndex = newIndex },
            onToggleCart = { previewData ->
                hapticFire(QpHaptic.REMOVE)
                cartViewModel.removeFromCart(previewData.id)
                // Website parity: removing from inside the preview closes it.
                previewIndex = null
            },
            onBuyNow = {
                previewIndex = null
                onContinueToCheckout()
            },
        )
    }

    // Clear-cart confirmation — website parity (CartModal calls a confirm()
    // dialog before firing). Empty-state snackbar fires on accept so the
    // action feels acknowledged.
    if (showClearConfirm) {
        AlertDialog(
            onDismissRequest = { showClearConfirm = false },
            title = {
                Text("Clear cart?", style = Typography.titleMedium, color = Ink)
            },
            text = {
                Text(
                    text = "This removes every photo from your cart. You'll have to add them again to check out.",
                    style = Typography.bodyMedium,
                    color = Slate,
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        hapticFire(QpHaptic.REMOVE)
                        cartViewModel.clearCart()
                        showClearConfirm = false
                        snackbarMessage = "Cart cleared."
                    },
                ) {
                    Text("Clear cart", color = ErrorRed, style = Typography.labelMedium)
                }
            },
            dismissButton = {
                TextButton(onClick = { showClearConfirm = false }) {
                    Text("Cancel", color = Slate, style = Typography.labelMedium)
                }
            },
            containerColor = Bone,
        )
    }
}

@Composable
private fun CartSheetRow(
    item: com.quickpitik.mobile.data.remote.CartItemDto,
    onPreview: () -> Unit,
    onRemove: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onPreview)
            .padding(horizontal = 24.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(TileShape)
                .background(Line),
            contentAlignment = Alignment.Center,
        ) {
            if (item.thumbnailUrl != null) {
                AsyncImage(
                    model = RetrofitClient.resolveImageUrl(item.thumbnailUrl),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                )
            } else {
                Text(
                    text = "QUICK\nPITIK",
                    style = Typography.labelSmall,
                    color = Slate,
                    textAlign = TextAlign.Center,
                )
            }
        }
        Spacer(modifier = Modifier.width(16.dp))
        Column(modifier = Modifier.weight(1f)) {
            Kicker("Race photo")
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = item.eventName ?: "QuickPitik",
                style = Typography.titleMedium,
                color = Ink,
                maxLines = 2,
            )
            // Bib + time — web cart-modal parity ("Untagged" when no bib).
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = buildString {
                    append(item.bib?.takeIf { it.isNotBlank() }?.let { "Bib $it" } ?: "Untagged")
                    item.time?.takeIf { it.isNotBlank() }?.let { append(" · $it") }
                },
                style = NumeralStyle.copy(fontSize = 11.sp),
                color = SlateSoft,
            )
        }
        Column(horizontalAlignment = Alignment.End) {
            Text(
                text = "₱${"%,.0f".format(item.price)}",
                style = NumeralStyle.copy(fontSize = 14.sp),
                color = Ink,
            )
            Spacer(modifier = Modifier.height(6.dp))
            TextButton(
                onClick = onRemove,
                contentPadding = PaddingValues(horizontal = 4.dp, vertical = 2.dp),
            ) {
                Text(
                    text = "✕ Remove",
                    style = Typography.labelSmall,
                    color = SlateSoft,
                )
            }
        }
    }
}

@Composable
private fun EmptyCartBody(onBrowse: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 32.dp, vertical = 48.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            modifier = Modifier
                .size(64.dp)
                .clip(CircleShape)
                .background(Color.Transparent),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                imageVector = Icons.Default.ShoppingCart,
                contentDescription = null,
                tint = Slate,
                modifier = Modifier.size(28.dp),
            )
        }
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = "Nothing here yet.",
            style = Typography.titleLarge,
            color = Ink,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Open an event, search by bib or selfie, then add the photos you want to keep.",
            style = Typography.bodyMedium,
            color = SlateSoft,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(24.dp))
        PrimaryCta(text = "Browse events →", onClick = onBrowse)
    }
}
