package com.quickpitik.mobile.ui.runner

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
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
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import androidx.navigation.compose.currentBackStackEntryAsState
import com.quickpitik.mobile.ui.theme.*

// Global floating-cart overlay — the mobile mirror of website FloatingCart.
// Mounted once at the root of MainActivity above the NavHost. Hides on auth
// routes and when the cart is empty. Pill can be minimized to a half-circle
// handle pinned to the right edge; auto-restores when a new item arrives.
@Composable
fun FloatingCart(
    navController: NavController,
    cartViewModel: CartViewModel,
    onAfterCheckoutSuccess: () -> Unit,
) {
    val items by cartViewModel.cartItems.collectAsState()
    val total by cartViewModel.cartTotal.collectAsState()
    val cartOpen by cartViewModel.cartSheetOpen.collectAsState()
    val checkoutOpen by cartViewModel.checkoutSheetOpen.collectAsState()
    val expressPending by cartViewModel.expressCheckoutPending.collectAsState()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val route = backStackEntry?.destination?.route

    val itemCount = items.size
    var minimized by remember { mutableStateOf(false) }
    var prevCount by remember { mutableStateOf(itemCount) }
    val badgeScale = remember { Animatable(1f) }

    // Count grew → pulse the badge, force-restore the pill, optionally jump
    // straight to checkout if a "Buy now" CTA set the express flag.
    LaunchedEffect(itemCount) {
        if (itemCount > prevCount) {
            minimized = false
            badgeScale.snapTo(1.35f)
            badgeScale.animateTo(1f, tween(durationMillis = 420))
            if (expressPending) {
                cartViewModel.openCheckoutSheet()
                cartViewModel.clearExpressCheckout()
            }
        }
        prevCount = itemCount
    }

    // Opening either sheet always brings the pill back — keeps state coherent.
    LaunchedEffect(cartOpen, checkoutOpen) {
        if (cartOpen || checkoutOpen) minimized = false
    }

    val hiddenRoute = route == "login" || route == "register"
    val showPill = itemCount > 0 && !hiddenRoute

    if (showPill) {
        // No padding on the parent — pill applies its own inset (so it sits
        // 20dp from the right edge), and the handle is offset by half its
        // diameter from the *actual* screen edge so half of it hangs off.
        Box(
            modifier = Modifier
                .fillMaxSize()
                .navigationBarsPadding(),
        ) {
            val pillOffsetX by animateDpAsState(
                targetValue = if (minimized) 280.dp else 0.dp,
                animationSpec = tween(durationMillis = 420),
                label = "qp-cart-pill-x",
            )
            val handleOffsetX by animateDpAsState(
                targetValue = if (minimized) 28.dp else 100.dp,
                animationSpec = tween(durationMillis = 420),
                label = "qp-cart-handle-x",
            )

            // ── Full pill ────────────────────────────────────────────
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(end = 20.dp, bottom = 24.dp)
                    .offset(x = pillOffsetX),
            ) {
                Row(
                    modifier = Modifier
                        .height(56.dp)
                        .clip(PillShape)
                        .background(Fresh)
                        .clickable(enabled = !minimized) { cartViewModel.openCartSheet() }
                        .padding(start = 8.dp, end = 20.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Box(
                        modifier = Modifier
                            .size(40.dp)
                            .clip(CircleShape)
                            .background(Color.White.copy(alpha = 0.18f)),
                        contentAlignment = Alignment.Center,
                    ) {
                        Icon(
                            imageVector = Icons.Default.ShoppingCart,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(20.dp),
                        )
                    }
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(
                            text = "CART",
                            style = Typography.labelSmall,
                            color = Color.White.copy(alpha = 0.75f),
                        )
                        Text(
                            text = "₱${"%,d".format(total.toInt())}",
                            style = NumeralStyle.copy(fontSize = 13.sp),
                            color = Color.White,
                        )
                    }
                }

                // Count badge — anchored over the top-right curve of the icon disc.
                Box(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .offset(x = 36.dp, y = -2.dp)
                        .scale(badgeScale.value)
                        .defaultMinSize(minWidth = 22.dp, minHeight = 22.dp)
                        .clip(CircleShape)
                        .background(Ink)
                        .border(2.dp, Fresh, CircleShape)
                        .padding(horizontal = 5.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = if (itemCount > 99) "99+" else itemCount.toString(),
                        style = NumeralStyle.copy(fontSize = 10.sp),
                        color = Color.White,
                    )
                }

                // Minimize chip — detached top-right of pill, doesn't crowd the count badge.
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .offset(x = 6.dp, y = -6.dp)
                        .size(28.dp)
                        .clip(CircleShape)
                        .background(Ink)
                        .border(2.dp, Bone, CircleShape)
                        .clickable(enabled = !minimized) { minimized = true },
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Hide cart",
                        tint = Color.White,
                        modifier = Modifier.size(14.dp),
                    )
                }
            }

            // ── Restore handle — half-circle pinned to the right edge ──
            // Outer Box doesn't clip — only the disc does. Otherwise the count
            // badge anchored over the disc's top-left corner gets eaten by the
            // CircleShape inscribed-circle path (the corners of a 56x56 box
            // sit outside a 28dp-radius circle).
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(bottom = 24.dp)
                    .offset(x = handleOffsetX),
            ) {
                Box(
                    modifier = Modifier
                        .size(56.dp)
                        .clip(CircleShape)
                        .background(Fresh)
                        .clickable(enabled = minimized) { minimized = false },
                ) {
                    Icon(
                        imageVector = Icons.Default.ShoppingCart,
                        contentDescription = "Show cart",
                        tint = Color.White,
                        modifier = Modifier
                            .align(Alignment.CenterStart)
                            .padding(start = 8.dp)
                            .size(22.dp),
                    )
                }
                // Count badge — sibling of disc, not subject to CircleShape clip.
                Box(
                    modifier = Modifier
                        .align(Alignment.TopStart)
                        .offset(x = -4.dp, y = -4.dp)
                        .defaultMinSize(minWidth = 22.dp, minHeight = 22.dp)
                        .clip(CircleShape)
                        .background(Ink)
                        .border(2.dp, Fresh, CircleShape)
                        .padding(horizontal = 5.dp),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = if (itemCount > 99) "99+" else itemCount.toString(),
                        style = NumeralStyle.copy(fontSize = 10.sp),
                        color = Color.White,
                    )
                }
            }
        }
    }

    // Sheets always allowed to render — they read sheet flags independently.
    if (cartOpen) {
        CartSheet(
            cartViewModel = cartViewModel,
            onDismiss = { cartViewModel.closeCartSheet() },
            onContinueToCheckout = { cartViewModel.openCheckoutSheet() },
            onBrowseEvents = {
                cartViewModel.closeCartSheet()
                navController.navigate("events")
            },
        )
    }
    if (checkoutOpen) {
        CheckoutSheet(
            cartViewModel = cartViewModel,
            onDismiss = { cartViewModel.closeCheckoutSheet() },
            onCheckoutSuccess = {
                cartViewModel.closeCheckoutSheet()
                onAfterCheckoutSuccess()
            },
        )
    }
}
