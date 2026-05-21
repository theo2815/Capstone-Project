package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CartScreen(
    viewModel: CartViewModel,
    onNavigateBack: () -> Unit,
    onNavigateToCheckout: () -> Unit
) {
    val cartItems by viewModel.cartItems.collectAsState()
    val totalAmount by viewModel.cartTotal.collectAsState()

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Bone
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            // Header Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = onNavigateBack,
                    colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                ) {
                    Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Ink)
                }
                Spacer(modifier = Modifier.width(16.dp))
                Column {
                    Text(
                        text = "MARKETPLACE",
                        style = Typography.labelSmall,
                        color = Slate
                    )
                    Text(
                        text = "Your Cart",
                        style = Typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                }
            }
            Spacer(modifier = Modifier.height(24.dp))

            if (cartItems.isEmpty()) {
                // Empty Cart State
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.padding(32.dp)
                    ) {
                        Box(
                            modifier = Modifier
                                .size(80.dp)
                                .clip(RoundedCornerShape(40.dp))
                                .background(BoneDeep),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.ShoppingCart,
                                contentDescription = "Empty Cart",
                                tint = SlateSoft,
                                modifier = Modifier.size(36.dp)
                            )
                        }
                        Spacer(modifier = Modifier.height(24.dp))
                        Text(
                            text = "Your Cart is Empty",
                            style = Typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "Browse the matched photo streams and add previews to your shopping cart to unlock high-res downloads.",
                            style = Typography.bodyMedium,
                            color = SlateSoft,
                            textAlign = TextAlign.Center
                        )
                        Spacer(modifier = Modifier.height(24.dp))
                        Button(
                            onClick = onNavigateBack,
                            colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                            shape = RoundedCornerShape(12.dp)
                        ) {
                            Text("GO BACK BROWSING", style = Typography.labelMedium)
                        }
                    }
                }
            } else {
                // Items List
                LazyColumn(
                    modifier = Modifier.weight(1f),
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    items(cartItems) { item ->
                        Card(
                            colors = CardDefaults.cardColors(containerColor = BoneDeep),
                            border = BorderStroke(1.dp, Line),
                            shape = RoundedCornerShape(16.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                // Thumbnail Container
                                Box(
                                    modifier = Modifier
                                        .size(70.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(Line),
                                    contentAlignment = Alignment.Center
                                ) {
                                    if (item.thumbnailUrl != null) {
                                        AsyncImage(
                                            model = item.thumbnailUrl,
                                            contentDescription = "Photo thumbnail",
                                            modifier = Modifier.fillMaxSize()
                                        )
                                    } else {
                                        // Standard fallback placeholder
                                        Text(
                                            text = "QUICK\nPITIK",
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Bold,
                                            color = Slate,
                                            textAlign = TextAlign.Center,
                                            lineHeight = 10.sp
                                        )
                                    }
                                }
                                Spacer(modifier = Modifier.width(16.dp))

                                // Metadata Column
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = item.eventName ?: "Marathon Photo",
                                        style = Typography.titleSmall,
                                        fontWeight = FontWeight.Bold,
                                        color = Ink,
                                        maxLines = 1
                                    )
                                    Text(
                                        text = "Bib: ${item.bib ?: "N/A"} | Time: ${item.time ?: "N/A"}",
                                        style = Typography.bodySmall,
                                        color = SlateSoft
                                    )
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(
                                        text = String.format("₱%,.2f", item.price),
                                        style = Typography.titleMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = Fresh
                                    )
                                }

                                // Delete Button
                                IconButton(
                                    onClick = { viewModel.removeFromCart(item.photoId) },
                                    colors = IconButtonDefaults.iconButtonColors(contentColor = ErrorRed)
                                ) {
                                    Icon(Icons.Default.Delete, contentDescription = "Delete")
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(24.dp))

                // Price Summary Card
                Card(
                    colors = CardDefaults.cardColors(containerColor = SurfaceWhite),
                    border = BorderStroke(1.dp, Line),
                    shape = RoundedCornerShape(20.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(20.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Subtotal", style = Typography.bodyMedium, color = Slate)
                            Text(String.format("₱%,.2f", totalAmount), style = Typography.bodyMedium, color = Ink)
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text("Platform Fee", style = Typography.bodyMedium, color = Slate)
                            Text("FREE", style = Typography.bodyMedium, fontWeight = FontWeight.Bold, color = Fresh)
                        }
                        Divider(modifier = Modifier.padding(vertical = 12.dp), color = Line)
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Total", style = Typography.titleMedium, fontWeight = FontWeight.Bold, color = Ink)
                            Text(
                                text = String.format("₱%,.2f", totalAmount),
                                style = Typography.titleLarge,
                                fontWeight = FontWeight.Bold,
                                color = Ink
                            )
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = onNavigateToCheckout,
                            colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("PROCEED TO CHECKOUT", style = Typography.labelMedium)
                        }
                    }
                }
            }
        }
    }
}
