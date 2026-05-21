package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OrdersScreen(
    viewModel: CartViewModel,
    onNavigateBack: () -> Unit
) {
    val ordersState by viewModel.ordersState.collectAsState()
    val orderDetailState by viewModel.orderDetailState.collectAsState()

    var selectedOrderId by remember { mutableStateOf<String?>(null) }
    val uriHandler = LocalUriHandler.current

    LaunchedEffect(selectedOrderId) {
        if (selectedOrderId != null) {
            viewModel.fetchOrderDetail(selectedOrderId!!)
        } else {
            viewModel.fetchOrders()
            viewModel.resetOrderDetailState()
        }
    }

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
            if (selectedOrderId == null) {
                // Main Orders List View
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
                            text = "Order History",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )
                    }
                }
                Spacer(modifier = Modifier.height(24.dp))

                when (val state = ordersState) {
                    is OrdersState.Loading -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            CircularProgressIndicator(color = Fresh)
                        }
                    }
                    is OrdersState.Error -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(24.dp)) {
                                Icon(Icons.Default.Warning, contentDescription = "Error", tint = ErrorRed, modifier = Modifier.size(48.dp))
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(state.message, color = ErrorRed, style = Typography.bodyMedium, textAlign = TextAlign.Center)
                            }
                        }
                    }
                    is OrdersState.Success -> {
                        if (state.orders.isEmpty()) {
                            Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.padding(32.dp)) {
                                    Text("No Orders Found", style = Typography.titleMedium, fontWeight = FontWeight.Bold, color = Ink)
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        text = "Any photo checkouts you complete on this account will be listed here for digital high-res downloading.",
                                        style = Typography.bodyMedium,
                                        color = SlateSoft,
                                        textAlign = TextAlign.Center
                                    )
                                }
                            }
                        } else {
                            LazyColumn(
                                modifier = Modifier.fillMaxWidth().weight(1f),
                                verticalArrangement = Arrangement.spacedBy(16.dp)
                            ) {
                                items(state.orders) { order ->
                                    Card(
                                        onClick = { selectedOrderId = order.id },
                                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                                        border = BorderStroke(1.dp, Line),
                                        shape = RoundedCornerShape(16.dp),
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(16.dp),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.SpaceBetween
                                        ) {
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = order.eventName ?: "Marathon Event",
                                                    style = Typography.titleSmall,
                                                    fontWeight = FontWeight.Bold,
                                                    color = Ink
                                                )
                                                Text(
                                                    text = "${order.photoIds.size} Photos | ${order.paymentMethod}",
                                                    style = Typography.bodySmall,
                                                    color = SlateSoft
                                                )
                                                Spacer(modifier = Modifier.height(4.dp))
                                                Text(
                                                    text = String.format("₱%,.2f", order.total),
                                                    style = Typography.titleSmall,
                                                    fontWeight = FontWeight.Bold,
                                                    color = Fresh
                                                )
                                            }
                                            Row(verticalAlignment = Alignment.CenterVertically) {
                                                Surface(
                                                    shape = RoundedCornerShape(8.dp),
                                                    color = if (order.status == "PAID") SuccessGreen.copy(alpha = 0.15f) else WarningOrange.copy(alpha = 0.15f)
                                                ) {
                                                    Text(
                                                        text = order.status ?: "PENDING",
                                                        color = if (order.status == "PAID") SuccessGreen else WarningOrange,
                                                        fontSize = 10.sp,
                                                        fontWeight = FontWeight.Bold,
                                                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                                    )
                                                }
                                                Spacer(modifier = Modifier.width(8.dp))
                                                Icon(Icons.Default.KeyboardArrowRight, contentDescription = "Detail", tint = Ink)
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Order Detail & High-Res Download View
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(
                        onClick = { selectedOrderId = null },
                        colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                    ) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "Back to List", tint = Ink)
                    }
                    Spacer(modifier = Modifier.width(16.dp))
                    Column {
                        Text(
                            text = "ORDER DETAILS",
                            style = Typography.labelSmall,
                            color = Slate
                        )
                        Text(
                            text = "Download Panel",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )
                    }
                }
                Spacer(modifier = Modifier.height(24.dp))

                when (val detailState = orderDetailState) {
                    is OrderDetailState.Loading -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            CircularProgressIndicator(color = Fresh)
                        }
                    }
                    is OrderDetailState.Error -> {
                        Box(modifier = Modifier.fillMaxWidth().weight(1f), contentAlignment = Alignment.Center) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Icon(Icons.Default.Warning, contentDescription = "Error", tint = ErrorRed, modifier = Modifier.size(48.dp))
                                Spacer(modifier = Modifier.height(16.dp))
                                Text(detailState.message, color = ErrorRed, style = Typography.bodyMedium, textAlign = TextAlign.Center)
                                Spacer(modifier = Modifier.height(16.dp))
                                Button(onClick = { selectedOrderId = null }, colors = ButtonDefaults.buttonColors(containerColor = Line, contentColor = Ink)) {
                                    Text("BACK TO LIST")
                                }
                            }
                        }
                    }
                    is OrderDetailState.Success -> {
                        val order = detailState.order
                        Column(modifier = Modifier.fillMaxWidth().weight(1f)) {
                            // Order Summary Summary Header Card
                            Card(
                                colors = CardDefaults.cardColors(containerColor = SurfaceWhite),
                                border = BorderStroke(1.dp, Line),
                                shape = RoundedCornerShape(16.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Column(modifier = Modifier.padding(16.dp)) {
                                    Text(
                                        text = order.eventName ?: "Event Photos Bundle",
                                        style = Typography.titleMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = Ink
                                    )
                                    Text(
                                        text = "Status: ${order.status} | Recipient: ${order.recipientEmail}",
                                        style = Typography.bodySmall,
                                        color = SlateSoft
                                    )
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = String.format("Total paid: ₱%,.2f", order.total),
                                            style = Typography.titleSmall,
                                            fontWeight = FontWeight.Bold,
                                            color = Fresh
                                        )
                                        if (order.downloadBundleUrl != null) {
                                            Button(
                                                onClick = { uriHandler.openUri(order.downloadBundleUrl) },
                                                colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                                                shape = RoundedCornerShape(8.dp)
                                            ) {
                                                Text("ZIP BUNDLE", style = Typography.labelSmall)
                                            }
                                        }
                                    }
                                }
                            }
                            Spacer(modifier = Modifier.height(16.dp))

                            Text(
                                text = "PURCHASED PHOTO STREAM",
                                style = Typography.labelSmall,
                                color = Slate,
                                modifier = Modifier.padding(bottom = 8.dp)
                            )

                            LazyColumn(
                                modifier = Modifier.fillMaxWidth().weight(1f),
                                verticalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                items(order.photos) { photo ->
                                    Card(
                                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                                        border = BorderStroke(1.dp, Line),
                                        shape = RoundedCornerShape(12.dp),
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(10.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Box(
                                                modifier = Modifier
                                                    .size(60.dp)
                                                    .clip(RoundedCornerShape(8.dp))
                                                    .background(Line),
                                                contentAlignment = Alignment.Center
                                            ) {
                                                if (photo.thumbnailUrl != null) {
                                                    AsyncImage(
                                                        model = photo.thumbnailUrl,
                                                        contentDescription = "Purchased photo thumbnail",
                                                        modifier = Modifier.fillMaxSize()
                                                    )
                                                } else {
                                                    Icon(
                                                        imageVector = Icons.Default.CheckCircle,
                                                        contentDescription = "Unlocked",
                                                        tint = Fresh
                                                    )
                                                }
                                            }
                                            Spacer(modifier = Modifier.width(16.dp))
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = "Bib: ${photo.bib ?: "N/A"}",
                                                    style = Typography.bodyMedium,
                                                    fontWeight = FontWeight.Bold,
                                                    color = Ink
                                                )
                                                Text(
                                                    text = "Time: ${photo.time}",
                                                    style = Typography.bodySmall,
                                                    color = SlateSoft
                                                )
                                            }
                                            
                                            // Download high res action button
                                            if (photo.downloadUrl != null) {
                                                Button(
                                                    onClick = { uriHandler.openUri(photo.downloadUrl) },
                                                    colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                                                    shape = RoundedCornerShape(8.dp)
                                                ) {
                                                    Text("DOWNLOAD", style = Typography.labelSmall)
                                                }
                                            } else {
                                                Text(
                                                    text = "Unlocking...",
                                                    style = Typography.bodySmall,
                                                    color = SlateSoft,
                                                    modifier = Modifier.padding(end = 8.dp)
                                                )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else -> {}
                }
            }
        }
    }
}
