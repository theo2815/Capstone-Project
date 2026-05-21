package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.CheckCircle
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
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CheckoutScreen(
    viewModel: CartViewModel,
    onNavigateBack: () -> Unit,
    onCheckoutSuccess: () -> Unit
) {
    val cartItems by viewModel.cartItems.collectAsState()
    val totalAmount by viewModel.cartTotal.collectAsState()
    val checkoutState by viewModel.checkoutState.collectAsState()

    var emailInput by remember { mutableStateOf(viewModel.getLoggedInUserEmail()) }
    var selectedPaymentMethod by remember { mutableStateOf("GCASH") } // GCASH, MAYA, CARD
    
    // Credit card fields
    var cardNumber by remember { mutableStateOf("") }
    var cardExpiry by remember { mutableStateOf("") }
    var cardCvv by remember { mutableStateOf("") }

    val scrollState = rememberScrollState()
    val uriHandler = LocalUriHandler.current

    LaunchedEffect(checkoutState) {
        if (checkoutState is CheckoutState.Success && (checkoutState as CheckoutState.Success).order.redirectUrl == null) {
            // If checkout succeeded and there's no redirectUrl (e.g. sandbox direct payment), auto navigate to orders after delay
            onCheckoutSuccess()
            viewModel.resetCheckoutState()
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Bone
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(24.dp)
                    .statusBarsPadding()
                    .navigationBarsPadding()
                    .verticalScroll(scrollState)
            ) {
                // Header
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
                            text = "Secure Checkout",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )
                    }
                }
                Spacer(modifier = Modifier.height(24.dp))

                // Email Form Field
                Text(
                    text = "RECIPIENT EMAIL",
                    style = Typography.labelSmall,
                    color = Slate,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                TextField(
                    value = emailInput,
                    onValueChange = { emailInput = it },
                    placeholder = { Text("Enter your email for receipt delivery", color = SlateSoft) },
                    singleLine = true,
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = BoneDeep,
                        unfocusedContainerColor = BoneDeep,
                        focusedIndicatorColor = Fresh,
                        unfocusedIndicatorColor = Color.Transparent,
                        focusedTextColor = Ink,
                        unfocusedTextColor = InkSoft
                    ),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(24.dp))

                // Payment Method Selector
                Text(
                    text = "SELECT PAYMENT PROVIDER",
                    style = Typography.labelSmall,
                    color = Slate,
                    modifier = Modifier.padding(bottom = 8.dp)
                )

                // GCash Card Selector
                Card(
                    onClick = { selectedPaymentMethod = "GCASH" },
                    border = BorderStroke(1.5.dp, if (selectedPaymentMethod == "GCASH") Ink else Line),
                    colors = CardDefaults.cardColors(containerColor = if (selectedPaymentMethod == "GCASH") BoneDeep else SurfaceWhite),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = (selectedPaymentMethod == "GCASH"),
                            onClick = { selectedPaymentMethod = "GCASH" },
                            colors = RadioButtonDefaults.colors(selectedColor = Fresh)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text("GCash e-Wallet", style = Typography.titleSmall, fontWeight = FontWeight.Bold, color = Ink)
                            Text("Pay via GCash app redirection", style = Typography.bodySmall, color = SlateSoft)
                        }
                    }
                }

                // Maya Card Selector
                Card(
                    onClick = { selectedPaymentMethod = "MAYA" },
                    border = BorderStroke(1.5.dp, if (selectedPaymentMethod == "MAYA") Ink else Line),
                    colors = CardDefaults.cardColors(containerColor = if (selectedPaymentMethod == "MAYA") BoneDeep else SurfaceWhite),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        RadioButton(
                            selected = (selectedPaymentMethod == "MAYA"),
                            onClick = { selectedPaymentMethod = "MAYA" },
                            colors = RadioButtonDefaults.colors(selectedColor = Fresh)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Column {
                            Text("Maya e-Wallet", style = Typography.titleSmall, fontWeight = FontWeight.Bold, color = Ink)
                            Text("Pay via Maya app redirection", style = Typography.bodySmall, color = SlateSoft)
                        }
                    }
                }

                // Card Selector
                Card(
                    onClick = { selectedPaymentMethod = "CARD" },
                    border = BorderStroke(1.5.dp, if (selectedPaymentMethod == "CARD") Ink else Line),
                    colors = CardDefaults.cardColors(containerColor = if (selectedPaymentMethod == "CARD") BoneDeep else SurfaceWhite),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = (selectedPaymentMethod == "CARD"),
                                onClick = { selectedPaymentMethod = "CARD" },
                                colors = RadioButtonDefaults.colors(selectedColor = Fresh)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text("Credit / Debit Card", style = Typography.titleSmall, fontWeight = FontWeight.Bold, color = Ink)
                                Text("Visa, Mastercard, JCB, etc.", style = Typography.bodySmall, color = SlateSoft)
                            }
                        }

                        if (selectedPaymentMethod == "CARD") {
                            Spacer(modifier = Modifier.height(16.dp))
                            // Card fields
                            TextField(
                                value = cardNumber,
                                onValueChange = { cardNumber = it },
                                placeholder = { Text("Card Number", color = SlateSoft) },
                                singleLine = true,
                                colors = TextFieldDefaults.colors(
                                    focusedContainerColor = Bone,
                                    unfocusedContainerColor = Bone,
                                    focusedIndicatorColor = Fresh,
                                    unfocusedIndicatorColor = Color.Transparent
                                ),
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                TextField(
                                    value = cardExpiry,
                                    onValueChange = { cardExpiry = it },
                                    placeholder = { Text("MM/YY", color = SlateSoft) },
                                    singleLine = true,
                                    colors = TextFieldDefaults.colors(
                                        focusedContainerColor = Bone,
                                        unfocusedContainerColor = Bone,
                                        focusedIndicatorColor = Fresh,
                                        unfocusedIndicatorColor = Color.Transparent
                                    ),
                                    shape = RoundedCornerShape(8.dp),
                                    modifier = Modifier.weight(1f)
                                )
                                TextField(
                                    value = cardCvv,
                                    onValueChange = { cardCvv = it },
                                    placeholder = { Text("CVV", color = SlateSoft) },
                                    singleLine = true,
                                    colors = TextFieldDefaults.colors(
                                        focusedContainerColor = Bone,
                                        unfocusedContainerColor = Bone,
                                        focusedIndicatorColor = Fresh,
                                        unfocusedIndicatorColor = Color.Transparent
                                    ),
                                    shape = RoundedCornerShape(8.dp),
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(32.dp))

                // Error Message if present
                if (checkoutState is CheckoutState.Error) {
                    Surface(
                        color = ErrorRed.copy(alpha = 0.1f),
                        border = BorderStroke(1.dp, ErrorRed),
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(bottom = 16.dp)
                    ) {
                        Text(
                            text = (checkoutState as CheckoutState.Error).message,
                            color = ErrorRed,
                            style = Typography.bodyMedium,
                            modifier = Modifier.padding(16.dp)
                        )
                    }
                }

                // Checkout Summary Details
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
                            Text("Photos Selected", style = Typography.bodyMedium, color = Slate)
                            Text("${cartItems.size} items", style = Typography.bodyMedium, fontWeight = FontWeight.Bold, color = Ink)
                        }
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Total Billing", style = Typography.titleMedium, color = Ink)
                            Text(
                                text = String.format("₱%,.2f", totalAmount),
                                style = Typography.titleLarge,
                                fontWeight = FontWeight.Bold,
                                color = Ink
                            )
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = { viewModel.checkout(emailInput, selectedPaymentMethod) },
                            enabled = emailInput.isNotBlank() && (selectedPaymentMethod != "CARD" || (cardNumber.isNotBlank() && cardExpiry.isNotBlank() && cardCvv.isNotBlank())),
                            colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("AUTHORIZE PAYMENT & PAY", style = Typography.labelMedium)
                        }
                    }
                }
            }

            // Processing Loading Overlay
            if (checkoutState is CheckoutState.Loading) {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.5f))
                        .clickable(enabled = false) {}, // Intercept clicks
                    contentAlignment = Alignment.Center
                ) {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = Bone),
                        shape = RoundedCornerShape(20.dp),
                        modifier = Modifier.width(300.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            CircularProgressIndicator(color = Fresh)
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "Securing checkout session via PayMongo...",
                                style = Typography.bodyMedium,
                                color = Ink,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                }
            }

            // Checkout Redirect URL Success Modal
            if (checkoutState is CheckoutState.Success && (checkoutState as CheckoutState.Success).order.redirectUrl != null) {
                val order = (checkoutState as CheckoutState.Success).order
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black.copy(alpha = 0.6f)),
                    contentAlignment = Alignment.Center
                ) {
                    Card(
                        colors = CardDefaults.cardColors(containerColor = Bone),
                        shape = RoundedCornerShape(24.dp),
                        modifier = Modifier
                            .fillMaxWidth(0.9f)
                            .padding(24.dp)
                    ) {
                        Column(
                            modifier = Modifier.padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = "Success",
                                tint = Fresh,
                                modifier = Modifier.size(64.dp)
                            )
                            Spacer(modifier = Modifier.height(16.dp))
                            Text(
                                text = "Order Pending Payment",
                                style = Typography.titleLarge,
                                fontWeight = FontWeight.Bold,
                                color = Ink
                            )
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = "Your PayMongo payment session has been generated. Tap below to complete GCash/Card auth.",
                                style = Typography.bodyMedium,
                                color = SlateSoft,
                                textAlign = TextAlign.Center
                            )
                            Spacer(modifier = Modifier.height(24.dp))
                            Button(
                                onClick = {
                                    order.redirectUrl?.let { uriHandler.openUri(it) }
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                                shape = RoundedCornerShape(12.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("OPEN PAYMONGO GATEWAY", style = Typography.labelMedium)
                            }
                            Spacer(modifier = Modifier.height(8.dp))
                            Button(
                                onClick = {
                                    onCheckoutSuccess()
                                    viewModel.resetCheckoutState()
                                },
                                colors = ButtonDefaults.buttonColors(containerColor = BoneDeep, contentColor = Ink),
                                shape = RoundedCornerShape(12.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("GO TO ORDER HISTORY", style = Typography.labelMedium)
                            }
                        }
                    }
                }
            }
        }
    }
}
