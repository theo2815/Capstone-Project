package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.*
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PhotoDto
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RunnerGalleryScreen(
    viewModel: RunnerGalleryViewModel,
    cartViewModel: CartViewModel,
    onNavigateToCart: () -> Unit,
    onNavigateToOrders: () -> Unit,
    onLogout: () -> Unit
) {
    var bibSearchQuery by remember { mutableStateOf("") }
    var activeSearchTab by remember { mutableStateOf(0) } // 0 = Selfie, 1 = Bib Number
    var showEventDropdown by remember { mutableStateOf(false) }
    var selectedPhotoForDetail by remember { mutableStateOf<PhotoDto?>(null) }

    val eventsState by viewModel.eventsState.collectAsState()
    val activeEvent by viewModel.activeEvent.collectAsState()
    val searchState by viewModel.searchState.collectAsState()

    // Lock the Runner Dashboard to the uniform Light Warm Cream Brand Theme
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
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "GALLERY HUB",
                        style = Typography.labelMedium,
                        color = Slate
                    )
                    Text(
                        text = "QuickPitik",
                        style = Typography.titleLarge,
                        fontWeight = FontWeight.Bold,
                        color = Ink
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    val cartItems by cartViewModel.cartItems.collectAsState()
                    IconButton(
                        onClick = onNavigateToCart,
                        colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                    ) {
                        BadgedBox(
                            badge = {
                                if (cartItems.isNotEmpty()) {
                                    Badge(containerColor = Fresh, contentColor = Bone) {
                                        Text("${cartItems.size}")
                                    }
                                }
                            }
                        ) {
                            Icon(Icons.Default.ShoppingCart, contentDescription = "Cart", tint = Ink)
                        }
                    }
                    IconButton(
                        onClick = onNavigateToOrders,
                        colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                    ) {
                        Icon(Icons.Default.List, contentDescription = "Orders", tint = Ink)
                    }
                    Button(
                        onClick = onLogout,
                        colors = ButtonDefaults.buttonColors(containerColor = BoneDeep, contentColor = Ink),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text("LEAVE", style = Typography.labelMedium)
                    }
                }
            }
            Spacer(modifier = Modifier.height(24.dp))

            // Active Event Dropdown Card (Matches the Photographer Dashboard)
            Card(
                onClick = { showEventDropdown = true },
                shape = RoundedCornerShape(16.dp),
                colors = CardDefaults.cardColors(containerColor = BoneDeep),
                border = BorderStroke(1.dp, Line),
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "SELECTED MARATHON",
                            style = Typography.labelSmall,
                            color = Slate
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = activeEvent?.name ?: "Loading Events...",
                            style = Typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )
                        Text(
                            text = activeEvent?.location ?: "Please select an event",
                            style = Typography.bodySmall,
                            color = SlateSoft
                        )
                    }
                    Icon(
                        imageVector = Icons.Default.ArrowDropDown,
                        contentDescription = "Expand",
                        tint = Ink
                    )
                }

                DropdownMenu(
                    expanded = showEventDropdown,
                    onDismissRequest = { showEventDropdown = false },
                    modifier = Modifier.background(BoneDeep)
                ) {
                    when (val state = eventsState) {
                        is RunnerEventsState.Success -> {
                            state.events.forEach { event ->
                                DropdownMenuItem(
                                    text = { Text(event.name, color = Ink, fontWeight = FontWeight.Bold) },
                                    onClick = {
                                        viewModel.selectEvent(event)
                                        showEventDropdown = false
                                    }
                                )
                            }
                        }
                        is RunnerEventsState.Error -> {
                            DropdownMenuItem(
                                text = { Text(state.message, color = Color.Red) },
                                onClick = { showEventDropdown = false }
                            )
                        }
                        else -> {
                            DropdownMenuItem(
                                text = { Text("Fetching active marathons...", color = Slate) },
                                onClick = { showEventDropdown = false }
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(20.dp))

            // AI Search Selector Cards (Selfie vs Bib)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Selfie Match Card
                Card(
                    onClick = { activeSearchTab = 0 },
                    border = BorderStroke(
                        width = 1.5.dp,
                        color = if (activeSearchTab == 0) Ink else Line
                    ),
                    colors = CardDefaults.cardColors(
                        containerColor = if (activeSearchTab == 0) BoneDeep else Bone
                    ),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Icon(Icons.Default.Face, contentDescription = "Selfie", tint = Fresh)
                        Spacer(modifier = Modifier.height(12.dp))
                        Text("Selfie Match", style = Typography.titleMedium, color = Ink)
                        Text("AI Face Search", style = Typography.bodyMedium, color = SlateSoft)
                    }
                }

                // Bib Number Search Card
                Card(
                    onClick = { activeSearchTab = 1 },
                    border = BorderStroke(
                        width = 1.5.dp,
                        color = if (activeSearchTab == 1) Ink else Line
                    ),
                    colors = CardDefaults.cardColors(
                        containerColor = if (activeSearchTab == 1) BoneDeep else Bone
                    ),
                    shape = RoundedCornerShape(16.dp),
                    modifier = Modifier.weight(1f)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Icon(Icons.Default.Search, contentDescription = "Bib", tint = Fresh)
                        Spacer(modifier = Modifier.height(12.dp))
                        Text("Bib Lookup", style = Typography.titleMedium, color = Ink)
                        Text("Search by Number", style = Typography.bodyMedium, color = SlateSoft)
                    }
                }
            }
            Spacer(modifier = Modifier.height(20.dp))

            // Search Action Interface
            if (activeSearchTab == 0) {
                // Selfie Upload Action Trigger
                Card(
                    border = BorderStroke(1.dp, Line),
                    colors = CardDefaults.cardColors(containerColor = BoneDeep),
                    shape = RoundedCornerShape(20.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(24.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Find photos of yourself instantly using our high-speed face recognition model.",
                            textAlign = TextAlign.Center,
                            style = Typography.bodyMedium,
                            color = InkSoft
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Button(
                                onClick = {},
                                shape = RoundedCornerShape(percent = 100),
                                colors = ButtonDefaults.buttonColors(containerColor = Line, contentColor = Ink),
                                modifier = Modifier.weight(1f)
                            ) {
                                Text("CAMERA SCAN", style = Typography.labelMedium)
                            }
                            Button(
                                onClick = { viewModel.simulateSelfieSearch() },
                                shape = RoundedCornerShape(percent = 100),
                                colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                                modifier = Modifier.weight(1.2f)
                            ) {
                                Text("⚡ SIMULATE MATCH", style = Typography.labelMedium)
                            }
                        }
                    }
                }
            } else {
                // Bib Entry Action Input
                TextField(
                    value = bibSearchQuery,
                    onValueChange = {
                        bibSearchQuery = it
                        viewModel.searchByBib(it)
                    },
                    placeholder = { Text("Enter bib number (e.g. 2948)", color = SlateSoft) },
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
            }
            Spacer(modifier = Modifier.height(24.dp))

            // Watermarked Photo Stream Title
            Text(
                text = "MATCHED PHOTOS (WATERMARKED PREVIEW)",
                style = Typography.labelMedium,
                color = Slate
            )
            Spacer(modifier = Modifier.height(12.dp))

            // Beautiful Watermarked Photo Grid
            when (val state = searchState) {
                is PhotosSearchState.Loading -> {
                    Box(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        contentAlignment = Alignment.Center
                    ) {
                        CircularProgressIndicator(color = Fresh)
                    }
                }
                is PhotosSearchState.Error -> {
                    Box(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = state.message,
                            color = Color.Red,
                            textAlign = TextAlign.Center,
                            style = Typography.bodyMedium
                        )
                    }
                }
                is PhotosSearchState.Success -> {
                    if (state.photos.isEmpty()) {
                        Box(
                            modifier = Modifier.fillMaxWidth().weight(1f),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "No matched photos found for this event.\nTry another search parameter or selfie scan!",
                                color = SlateSoft,
                                textAlign = TextAlign.Center,
                                style = Typography.bodyMedium
                            )
                        }
                    } else {
                        LazyVerticalGrid(
                            columns = GridCells.Fixed(2),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp),
                            modifier = Modifier.fillMaxWidth().weight(1f)
                        ) {
                            items(state.photos) { photo ->
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .aspectRatio(0.85f)
                                        .clip(RoundedCornerShape(16.dp))
                                        .background(BoneDeep)
                                        .clickable { selectedPhotoForDetail = photo },
                                    contentAlignment = Alignment.Center
                                ) {
                                    // Background mock runner photo texture details
                                    Column(
                                        modifier = Modifier.fillMaxSize().padding(12.dp),
                                        verticalArrangement = Arrangement.SpaceBetween
                                    ) {
                                        // Header Tag (KM check)
                                        Surface(
                                            shape = RoundedCornerShape(8.dp),
                                            color = Fresh.copy(alpha = 0.15f),
                                            modifier = Modifier.padding(4.dp)
                                        ) {
                                            Text(
                                                text = "KM ${photo.km ?: 0}",
                                                color = Fresh,
                                                fontSize = 10.sp,
                                                fontWeight = FontWeight.Bold,
                                                modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                                            )
                                        }

                                        // Footer metadata
                                        Column {
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
                                    }

                                    // Premium transparent watermark overlay (Fulfills NFR-P-2 security preview)
                                    Box(
                                        modifier = Modifier
                                            .fillMaxSize()
                                            .background(Color.Black.copy(alpha = 0.04f)),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text(
                                            text = "QUICKPITIK\nPREVIEW",
                                            color = Color.White.copy(alpha = 0.35f),
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold,
                                            textAlign = TextAlign.Center,
                                            letterSpacing = 1.5.sp
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
                else -> {
                    Box(
                        modifier = Modifier.fillMaxWidth().weight(1f),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "Please select a marathon and run a search to fetch premium matched photos.",
                            color = SlateSoft,
                            textAlign = TextAlign.Center,
                            style = Typography.bodyMedium
                        )
                    }
                }
            }
        }
    }

    if (selectedPhotoForDetail != null) {
        val photo = selectedPhotoForDetail!!
        val cartItems by cartViewModel.cartItems.collectAsState()
        val isInCart = cartItems.any { it.photoId == photo.id }
        
        AlertDialog(
            onDismissRequest = { selectedPhotoForDetail = null },
            title = { Text("Photo Details", fontWeight = FontWeight.Bold, color = Ink) },
            text = {
                Column(modifier = Modifier.fillMaxWidth()) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(180.dp)
                            .clip(RoundedCornerShape(12.dp))
                            .background(BoneDeep),
                        contentAlignment = Alignment.Center
                    ) {
                        if (photo.imageUrl != null) {
                            AsyncImage(
                                model = photo.imageUrl,
                                contentDescription = "Photo preview",
                                modifier = Modifier.fillMaxSize()
                            )
                        } else {
                            Text("QUICKPITIK PREVIEW", fontWeight = FontWeight.Bold, color = SlateSoft)
                        }
                        
                        // Watermark overlay
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color.Black.copy(alpha = 0.04f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "QUICKPITIK\nPREVIEW",
                                color = Color.White.copy(alpha = 0.35f),
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign = TextAlign.Center
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    Text("Bib Number: ${photo.bib ?: "N/A"}", style = Typography.bodyMedium, color = Ink)
                    Text("Timestamp: ${photo.time}", style = Typography.bodyMedium, color = Ink)
                    Text("Distance Mark: KM ${photo.km ?: 0}", style = Typography.bodyMedium, color = Ink)
                    Spacer(modifier = Modifier.height(12.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("Price:", style = Typography.titleMedium, color = Slate)
                        Text(
                            text = String.format("₱%,.2f", photo.price),
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Fresh
                        )
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        val event = activeEvent
                        if (event != null) {
                            if (isInCart) {
                                cartViewModel.removeFromCart(photo.id)
                            } else {
                                cartViewModel.addToCart(photo, event.id, event.slug, event.name)
                            }
                        }
                        selectedPhotoForDetail = null
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = if (isInCart) ErrorRed else Fresh, contentColor = Bone),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    Text(if (isInCart) "REMOVE FROM CART" else "ADD TO CART")
                }
            },
            dismissButton = {
                TextButton(onClick = { selectedPhotoForDetail = null }) {
                    Text("CLOSE", color = Ink)
                }
            },
            containerColor = Bone,
            shape = RoundedCornerShape(16.dp)
        )
    }
}
