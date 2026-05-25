package com.quickpitik.mobile.ui.runner

import android.content.ContentValues
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.ArrowBack
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
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.foundation.shape.CircleShape
import com.quickpitik.mobile.data.local.SessionManager
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
    onNavigateToProfile: () -> Unit,
    onNavigateToSettings: () -> Unit,
    onNavigateBack: () -> Unit,
    onLogout: () -> Unit
) {
    var bibSearchQuery by remember { mutableStateOf("") }
    var activeSearchTab by remember { mutableStateOf(0) } // 0 = Selfie, 1 = Bib Number
    var selectedPhotoForDetail by remember { mutableStateOf<PhotoDto?>(null) }

    val activeEvent by viewModel.activeEvent.collectAsState()
    val searchState by viewModel.searchState.collectAsState()
    val isFiltered by viewModel.isFiltered.collectAsState()

    // Live selfie capture (camera) + gallery pick — reuse the proven ProfileScreen
    // pattern: a MediaStore URI handed to TakePicture(), then face-search the bytes.
    val context = LocalContext.current
    var pendingSelfieUri by remember { mutableStateOf<Uri?>(null) }
    val selfieCameraLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.TakePicture()
    ) { success ->
        if (success) pendingSelfieUri?.let { viewModel.searchBySelfieUri(it) }
    }
    val selfieGalleryLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { viewModel.searchBySelfieUri(it) }
    }

    // Lock the Runner Dashboard to the uniform Light Warm Cream Brand Theme
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Bone
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp)
                .verticalScroll(rememberScrollState())
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
                    BadgedBox(
                        badge = {
                            if (cartItems.isNotEmpty()) {
                                Badge(containerColor = Fresh, contentColor = Bone) {
                                    Text("${cartItems.size}")
                                }
                            }
                        }
                    ) {
                        IconButton(
                            onClick = onNavigateToCart,
                            colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                        ) {
                            Icon(Icons.Default.ShoppingCart, contentDescription = "Cart", tint = Ink)
                        }
                    }

                    var menuExpanded by remember { mutableStateOf(false) }
                    val sessionManager = remember { SessionManager.getInstance(context) }
                    val userName = sessionManager.getUserName() ?: "Runner"

                    Box {
                        Box(
                            modifier = Modifier
                                .size(40.dp)
                                .clip(CircleShape)
                                .background(Fresh)
                                .clickable { menuExpanded = true },
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = userName.take(1).uppercase(),
                                color = Bone,
                                fontWeight = FontWeight.Bold,
                                fontSize = 16.sp
                            )
                        }

                        DropdownMenu(
                            expanded = menuExpanded,
                            onDismissRequest = { menuExpanded = false },
                            modifier = Modifier.background(Bone)
                        ) {
                            DropdownMenuItem(
                                text = { Text("Profile & Selfies", color = Ink) },
                                onClick = {
                                    menuExpanded = false
                                    onNavigateToProfile()
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Account Settings", color = Ink) },
                                onClick = {
                                    menuExpanded = false
                                    onNavigateToSettings()
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Purchased Photos", color = Ink) },
                                onClick = {
                                    menuExpanded = false
                                    onNavigateToOrders()
                                }
                            )
                            Divider(color = SlateSoft)
                            DropdownMenuItem(
                                text = { Text("Sign Out", color = ErrorRed) },
                                onClick = {
                                    menuExpanded = false
                                    onLogout()
                                }
                            )
                        }
                    }
                }
            }
            Spacer(modifier = Modifier.height(24.dp))

            if (activeEvent == null) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 64.dp),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = Fresh)
                }
            } else {
                // SELECTED EVENT GALLERY VIEW
                Card(
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(containerColor = BoneDeep),
                    border = BorderStroke(1.dp, Line),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        if (!activeEvent?.bannerUrl.isNullOrEmpty()) {
                            AsyncImage(
                                model = activeEvent?.bannerUrl,
                                contentDescription = "Event Banner",
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(160.dp)
                                    .clip(RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp)),
                                contentScale = ContentScale.Crop
                            )
                        }

                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            IconButton(
                                onClick = {
                                    viewModel.clearSelectedEvent()
                                    onNavigateBack()
                                },
                                modifier = Modifier.padding(end = 8.dp)
                            ) {
                                Icon(
                                    imageVector = Icons.Default.ArrowBack,
                                    contentDescription = "Back to Events",
                                    tint = Ink
                                )
                            }
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = "SELECTED MARATHON",
                                    style = Typography.labelSmall,
                                    color = Slate
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = activeEvent?.name ?: "",
                                    style = Typography.titleMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = Ink
                                )
                                Text(
                                    text = activeEvent?.location ?: "",
                                    style = Typography.bodySmall,
                                    color = SlateSoft
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
                            // Primary: search with a saved library selfie
                            Button(
                                onClick = { viewModel.searchByStoredSelfie() },
                                shape = RoundedCornerShape(percent = 100),
                                colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text("USE STORED SELFIE", style = Typography.labelMedium)
                            }
                            Spacer(modifier = Modifier.height(8.dp))
                            Row(
                                horizontalArrangement = Arrangement.spacedBy(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                // Take a selfie now — opens the camera via a MediaStore URI
                                Button(
                                    onClick = {
                                        try {
                                            val values = ContentValues().apply {
                                                put(MediaStore.Images.Media.TITLE, "selfie_search_${System.currentTimeMillis()}")
                                                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                                            }
                                            val uri = context.contentResolver.insert(
                                                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                                                values
                                            )
                                            pendingSelfieUri = uri
                                            if (uri != null) selfieCameraLauncher.launch(uri)
                                        } catch (e: Exception) {
                                            // Swallow — camera unavailable; user can still use Upload/Stored.
                                        }
                                    },
                                    shape = RoundedCornerShape(percent = 100),
                                    colors = ButtonDefaults.buttonColors(containerColor = Ink, contentColor = Bone),
                                    modifier = Modifier.weight(1f)
                                ) {
                                    Text("TAKE SELFIE", style = Typography.labelMedium)
                                }
                                // Upload an existing photo from the device
                                Button(
                                    onClick = { selfieGalleryLauncher.launch("image/*") },
                                    shape = RoundedCornerShape(percent = 100),
                                    colors = ButtonDefaults.buttonColors(containerColor = Line, contentColor = Ink),
                                    modifier = Modifier.weight(1f)
                                ) {
                                    Text("UPLOAD", style = Typography.labelMedium)
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

                // Watermarked Photo Stream Title with Reset Button
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "MATCHED PHOTOS (WATERMARKED PREVIEW)",
                        style = Typography.labelMedium,
                        color = Slate
                    )
                    if (isFiltered || searchState is PhotosSearchState.Error) {
                        Text(
                            text = "Reset Search",
                            style = Typography.labelMedium,
                            color = Fresh,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.clickable {
                                bibSearchQuery = ""
                                viewModel.clearFilter()
                            }
                        )
                    }
                }
                Spacer(modifier = Modifier.height(12.dp))

                // Beautiful Watermarked Photo Grid
                when (val state = searchState) {
                    is PhotosSearchState.Loading -> {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 32.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(color = Fresh)
                        }
                    }
                    is PhotosSearchState.Error -> {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 32.dp),
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
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 32.dp),
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
                            state.photos.chunked(2).forEach { rowPhotos ->
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    rowPhotos.forEach { photo ->
                                        Box(
                                            modifier = Modifier
                                                .weight(1f)
                                                .aspectRatio(0.85f)
                                                .clip(RoundedCornerShape(16.dp))
                                                .background(BoneDeep)
                                                .clickable { selectedPhotoForDetail = photo },
                                            contentAlignment = Alignment.Center
                                        ) {
                                            if (photo.imageUrl != null) {
                                                AsyncImage(
                                                    model = photo.imageUrl,
                                                    contentDescription = "Photo preview",
                                                    modifier = Modifier.fillMaxSize(),
                                                    contentScale = ContentScale.Crop
                                                )
                                                Box(
                                                    modifier = Modifier
                                                        .fillMaxSize()
                                                        .background(Color.Black.copy(alpha = 0.3f))
                                                )
                                            }

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
                                                        color = if (photo.imageUrl != null) Color.White else Ink
                                                    )
                                                    Text(
                                                        text = "Time: ${photo.time}",
                                                        style = Typography.bodySmall,
                                                        color = if (photo.imageUrl != null) Color.White.copy(alpha = 0.8f) else SlateSoft
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
                                    if (rowPhotos.size == 1) {
                                        Spacer(modifier = Modifier.weight(1f))
                                    }
                                }
                                Spacer(modifier = Modifier.height(12.dp))
                            }
                        }
                    }
                    else -> {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 32.dp),
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
