package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.ShoppingCart
import androidx.compose.material3.*
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
    onLogout: () -> Unit
) {
    var bibSearchQuery by remember { mutableStateOf("") }
    var activeSearchTab by remember { mutableStateOf(0) } // 0 = Selfie, 1 = Bib Number

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
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    IconButton(
                        onClick = {},
                        colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                    ) {
                        Icon(Icons.Default.ShoppingCart, contentDescription = "Cart", tint = Ink)
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
                        Button(
                            onClick = {},
                            shape = RoundedCornerShape(percent = 100),
                            colors = ButtonDefaults.buttonColors(containerColor = Fresh, contentColor = Bone)
                        ) {
                            Text("SCAN SELFIE CAMERA", style = Typography.labelLarge)
                        }
                    }
                }
            } else {
                // Bib Entry Action Input
                TextField(
                    value = bibSearchQuery,
                    onValueChange = { bibSearchQuery = it },
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
            val mockPhotos = listOf("Photo 1", "Photo 2", "Photo 3", "Photo 4")
            LazyVerticalGrid(
                columns = GridCells.Fixed(2),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                modifier = Modifier.fillMaxWidth().weight(1f)
            ) {
                items(mockPhotos) { photo ->
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(1f)
                            .clip(RoundedCornerShape(16.dp))
                            .background(BoneDeep)
                            .clickable {},
                        contentAlignment = Alignment.Center
                    ) {
                        // Background mock runner photo texture placeholder
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(Icons.Default.Info, contentDescription = null, tint = Line, modifier = Modifier.size(32.dp))
                            Spacer(modifier = Modifier.height(4.dp))
                            Text("Runner Frame", style = Typography.bodyMedium, color = SlateSoft)
                        }

                        // Premium transparent watermark overlay (Fulfills NFR-P-2 security preview)
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .background(Color.Black.copy(alpha = 0.05f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = "QUICKPITIK\nPREVIEW",
                                color = Color.White.copy(alpha = 0.4f),
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign = TextAlign.Center,
                                letterSpacing = 2.sp
                            )
                        }
                    }
                }
            }
        }
    }
}
