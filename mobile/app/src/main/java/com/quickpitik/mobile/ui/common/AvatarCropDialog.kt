package com.quickpitik.mobile.ui.common

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.PathOperation
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.Typography
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min

/**
 * Interactive Avatar Cropper & Alignment Dialog.
 *
 * Allows users to pinch-to-zoom, pan, and visually frame their face
 * inside a circular preview before uploading. Produces a 512x512 square JPEG
 * cropped precisely to the user's circular framed selection.
 */
@Composable
fun AvatarCropDialog(
    imageUri: Uri,
    onDismiss: () -> Unit,
    onCropSuccess: (Uri) -> Unit,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var originalBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var isLoading by remember { mutableStateOf(true) }
    var isSaving by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    var containerSize by remember { mutableStateOf(IntSize.Zero) }

    // Load and rotate the image according to EXIF data
    LaunchedEffect(imageUri) {
        isLoading = true
        errorMessage = null
        withContext(Dispatchers.IO) {
            try {
                val bmp = loadOrientedBitmap(context, imageUri)
                if (bmp != null) {
                    originalBitmap = bmp
                } else {
                    errorMessage = "Failed to load the selected image."
                }
            } catch (e: Exception) {
                errorMessage = e.message ?: "An error occurred loading the photo."
            } finally {
                isLoading = false
            }
        }
    }

    Dialog(
        onDismissRequest = { if (!isSaving) onDismiss() },
        properties = DialogProperties(
            dismissOnBackPress = !isSaving,
            dismissOnClickOutside = !isSaving,
            usePlatformDefaultWidth = false
        )
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Ink.copy(alpha = 0.85f))
                .padding(horizontal = 20.dp, vertical = 24.dp),
            contentAlignment = Alignment.Center
        ) {
            Card(
                shape = QpCardShape,
                colors = CardDefaults.cardColors(containerColor = Bone),
                border = androidx.compose.foundation.BorderStroke(1.dp, Line),
                modifier = Modifier
                    .fillMaxWidth()
                    .wrapContentHeight()
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(16.dp)
                ) {
                    // Header
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Kicker("ADJUST PROFILE PHOTO")
                        Text(
                            text = "Align Your Face",
                            style = Typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )
                        Text(
                            text = "Pinch to zoom and drag to position your face inside the circle.",
                            style = Typography.bodySmall,
                            color = Slate,
                            textAlign = TextAlign.Center
                        )
                    }

                    // Content State
                    if (isLoading) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(260.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            CircularProgressIndicator(
                                color = Fresh,
                                strokeWidth = 2.dp,
                                modifier = Modifier.size(36.dp)
                            )
                        }
                    } else if (errorMessage != null || originalBitmap == null) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(200.dp),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = errorMessage ?: "Unable to load image",
                                style = Typography.bodyMedium,
                                color = ErrorRed,
                                textAlign = TextAlign.Center
                            )
                        }
                    } else {
                        val bitmap = originalBitmap!!

                        // Cropping viewport box
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(1f)
                                .clipToBounds()
                                .background(Ink)
                                .onGloballyPositioned { coordinates ->
                                    containerSize = coordinates.size
                                }
                                .pointerInput(Unit) {
                                    detectTransformGestures { _, pan, zoom, _ ->
                                        scale = (scale * zoom).coerceIn(1f, 5f)
                                        offset += pan
                                    }
                                }
                        ) {
                            // Render the transformable image
                            Image(
                                bitmap = bitmap.asImageBitmap(),
                                contentDescription = "Cropping target",
                                contentScale = ContentScale.Crop,
                                modifier = Modifier
                                    .fillMaxSize()
                                    .graphicsLayer {
                                        scaleX = scale
                                        scaleY = scale
                                        translationX = offset.x
                                        translationY = offset.y
                                    }
                            )

                            // Circular cutout stencil overlay
                            Canvas(modifier = Modifier.fillMaxSize()) {
                                val canvasWidth = size.width
                                val canvasHeight = size.height
                                val radius = min(canvasWidth, canvasHeight) / 2f
                                val center = Offset(canvasWidth / 2f, canvasHeight / 2f)

                                val outerRect = Path().apply {
                                    addRect(Rect(0f, 0f, canvasWidth, canvasHeight))
                                }
                                val circleCutout = Path().apply {
                                    addOval(Rect(center.x - radius, center.y - radius, center.x + radius, center.y + radius))
                                }

                                val scrimPath = Path.combine(
                                    operation = PathOperation.Difference,
                                    path1 = outerRect,
                                    path2 = circleCutout
                                )

                                // Darken outside the circle
                                drawPath(
                                    path = scrimPath,
                                    color = Ink.copy(alpha = 0.65f)
                                )

                                // Crisp circular framing border
                                drawCircle(
                                    color = Fresh,
                                    radius = radius,
                                    center = center,
                                    style = Stroke(width = 2.dp.toPx())
                                )

                                // Subtle center guide crosshair / eye-line guide
                                drawCircle(
                                    color = Color.White.copy(alpha = 0.25f),
                                    radius = radius * 0.5f,
                                    center = center,
                                    style = Stroke(
                                        width = 1.dp.toPx(),
                                        pathEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 12f), 0f)
                                    )
                                )
                            }
                        }

                        // Zoom control slider and reset button
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "ZOOM",
                                    style = Typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 0.5.sp),
                                    color = Slate
                                )
                                Text(
                                    text = "${"%.1f".format(scale)}x",
                                    style = Typography.labelSmall.copy(fontWeight = FontWeight.Bold),
                                    color = Ink
                                )
                            }

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Slider(
                                    value = scale,
                                    onValueChange = { scale = it },
                                    valueRange = 1f..5f,
                                    colors = SliderDefaults.colors(
                                        thumbColor = Fresh,
                                        activeTrackColor = Fresh,
                                        inactiveTrackColor = Line
                                    ),
                                    modifier = Modifier.weight(1f)
                                )

                                Spacer(Modifier.width(8.dp))

                                IconButton(
                                    onClick = {
                                        scale = 1f
                                        offset = Offset.Zero
                                    },
                                    modifier = Modifier.size(36.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Refresh,
                                        contentDescription = "Reset zoom and position",
                                        tint = Slate,
                                        modifier = Modifier.size(20.dp)
                                    )
                                }
                            }
                        }
                    }

                    // Bottom Action buttons
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        GhostCta(
                            text = "Cancel",
                            onClick = onDismiss,
                            enabled = !isSaving,
                            modifier = Modifier.weight(1f)
                        )

                        PrimaryCta(
                            text = "Save Avatar",
                            loading = isSaving,
                            enabled = !isLoading && originalBitmap != null && !isSaving,
                            onClick = {
                                val bmp = originalBitmap ?: return@PrimaryCta
                                val cSize = containerSize
                                if (cSize.width == 0 || cSize.height == 0) return@PrimaryCta

                                isSaving = true
                                scope.launch {
                                    try {
                                        val croppedUri = withContext(Dispatchers.IO) {
                                            cropBitmapToSquare(
                                                context = context,
                                                bitmap = bmp,
                                                containerWidth = cSize.width.toFloat(),
                                                containerHeight = cSize.height.toFloat(),
                                                scale = scale,
                                                offset = offset
                                            )
                                        }
                                        if (croppedUri != null) {
                                            onCropSuccess(croppedUri)
                                        } else {
                                            errorMessage = "Failed to crop image."
                                        }
                                    } catch (e: Exception) {
                                        errorMessage = e.message ?: "Failed to process cropped photo."
                                    } finally {
                                        isSaving = false
                                    }
                                }
                            },
                            modifier = Modifier.weight(1f)
                        )
                    }
                }
            }
        }
    }
}

/**
 * Loads a bitmap from URI and applies EXIF orientation transformation.
 * Downsamples exceptionally large images to prevent OutOfMemory issues.
 */
private fun loadOrientedBitmap(context: Context, uri: Uri): Bitmap? {
    val options = BitmapFactory.Options().apply {
        inJustDecodeBounds = true
    }
    context.contentResolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, options)
    }

    val maxDim = 2048
    var sampleSize = 1
    while (options.outWidth / sampleSize > maxDim || options.outHeight / sampleSize > maxDim) {
        sampleSize *= 2
    }

    val decodeOptions = BitmapFactory.Options().apply {
        inSampleSize = sampleSize
    }

    val bitmap = context.contentResolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, decodeOptions)
    } ?: return null

    // Determine orientation from EXIF
    val orientation = try {
        context.contentResolver.openInputStream(uri)?.use { stream ->
            val exif = ExifInterface(stream)
            exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } ?: ExifInterface.ORIENTATION_NORMAL
    } catch (_: Exception) {
        ExifInterface.ORIENTATION_NORMAL
    }

    val matrix = Matrix()
    when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
        ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
        ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
        ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
        else -> return bitmap
    }

    return try {
        Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    } catch (_: Exception) {
        bitmap
    }
}

/**
 * Calculates exact crop coordinates based on ContentScale.Crop and user transform (scale & offset),
 * cuts the 1:1 square region, scales it to standard 512x512 avatar size, and saves to a cache file.
 */
private fun cropBitmapToSquare(
    context: Context,
    bitmap: Bitmap,
    containerWidth: Float,
    containerHeight: Float,
    scale: Float,
    offset: Offset,
): Uri? {
    val bmpW = bitmap.width.toFloat()
    val bmpH = bitmap.height.toFloat()

    // ContentScale.Crop base scale
    val baseScale = max(containerWidth / bmpW, containerHeight / bmpH)
    val totalScale = baseScale * scale

    val dispW = bmpW * totalScale
    val dispH = bmpH * totalScale

    val centerX = (containerWidth / 2f) + offset.x
    val centerY = (containerHeight / 2f) + offset.y

    val imgLeft = centerX - (dispW / 2f)
    val imgTop = centerY - (dispH / 2f)

    // Crop circle bounds on screen = [0, 0, containerWidth, containerHeight]
    val cropRelX = (0f - imgLeft) / totalScale
    val cropRelY = (0f - imgTop) / totalScale
    val cropRelW = containerWidth / totalScale
    val cropRelH = containerHeight / totalScale

    val clampedX = cropRelX.toInt().coerceIn(0, bitmap.width - 1)
    val clampedY = cropRelY.toInt().coerceIn(0, bitmap.height - 1)
    val clampedW = cropRelW.toInt().coerceIn(1, bitmap.width - clampedX)
    val clampedH = cropRelH.toInt().coerceIn(1, bitmap.height - clampedY)

    // Ensure square crop
    val squareSize = min(clampedW, clampedH)
    if (squareSize <= 0) return null

    val cropped = Bitmap.createBitmap(bitmap, clampedX, clampedY, squareSize, squareSize)

    // Resize to high-quality standard avatar resolution (512x512)
    val finalBitmap = if (squareSize != 512) {
        Bitmap.createScaledBitmap(cropped, 512, 512, true)
    } else {
        cropped
    }

    val cacheDir = context.cacheDir
    val outputFile = File(cacheDir, "cropped_avatar_${System.currentTimeMillis()}.jpg")

    FileOutputStream(outputFile).use { out ->
        finalBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
    }

    return Uri.fromFile(outputFile)
}
