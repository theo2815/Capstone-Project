package com.quickpitik.mobile.ui.runner

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsFocusedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.selection.toggleable
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.blur
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.EventDetailDto
import com.quickpitik.mobile.data.remote.EventDto
import com.quickpitik.mobile.data.remote.PhotoDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.SelfieRefDto
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.CardLiftElevation
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.FieldShape
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.HeroText
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.InkSoft
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.LoadingSkeleton
import com.quickpitik.mobile.ui.theme.MonoFontFamily
import com.quickpitik.mobile.ui.theme.MosaicTileShape
import com.quickpitik.mobile.ui.theme.NumeralStyle
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCard
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.QuickPitikMobileTheme
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.StatusChip
import com.quickpitik.mobile.ui.theme.StatusTone
import com.quickpitik.mobile.ui.theme.SurfaceWhite
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Typography
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.Locale

// Cockpit-mode building blocks for the runner Event page — a phone-sized port of
// the website's events/[slug] CockpitMode (event-cockpit.tsx): dimmed photo
// backdrop, the "Find your photos." card with the bib / selfie panels, the
// "Browse all" link and the About strip. Everything here is stateless; the
// screen (GalleryScreen.kt) owns the mode and wires the view model.

enum class SearchPanelMode { Bib, Selfie }

/** Everything the selfie panel renders from; the screen builds it from VM state. */
data class SelfiePanelState(
    val selfies: List<SelfieRefDto>,
    val saveToLibrary: Boolean,
    /** Runner with room left in the library — hides the save checkbox otherwise. */
    val canSave: Boolean,
    val saveNotice: String?,
    /** A face search is in flight — rows disable and "Matching…" shows. */
    val matching: Boolean,
    val error: String?,
)

data class SearchPanelCallbacks(
    val onBibChange: (String) -> Unit,
    val onSubmitBib: () -> Unit,
    val onSwitchToSelfie: () -> Unit,
    val onSwitchToBib: () -> Unit,
    val onTakeSelfie: () -> Unit,
    val onUploadSelfie: () -> Unit,
    val onMatchAllSelfies: () -> Unit,
    val onPickSelfie: (SelfieRefDto) -> Unit,
    val onSaveToLibraryChange: (Boolean) -> Unit,
)

/* ─────────────── TOP ROW ─────────────── */

/** `← Back to events` + trailing actions — the website's CockpitTopBar. */
@Composable
fun EventCockpitTopRow(
    onBack: () -> Unit,
    trailing: @Composable RowScope.() -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 12.dp, end = 12.dp, top = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        BackKicker(text = "Back to events", onClick = onBack)
        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
            verticalAlignment = Alignment.CenterVertically,
            content = trailing,
        )
    }
}

/** `← Label` mono link with a real arrow glyph (Geist Mono has no reliable ←). */
@Composable
fun BackKicker(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    color: Color = Slate,
) {
    Row(
        modifier = modifier
            .heightIn(min = 48.dp)
            .clip(PillShape)
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            imageVector = Icons.Default.ArrowBack,
            contentDescription = null,
            tint = color,
            modifier = Modifier.size(16.dp),
        )
        Spacer(Modifier.width(8.dp))
        Kicker(text, color = color)
    }
}

/* ─────────────── BACKDROP ─────────────── */

/**
 * The website's DimWall: a tidy 3-column grid of 3:4 tiles behind the card,
 * each holding a real event photo softened to a texture (55% over bone-deep,
 * light blur — the blur is a no-op below API 31, the alpha carries it), with a
 * gradient fading the bottom rows into the page. Faint ink tiles when the
 * event has no photos yet.
 */
@Composable
fun DimWall(photos: List<PhotoDto>, modifier: Modifier = Modifier) {
    val pics = remember(photos) { photos.filter { it.imageUrl != null } }
    Box(modifier = modifier.clipToBounds()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            repeat(DIM_WALL_ROWS) { row ->
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    repeat(DIM_WALL_COLS) { col ->
                        val i = row * DIM_WALL_COLS + col
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .aspectRatio(3f / 4f)
                                .clip(MosaicTileShape)
                                .background(BoneDeep),
                        ) {
                            if (pics.isNotEmpty()) {
                                val pic = pics[i % pics.size]
                                AsyncImage(
                                    model = RetrofitClient.resolveImageUrl(pic.imageUrl),
                                    contentDescription = null,
                                    contentScale = ContentScale.Crop,
                                    alpha = 0.55f,
                                    modifier = Modifier
                                        .fillMaxSize()
                                        // Slight zoom keeps the blurred edge
                                        // inside the clip so tiles stay crisp.
                                        .graphicsLayer { scaleX = 1.1f; scaleY = 1.1f }
                                        .blur(1.5.dp),
                                )
                            } else {
                                Box(
                                    modifier = Modifier
                                        .fillMaxSize()
                                        .background(Ink.copy(alpha = 0.05f + ((i * 17) % 11) * 0.009f)),
                                )
                            }
                        }
                    }
                }
            }
        }
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        0f to Bone.copy(alpha = 0f),
                        0.62f to Bone.copy(alpha = if (pics.isEmpty()) 0.55f else 0.1f),
                        1f to Bone,
                    ),
                ),
        )
    }
}

private const val DIM_WALL_ROWS = 4
private const val DIM_WALL_COLS = 3

/* ─────────────── THE CARD ─────────────── */

/**
 * The lifted white card: event kicker, a two-line Anton hero (second line in
 * Fresh) and whatever panel the caller puts under it.
 */
@Composable
fun CockpitCard(
    eventName: String,
    heroLine1: String,
    heroLine2: String,
    modifier: Modifier = Modifier,
    content: @Composable ColumnScope.() -> Unit,
) {
    Surface(
        shape = QpCardShape,
        color = SurfaceWhite,
        border = BorderStroke(1.dp, Line),
        shadowElevation = CardLiftElevation,
        modifier = modifier,
    ) {
        Column(modifier = Modifier.padding(24.dp)) {
            Kicker(eventName)
            Spacer(Modifier.height(20.dp))
            HeroText(heroLine1)
            HeroText(heroLine2, color = Fresh)
            content()
        }
    }
}

/**
 * Bib ⇄ selfie panel, shared by the cockpit card and the browse-mode search
 * sheet so the two can never drift (website: BibPanel / SelfieSearchPanel).
 */
@Composable
fun SearchPanel(
    mode: SearchPanelMode,
    bib: String,
    photoCount: Int,
    selfie: SelfiePanelState,
    callbacks: SearchPanelCallbacks,
) {
    AnimatedContent(
        targetState = mode,
        transitionSpec = { fadeIn(tween(200)) togetherWith fadeOut(tween(120)) },
        label = "search-panel",
    ) { target ->
        when (target) {
            SearchPanelMode.Bib -> BibPanel(bib, photoCount, callbacks)
            SearchPanelMode.Selfie -> SelfiePanel(selfie, callbacks)
        }
    }
}

@Composable
private fun BibPanel(bib: String, photoCount: Int, cb: SearchPanelCallbacks) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Spacer(Modifier.height(28.dp))
        Kicker("Your bib number")
        Spacer(Modifier.height(4.dp))
        BibField(value = bib, onValueChange = cb.onBibChange, onSubmit = cb.onSubmitBib)
        Spacer(Modifier.height(20.dp))
        // Always enabled, like the website's submit: it is the card's one Fresh
        // accent, and submitting an empty bib simply does nothing.
        PrimaryCta(
            text = "Search by bib →",
            onClick = cb.onSubmitBib,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(24.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            HorizontalDivider(modifier = Modifier.weight(1f), color = Line)
            Kicker("or", color = SlateSoft, modifier = Modifier.padding(horizontal = 12.dp))
            HorizontalDivider(modifier = Modifier.weight(1f), color = Line)
        }
        Spacer(Modifier.height(24.dp))
        GhostCta(
            text = "Match by selfie →",
            onClick = cb.onSwitchToSelfie,
            modifier = Modifier.fillMaxWidth(),
        )
        Spacer(Modifier.height(24.dp))
        Kicker(
            text = "$photoCount ${if (photoCount == 1) "photo" else "photos"} · free to search",
            color = SlateSoft,
        )
    }
}

/** Bottom-border mono field — the website's `border-b border-line focus:border-fresh` input. */
@Composable
private fun BibField(value: String, onValueChange: (String) -> Unit, onSubmit: () -> Unit) {
    val interaction = remember { MutableInteractionSource() }
    val focused by interaction.collectIsFocusedAsState()
    val ruleColor = if (focused) Fresh else Line
    val textStyle = Typography.titleMedium.copy(
        fontFamily = MonoFontFamily,
        letterSpacing = 1.5.sp,
        color = Ink,
    )
    BasicTextField(
        value = value,
        onValueChange = onValueChange,
        singleLine = true,
        textStyle = textStyle,
        cursorBrush = SolidColor(Fresh),
        interactionSource = interaction,
        keyboardOptions = KeyboardOptions(
            capitalization = KeyboardCapitalization.Characters,
            imeAction = ImeAction.Search,
        ),
        keyboardActions = KeyboardActions(onSearch = { onSubmit() }),
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 48.dp)
            .drawBehind {
                drawLine(
                    color = ruleColor,
                    start = Offset(0f, size.height),
                    end = Offset(size.width, size.height),
                    strokeWidth = 1.dp.toPx(),
                )
            },
        decorationBox = { inner ->
            Box(contentAlignment = Alignment.CenterStart, modifier = Modifier.fillMaxWidth()) {
                if (value.isEmpty()) Text("B-4082", style = textStyle, color = SlateSoft)
                inner()
            }
        },
    )
}

@Composable
private fun SelfiePanel(state: SelfiePanelState, cb: SearchPanelCallbacks) {
    var libraryOpen by remember { mutableStateOf(false) }
    val hasLibrary = state.selfies.isNotEmpty()
    val enabled = !state.matching
    Column(modifier = Modifier.fillMaxWidth()) {
        Spacer(Modifier.height(28.dp))
        Kicker("Selfie match")
        Spacer(Modifier.height(12.dp))
        SelfieActionRow(
            label = "Take a selfie",
            caption = "Use your camera to snap one now.",
            icon = Icons.Default.Face,
            primary = !hasLibrary,
            enabled = enabled,
            onClick = cb.onTakeSelfie,
        )
        Spacer(Modifier.height(10.dp))
        SelfieActionRow(
            label = "Upload a selfie",
            caption = "Pick an existing photo from your device.",
            icon = Icons.Default.Add,
            primary = false,
            enabled = enabled,
            onClick = cb.onUploadSelfie,
        )
        if (hasLibrary) {
            val count = state.selfies.size
            Spacer(Modifier.height(10.dp))
            SelfieActionRow(
                label = "Use saved selfie",
                caption = "$count ${if (count == 1) "selfie" else "selfies"} in your library.",
                icon = Icons.Default.List,
                primary = true,
                enabled = enabled,
                onClick = { libraryOpen = !libraryOpen },
                trailingRotation = if (libraryOpen) 90f else 0f,
            )
            AnimatedVisibility(
                visible = libraryOpen,
                enter = expandVertically(tween(220)) + fadeIn(tween(220)),
                exit = shrinkVertically(tween(160)) + fadeOut(tween(120)),
            ) {
                SelfieLibraryGrid(
                    selfies = state.selfies,
                    enabled = enabled,
                    onMatchAll = { libraryOpen = false; cb.onMatchAllSelfies() },
                    onPick = { libraryOpen = false; cb.onPickSelfie(it) },
                )
            }
        }
        if (state.canSave) {
            Spacer(Modifier.height(8.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .toggleable(
                        value = state.saveToLibrary,
                        enabled = enabled,
                        role = Role.Checkbox,
                        onValueChange = cb.onSaveToLibraryChange,
                    ),
            ) {
                Checkbox(
                    checked = state.saveToLibrary,
                    onCheckedChange = null,
                    enabled = enabled,
                    colors = CheckboxDefaults.colors(checkedColor = Ink, checkmarkColor = Bone),
                )
                Text(
                    text = "Also save it to my selfie library (${state.selfies.size} of $SELFIE_MAX)",
                    style = Typography.bodySmall,
                    color = Slate,
                )
            }
        }
        state.saveNotice?.let {
            Spacer(Modifier.height(8.dp))
            Text(it, style = Typography.bodySmall, color = Slate)
        }
        state.error?.let {
            Spacer(Modifier.height(12.dp))
            Kicker("Couldn't match", color = ErrorRed)
            Spacer(Modifier.height(4.dp))
            Text(it, style = Typography.bodySmall, color = Slate)
        }
        if (state.matching) {
            Spacer(Modifier.height(16.dp))
            Kicker("Matching…", color = Fresh)
        }
        Spacer(Modifier.height(12.dp))
        BackKicker(
            text = "Use bib instead",
            onClick = cb.onSwitchToBib,
            modifier = Modifier.padding(start = 0.dp),
        )
    }
}

/** One selfie action (website SelfieActionCard): glyph disc, title, caption, arrow. */
@Composable
private fun SelfieActionRow(
    label: String,
    caption: String,
    icon: ImageVector,
    primary: Boolean,
    enabled: Boolean,
    onClick: () -> Unit,
    trailingRotation: Float = 0f,
) {
    val fg = if (primary) SurfaceWhite else Ink
    val sub = if (primary) SurfaceWhite.copy(alpha = 0.75f) else Slate
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 64.dp)
            .clip(FieldShape)
            .background(if (primary) Fresh else Bone)
            .border(1.dp, if (primary) Color.Transparent else Line, FieldShape)
            .clickable(enabled = enabled, onClick = onClick)
            .graphicsLayer { alpha = if (enabled) 1f else 0.6f }
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(CircleShape)
                .background(if (primary) SurfaceWhite.copy(alpha = 0.15f) else Color.Transparent)
                .border(1.dp, if (primary) Color.Transparent else Line, CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(icon, contentDescription = null, tint = if (primary) SurfaceWhite else Slate, modifier = Modifier.size(18.dp))
        }
        Spacer(Modifier.width(14.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(label, style = Typography.titleSmall, fontWeight = FontWeight.SemiBold, color = fg)
            Text(caption, style = Typography.bodySmall, color = sub)
        }
        Icon(
            imageVector = Icons.Default.ArrowForward,
            contentDescription = null,
            tint = fg,
            modifier = Modifier
                .size(18.dp)
                .graphicsLayer { rotationZ = trailingRotation },
        )
    }
}

/** Expanded library: "Match with all N" ink block + 3-up thumbnails; a tap searches. */
@Composable
private fun SelfieLibraryGrid(
    selfies: List<SelfieRefDto>,
    enabled: Boolean,
    onMatchAll: () -> Unit,
    onPick: (SelfieRefDto) -> Unit,
) {
    val count = selfies.size
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 10.dp)
            .clip(FieldShape)
            .background(BoneDeep)
            .border(1.dp, Line, FieldShape)
            .padding(12.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(TileShape)
                .background(Ink)
                .clickable(enabled = enabled, onClick = onMatchAll)
                .padding(horizontal = 16.dp, vertical = 12.dp),
        ) {
            Text(
                text = "Match with all $count ${if (count == 1) "selfie" else "selfies"}",
                style = Typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = Bone,
            )
            Text(
                text = "Best results — different angles find more of your photos. Or pick one below.",
                style = Typography.bodySmall,
                color = Bone.copy(alpha = 0.7f),
            )
        }
        Spacer(Modifier.height(10.dp))
        selfies.chunked(3).forEach { row ->
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(bottom = 10.dp)) {
                row.forEach { s ->
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .aspectRatio(1f)
                            .clip(TileShape)
                            .background(Ink)
                            .clickable(enabled = enabled) { onPick(s) },
                    ) {
                        AsyncImage(
                            model = RetrofitClient.resolveImageUrl(s.dataUrl),
                            contentDescription = if (s.isPrimary) "Search with primary selfie" else "Search with this selfie",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize(),
                        )
                        if (s.isPrimary) {
                            Kicker(
                                text = "Primary",
                                color = SurfaceWhite,
                                modifier = Modifier
                                    .padding(6.dp)
                                    .background(Fresh, PillShape)
                                    .padding(horizontal = 6.dp, vertical = 2.dp),
                            )
                        }
                    }
                }
                repeat(3 - row.size) { Spacer(Modifier.weight(1f)) }
            }
        }
    }
}

/* ─────────────── LINKS + STRIPS ─────────────── */

/** `Browse all N photos ↓` — the kicker button under the card. */
@Composable
fun BrowseAllLink(label: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .heightIn(min = 48.dp)
            .clip(PillShape)
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Kicker(label, color = Slate)
        Spacer(Modifier.width(6.dp))
        Icon(
            imageVector = Icons.Default.KeyboardArrowDown,
            contentDescription = null,
            tint = Slate,
            modifier = Modifier.size(16.dp),
        )
    }
}

/**
 * The website's AboutStrip: bone-deep band with organizer, date, description,
 * categories and pricing. Renders the list DTO immediately and fills the
 * editorial fields in when the detail fetch lands.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun AboutStrip(
    event: EventDto,
    detail: EventDetailDto?,
    onRefundPolicy: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .background(BoneDeep)
            .drawBehind {
                val w = 1.dp.toPx()
                drawLine(Line, Offset(0f, 0f), Offset(size.width, 0f), w)
                drawLine(Line, Offset(0f, size.height), Offset(size.width, size.height), w)
            }
            .padding(horizontal = 24.dp, vertical = 40.dp),
    ) {
        Kicker("About this race")
        Spacer(Modifier.height(12.dp))
        Text("Race day notes.", style = Typography.displayMedium, color = Ink)
        Spacer(Modifier.height(16.dp))
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (!detail?.organizerName.isNullOrBlank()) {
                Kicker("Organizer · ${detail?.organizerName}", color = SlateSoft)
            }
            Kicker("${eventDateLabel(event.date)} · ${event.location}", color = SlateSoft)
            Kicker("${"%,d".format(event.photoCount)} photos", color = SlateSoft)
        }
        Spacer(Modifier.height(24.dp))
        if (detail == null) {
            LoadingSkeleton(shape = TileShape, modifier = Modifier.fillMaxWidth(0.9f).height(14.dp))
            Spacer(Modifier.height(8.dp))
            LoadingSkeleton(shape = TileShape, modifier = Modifier.fillMaxWidth(0.6f).height(14.dp))
        } else {
            if (!detail.description.isNullOrBlank()) {
                Text(detail.description, style = Typography.bodyLarge, color = InkSoft)
                Spacer(Modifier.height(20.dp))
            }
            if (detail.categories.isNotEmpty()) {
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    detail.categories.forEach { StatusChip(text = it, tone = StatusTone.Neutral) }
                }
                Spacer(Modifier.height(20.dp))
            }
            HorizontalDivider(color = Line)
            Spacer(Modifier.height(20.dp))
            Kicker("Pricing")
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.Bottom) {
                Text(
                    text = "₱${"%,.0f".format(detail.pricePerPhoto)}",
                    style = NumeralStyle.copy(fontSize = 40.sp, lineHeight = 44.sp),
                    color = Fresh,
                )
                Spacer(Modifier.width(12.dp))
                Kicker("per photo", modifier = Modifier.padding(bottom = 8.dp))
            }
            if (detail.bundlePrice != null && detail.bundleSize != null) {
                Spacer(Modifier.height(4.dp))
                Kicker(
                    "or ₱${"%,.0f".format(detail.bundlePrice)} for ${detail.bundleSize}",
                    color = Ink,
                )
            }
            Spacer(Modifier.height(10.dp))
            Text(
                "Watermarked previews are free. Pay once, download forever.",
                style = Typography.bodySmall,
                color = SlateSoft,
            )
        }
        Spacer(Modifier.height(12.dp))
        Row(
            modifier = Modifier
                .heightIn(min = 48.dp)
                .clip(PillShape)
                .clickable(onClick = onRefundPolicy),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            ArrowLabel("Refund policy →", color = Slate)
        }
    }
}

/* ─────────────── EMPTY + UPCOMING ─────────────── */

/** photoCount == 0: nothing to search yet (website EmptyCockpit). */
@Composable
fun EmptyCockpitCard(eventName: String, modifier: Modifier = Modifier) {
    CockpitCard(
        eventName = eventName,
        heroLine1 = "Photos aren't",
        heroLine2 = "ready yet.",
        modifier = modifier,
    ) {
        Spacer(Modifier.height(20.dp))
        Text(
            text = "Photographers have a few days from race day to upload. Get notified the moment your shots land.",
            style = Typography.bodyLarge,
            color = InkSoft,
        )
    }
}

// Pre-race-day stand-in for the search cockpit. Faithful port of the website's
// UpcomingEventNotice (events/[slug]/page.tsx): 16:9 cover, Fresh "OPENS ·
// [date]" kicker, name, city, venue, and the race-day + four-day-window copy.
// The runner sees why there's nothing to search yet instead of an empty grid.
@Composable
internal fun UpcomingEventNotice(
    event: EventDto,
    onBack: () -> Unit,
) {
    val dateLabel = remember(event.date) { formatUpcomingDate(event.date) }
    val cityUpper = remember(event.location) {
        event.location.substringAfterLast(',').trim().uppercase()
    }

    Column(modifier = Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = onBack) {
                Icon(
                    imageVector = Icons.Default.ArrowBack,
                    contentDescription = "Back to Events",
                    tint = Ink,
                )
            }
            Text(
                text = "ALL EVENTS",
                style = Typography.labelMedium,
                color = Slate,
            )
        }
        Spacer(modifier = Modifier.height(12.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(16f / 9f)
                .clip(QpCardShape)
                .background(Ink),
            contentAlignment = Alignment.Center,
        ) {
            if (!event.bannerUrl.isNullOrEmpty()) {
                AsyncImage(
                    model = RetrofitClient.resolveImageUrl(event.bannerUrl),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop,
                )
            } else {
                Text(
                    text = event.name,
                    style = Typography.titleLarge,
                    color = Bone.copy(alpha = 0.25f),
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 24.dp),
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
        // The one Fresh element in this viewport — the notice has no CTA, so the
        // date kicker carries the accent, as it does on the web.
        Kicker(text = "Opens · $dateLabel", color = Fresh)
        Spacer(modifier = Modifier.height(12.dp))
        // displayMedium, not the Anton hero style: the event name is
        // user-generated text and uppercasing it in a condensed display face
        // reads wrong (flagged during the Finish Line migration).
        Text(
            text = event.name,
            style = Typography.displayMedium,
            color = Ink,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Kicker(text = cityUpper, color = Slate)
        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = event.location,
            style = Typography.bodyMedium,
            color = InkSoft,
        )
        Spacer(modifier = Modifier.height(24.dp))
        Text(
            text = "The gallery and runner search open on race day. " +
                "Photographers have a four-day window from race day to upload — " +
                "check back then to find your photos.",
            style = Typography.bodyMedium,
            color = InkSoft,
        )
    }
}

// "Saturday, October 3, 2026" — matches the website's toLocaleDateString with
// weekday/month/day/year. Falls back to the raw ISO date if it can't parse.
private fun formatUpcomingDate(iso: String): String = try {
    LocalDate.parse(iso).format(
        DateTimeFormatter.ofPattern("EEEE, MMMM d, yyyy", Locale.US)
    )
} catch (e: Exception) {
    iso
}

// Runner opt-in card — "Get notified when your photos are ready". Mirrors the
// website's PhotoAlertToggle. GhostCta (not PrimaryCta) keeps the single Fresh
// accent for the page's real highlight; the registered state uses a
// SuccessGreen StatusChip, a distinct token from the Fresh CTA.
@Composable
internal fun PhotoAlertCard(
    state: PhotoAlertUiState,
    onToggle: (Boolean) -> Unit,
    onAddSelfie: () -> Unit,
) {
    when (state) {
        is PhotoAlertUiState.Loading -> {
            LoadingSkeleton(
                shape = QpCardShape,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(96.dp),
            )
        }
        is PhotoAlertUiState.NeedsSelfie -> {
            QpCard(modifier = Modifier.fillMaxWidth()) {
                Kicker("Photo alerts", color = Slate)
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Get notified when your photos are ready",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = "Add a selfie and we'll email you the moment we spot you.",
                    style = Typography.bodyMedium,
                    color = SlateSoft,
                )
                if (state.message != null) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = state.message,
                        style = Typography.bodySmall,
                        color = ErrorRed,
                    )
                }
                Spacer(Modifier.height(16.dp))
                // Adds in place (sheet on this screen) — the web's
                // mutation.isPending pattern while the upload runs.
                GhostCta(
                    text = if (state.uploading) "Uploading…" else "Add a selfie →",
                    onClick = onAddSelfie,
                    enabled = !state.uploading,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
        }
        is PhotoAlertUiState.Ready -> {
            QpCard(modifier = Modifier.fillMaxWidth()) {
                Kicker("Photo alerts", color = Slate)
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "Get notified when your photos are ready",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = if (state.registered)
                        "You're on the list — we'll email you when your photos land."
                    else
                        "We'll email you the moment your photos land.",
                    style = Typography.bodyMedium,
                    color = SlateSoft,
                )
                if (state.message != null) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        text = state.message,
                        style = Typography.bodySmall,
                        color = ErrorRed,
                    )
                }
                Spacer(Modifier.height(16.dp))
                if (state.registered) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        StatusChip(text = "Notifications on", tone = StatusTone.Approved)
                        Spacer(Modifier.weight(1f))
                        Text(
                            text = "Turn off",
                            style = Typography.labelMedium,
                            color = Slate,
                            modifier = Modifier
                                .clip(PillShape)
                                .clickable(enabled = !state.updating) { onToggle(false) }
                                .padding(horizontal = 10.dp, vertical = 8.dp),
                        )
                    }
                } else {
                    GhostCta(
                        text = "Notify me when ready",
                        onClick = { onToggle(true) },
                        enabled = !state.updating,
                        modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }
    }
}

/* ─────────────── PREVIEWS ─────────────── */

private val previewEvent = EventDto(
    id = "e1",
    slug = "cebu-city-marathon-2026",
    name = "Cebu City Marathon 2026",
    date = "2026-09-02",
    location = "Cebu Business Park, Cebu City",
    bannerUrl = null,
    photoCount = 244,
    participantCount = 0,
    status = "ACTIVE",
)

private val previewCallbacks = SearchPanelCallbacks(
    onBibChange = {}, onSubmitBib = {}, onSwitchToSelfie = {}, onSwitchToBib = {},
    onTakeSelfie = {}, onUploadSelfie = {}, onMatchAllSelfies = {}, onPickSelfie = {},
    onSaveToLibraryChange = {},
)

@Preview(showBackground = true)
@Composable
private fun CockpitCardBibPreview() {
    QuickPitikMobileTheme {
        Box(Modifier.background(Bone).padding(24.dp)) {
            CockpitCard(previewEvent.name, "Find your", "photos.") {
                SearchPanel(
                    mode = SearchPanelMode.Bib,
                    bib = "",
                    photoCount = 244,
                    selfie = SelfiePanelState(emptyList(), true, false, null, false, null),
                    callbacks = previewCallbacks,
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun CockpitCardSelfiePreview() {
    val selfies = listOf(
        SelfieRefDto(id = "s1", dataUrl = "", uploadedAt = "", isPrimary = true, qualityScore = 0.0),
        SelfieRefDto(id = "s2", dataUrl = "", uploadedAt = "", isPrimary = false, qualityScore = 0.0),
    )
    QuickPitikMobileTheme {
        Box(Modifier.background(Bone).padding(24.dp)) {
            CockpitCard(previewEvent.name, "Find your", "photos.") {
                SearchPanel(
                    mode = SearchPanelMode.Selfie,
                    bib = "",
                    photoCount = 244,
                    selfie = SelfiePanelState(selfies, true, true, null, false, "We didn't find your face in this event yet. Try another shot."),
                    callbacks = previewCallbacks,
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun PhotoAlertCardNeedsSelfieErrorPreview() {
    QuickPitikMobileTheme {
        Box(Modifier.background(Bone).padding(24.dp)) {
            PhotoAlertCard(
                state = PhotoAlertUiState.NeedsSelfie(message = "No face found — try a clearer, frontal shot."),
                onToggle = {},
                onAddSelfie = {},
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun AboutStripPreview() {
    QuickPitikMobileTheme {
        AboutStrip(
            event = previewEvent,
            detail = EventDetailDto(
                id = "e1", slug = previewEvent.slug, name = previewEvent.name,
                date = previewEvent.date, location = previewEvent.location,
                photoCount = 244, organizerName = "Cebu Runners Club",
                description = "Start at 4:00 AM from Cebu Business Park. Photographers cover km 5, km 10 and the finish.",
                categories = listOf("42K", "21K", "10K"), pricePerPhoto = 150.0,
            ),
            onRefundPolicy = {},
        )
    }
}
