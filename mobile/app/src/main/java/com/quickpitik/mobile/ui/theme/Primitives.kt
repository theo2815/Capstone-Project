package com.quickpitik.mobile.ui.theme

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.R

// ── Finish Line shared primitives ───────────────────────────────────────────
// Encode the website's hard rules ONCE so screens stop hand-rolling styles:
//   • mono UPPERCASE kickers   • single-accent (Fresh #1B7A46) pill CTAs
//   • Anton UPPERCASE heroes (HeroText — the ONLY non-kicker uppercase)
//   • ghost (ink-outline) CTAs • 16dp Bone-deep cards • tabular stat numerals
// New depth tokens FreshTint / Pine / LineStrong are available for washes and
// stronger borders. All reference the `Typography`/token vals directly (not
// MaterialTheme.*) so they stay font- and color-correct even inside a nested
// MaterialTheme override.

/** Pill radius for CTAs/chips (website `rounded-full`). */
val PillShape = RoundedCornerShape(100)

/** Card radius (website `rounded-2xl` = 16px). */
val QpCardShape = RoundedCornerShape(16.dp)

/** Official camera-mark + wordmark lockup from the shared QuickPitik asset. */
@Composable
fun BrandLogo(
    modifier: Modifier = Modifier,
    compact: Boolean = false,
    contentDescription: String? = "QuickPitik",
) {
    Image(
        painter = painterResource(R.drawable.quickpitik_logo),
        contentDescription = contentDescription,
        contentScale = ContentScale.Fit,
        modifier = modifier
            .width(if (compact) 112.dp else 144.dp)
            .height(if (compact) 28.dp else 36.dp),
    )
}

/**
 * Anton hero headline — the website's `.font-hero` (Finish Line). UPPERCASES
 * internally because Compose has no text-transform; this and Kicker are the
 * only uppercase text. Marketing/auth/discovery surfaces only — Anton never
 * appears in the studio (website rule: no Anton in dashboards). Anton is
 * single-weight; displayLarge already pins FontWeight.Normal.
 */
@Composable
fun HeroText(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = Ink,
) {
    Text(
        text = text.uppercase(),
        style = Typography.displayLarge,
        color = color,
        modifier = modifier,
    )
}

/** Mono, UPPERCASE eyebrow. Kickers + HeroText are the only uppercase text. */
@Composable
fun Kicker(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = Slate,
) {
    Text(
        text = text.uppercase(),
        style = Typography.labelMedium,
        color = color,
        modifier = modifier,
    )
}

/**
 * CTA label content. Funnel Sans (our downloadable body font) carries no U+2192
 * glyph, so a literal "→" in a label renders as a broken fallback ("ʼn") on some
 * devices. Labels ending in "→" render the arrow as a vector icon instead —
 * crisp on every device, RTL-aware — and the character is stripped from the
 * visible text. The icon takes no explicit tint so it inherits the button's
 * content color (tracking enabled/disabled and primary/ghost automatically).
 */
@Composable
private fun CtaContent(text: String, style: TextStyle) {
    val trimmed = text.trimEnd()
    val hasArrow = trimmed.endsWith("→")
    val label = if (hasArrow) trimmed.removeSuffix("→").trimEnd() else text
    Row(verticalAlignment = Alignment.CenterVertically) {
        // Single line, ellipsize — a CTA in a narrow weight() column must never
        // wrap to two lines (it gets cramped in the fixed 48dp height). This one
        // guard covers every PrimaryCta / GhostCta app-wide.
        Text(
            label,
            style = style,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis,
        )
        if (hasArrow) {
            Spacer(Modifier.width(6.dp))
            Icon(
                imageVector = Icons.Default.ArrowForward,
                contentDescription = null,
                modifier = Modifier.size(18.dp),
            )
        }
    }
}

/**
 * Primary call-to-action. The single `fresh` accent — use at most one per
 * viewport. Funnel-semibold label on a fresh pill.
 */
@Composable
fun PrimaryCta(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    loading: Boolean = false,
) {
    Button(
        onClick = onClick,
        enabled = enabled && !loading,
        shape = PillShape,
        colors = ButtonDefaults.buttonColors(
            containerColor = Fresh,
            contentColor = Color.White,
            disabledContainerColor = Line,
            disabledContentColor = SlateSoft,
        ),
        modifier = modifier.height(48.dp),
    ) {
        if (loading) {
            CircularProgressIndicator(
                color = Color.White,
                strokeWidth = 2.dp,
                modifier = Modifier.size(20.dp),
            )
        } else {
            CtaContent(text, Typography.bodyMedium)
        }
    }
}

/** Secondary CTA — ink outline, no fill. Keeps the single accent for PrimaryCta. */
@Composable
fun GhostCta(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        shape = PillShape,
        border = BorderStroke(1.dp, Ink),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = Ink),
        modifier = modifier.height(48.dp),
    ) {
        CtaContent(text, Typography.bodyMedium)
    }
}

/**
 * Inline directional label — the non-button counterpart to the CTA arrow
 * handling. Renders [text] (minus any trailing "→") followed by a vector
 * ArrowForward tinted to [color], so links like "Open →" / "VIEW GALLERY →"
 * read the same on every device without relying on the body font's missing
 * U+2192 glyph. Text without a trailing arrow renders unchanged (no icon).
 */
@Composable
fun ArrowLabel(
    text: String,
    color: Color,
    modifier: Modifier = Modifier,
    style: TextStyle = Typography.labelMedium,
    fontWeight: FontWeight? = null,
    fontSize: TextUnit = TextUnit.Unspecified,
    iconSize: Dp = 14.dp,
) {
    val trimmed = text.trimEnd()
    val hasArrow = trimmed.endsWith("→")
    val label = if (hasArrow) trimmed.removeSuffix("→").trimEnd() else text
    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        Text(label, style = style, color = color, fontWeight = fontWeight, fontSize = fontSize)
        if (hasArrow) {
            Spacer(Modifier.width(4.dp))
            Icon(
                imageVector = Icons.Default.ArrowForward,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(iconSize),
            )
        }
    }
}

/** Standard surface card: bone-deep fill, 1dp line border, 16dp radius. */
@Composable
fun QpCard(
    modifier: Modifier = Modifier,
    padding: Dp = 16.dp,
    content: @Composable ColumnScope.() -> Unit,
) {
    Surface(
        shape = QpCardShape,
        color = BoneDeep,
        border = BorderStroke(1.dp, Line),
        modifier = modifier,
    ) {
        Column(modifier = Modifier.padding(padding), content = content)
    }
}

/** Big tabular numeral over a mono kicker label — the Quiet Studio stat unit. */
@Composable
fun StatNumber(
    value: String,
    label: String,
    modifier: Modifier = Modifier,
    valueColor: Color = Ink,
    valueSize: TextUnit = 22.sp,
) {
    Column(modifier = modifier) {
        Text(
            text = value,
            style = NumeralStyle.copy(fontSize = valueSize),
            color = valueColor,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Kicker(label, color = SlateSoft)
    }
}

/**
 * Skeleton placeholder block — a soft BoneDeep rectangle whose alpha breathes
 * between 0.55 → 1.0 every ~1.1s. Use it to stand in for the final layout
 * while data loads (one Box per atom: tile, line, avatar). Quiet Studio rule:
 * skeleton matches the final layout, not a centered spinner.
 */
@Composable
fun LoadingSkeleton(
    modifier: Modifier = Modifier,
    shape: Shape = QpCardShape,
) {
    val transition = rememberInfiniteTransition(label = "qp-skeleton")
    val alpha by transition.animateFloat(
        initialValue = 0.55f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1100, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "qp-skeleton-alpha",
    )
    Box(
        modifier = modifier.background(color = BoneDeep.copy(alpha = alpha), shape = shape),
    )
}

/**
 * Standard error surface — ErrorRed kicker + plain-English body + optional
 * retry PrimaryCta. Used in place of bare red Text wherever a fetch can fail.
 * Pass `onRetry = null` for terminal errors with no action.
 */
@Composable
fun ErrorView(
    message: String,
    modifier: Modifier = Modifier,
    title: String = "Something went wrong",
    onRetry: (() -> Unit)? = null,
    retryLabel: String = "Try again",
) {
    Column(
        modifier = modifier.padding(horizontal = 24.dp, vertical = 32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Kicker(title, color = ErrorRed)
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = message,
            style = Typography.bodyMedium,
            color = Slate,
            textAlign = TextAlign.Center,
        )
        if (onRetry != null) {
            Spacer(modifier = Modifier.height(20.dp))
            PrimaryCta(text = retryLabel, onClick = onRetry)
        }
    }
}

/** Status-chip tone — picks the semantic color. */
enum class StatusTone { Approved, Warning, Danger, Neutral }

/**
 * Compact uppercase status pill. One shape (BadgeShape), four tones. Replaces
 * the hand-rolled "VERIFIED / PENDING / CHANGES REQUIRED / INFO" banners that
 * were duplicated across photographer screens. Foreground = semantic color;
 * background = 10% alpha of the same (Neutral uses Line for contrast).
 */
@Composable
fun StatusChip(
    text: String,
    tone: StatusTone,
    modifier: Modifier = Modifier,
    leadingIcon: ImageVector? = null,
) {
    val fg = when (tone) {
        StatusTone.Approved -> SuccessGreen
        StatusTone.Warning -> WarningOrange
        StatusTone.Danger -> ErrorRed
        StatusTone.Neutral -> Slate
    }
    // Approved uses the Finish Line green wash token (the website's exact
    // chip background) instead of a computed alpha of the darker green.
    val bg = when (tone) {
        StatusTone.Neutral -> Line
        StatusTone.Approved -> FreshTint
        else -> fg.copy(alpha = 0.10f)
    }
    Row(
        modifier = modifier
            .background(bg, BadgeShape)
            .padding(horizontal = 10.dp, vertical = 5.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (leadingIcon != null) {
            Icon(
                imageVector = leadingIcon,
                contentDescription = null,
                tint = fg,
                modifier = Modifier.size(12.dp),
            )
            Spacer(modifier = Modifier.width(6.dp))
        }
        Text(
            text = text.uppercase(),
            style = Typography.labelMedium,
            color = fg,
        )
    }
}
