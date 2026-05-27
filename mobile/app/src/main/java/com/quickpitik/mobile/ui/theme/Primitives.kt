package com.quickpitik.mobile.ui.theme

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
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
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

// ── Quiet Studio shared primitives ──────────────────────────────────────────
// Encode the website's hard rules ONCE so screens stop hand-rolling styles:
//   • mono UPPERCASE kickers   • single-accent (Fresh) pill CTAs
//   • ghost (ink-outline) CTAs • 16dp Bone-deep cards • tabular stat numerals
// All reference the `Typography`/token vals directly (not MaterialTheme.*) so
// they stay font- and color-correct even inside a nested MaterialTheme override.

/** Pill radius for CTAs/chips (website `rounded-full`). */
val PillShape = RoundedCornerShape(100)

/** Card radius (website `rounded-2xl` = 16px). */
val QpCardShape = RoundedCornerShape(16.dp)

/** Mono, UPPERCASE eyebrow. Rule: kickers are the only uppercase text. */
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
            Text(text, style = Typography.bodyMedium, fontWeight = FontWeight.SemiBold)
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
        Text(text, style = Typography.bodyMedium, fontWeight = FontWeight.SemiBold)
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
    val bg = if (tone == StatusTone.Neutral) Line else fg.copy(alpha = 0.10f)
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
