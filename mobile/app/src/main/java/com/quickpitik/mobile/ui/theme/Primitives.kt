package com.quickpitik.mobile.ui.theme

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
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
