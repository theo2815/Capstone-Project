package com.quickpitik.mobile.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.dp

// Quiet Studio shape tokens. PillShape (radius 100) and QpCardShape (16dp)
// live in Primitives.kt next to the primitives that use them; this file holds
// the smaller, screen-level shapes so screens stop hand-rolling magic radii.

/** Status chip / quality-score badge radius (website `rounded` ≈ 4px). */
val BadgeShape = RoundedCornerShape(4.dp)

/** Square photo-tile / mosaic radius (website `rounded-lg` ≈ 8px). */
val TileShape = RoundedCornerShape(8.dp)

/** Input field / filter chip radius (website `rounded-xl` ≈ 12px). */
val FieldShape = RoundedCornerShape(12.dp)

/** Sparkline bar cap — rounded top only, square base. */
val SparklineBarShape = RoundedCornerShape(topStart = 3.dp, topEnd = 3.dp)
