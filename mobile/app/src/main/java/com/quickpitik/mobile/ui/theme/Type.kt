package com.quickpitik.mobile.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// "Finish Line" type scale — four fonts, four jobs (website parity, 2026-08-26):
//   • Hero (Anton) — displayLarge only, rendered UPPERCASE via HeroText.
//     Single-weight face: always FontWeight.Normal. Never in studio surfaces.
//   • Display (Archivo) — section headlines + titles, sentence case.
//   • Body (Archivo) — paragraph copy, sub-lines, links, button labels.
//   • Mono (Geist Mono) — kickers (rendered UPPERCASE at call sites) and ALL numerals.
//     Mono styles carry `tnum` so figures stay column-aligned (tabular-nums).
//
// Every role is set explicitly so nothing silently falls back to the system font.
private val mono = "tnum" // tabular figures for mono kickers + numerals

val Typography = Typography(
    // ── Hero (Anton — via HeroText, which uppercases) ────────────────────
    displayLarge = TextStyle(
        fontFamily = HeroFontFamily,
        fontWeight = FontWeight.Normal,      // Anton's ONLY weight — never bold
        fontSize = 32.sp,
        lineHeight = 38.sp,
        // Anton is condensed; the negative tracking Bricolage wanted reads
        // cramped here. Tune on device if needed.
        letterSpacing = 0.5.sp
    ),
    displayMedium = TextStyle(
        fontFamily = DisplayFontFamily,
        fontWeight = FontWeight.SemiBold,
        fontSize = 26.sp,
        lineHeight = 32.sp,
        letterSpacing = (-0.4).sp
    ),
    displaySmall = TextStyle(
        fontFamily = DisplayFontFamily,
        fontWeight = FontWeight.Medium,      // house default weight
        fontSize = 22.sp,
        lineHeight = 28.sp,
        letterSpacing = (-0.3).sp
    ),
    // ── Titles (Archivo) ─────────────────────────────────────────────────
    titleLarge = TextStyle(
        fontFamily = DisplayFontFamily,
        fontWeight = FontWeight.SemiBold,
        fontSize = 22.sp,
        lineHeight = 28.sp,
        letterSpacing = (-0.2).sp
    ),
    titleMedium = TextStyle(
        fontFamily = DisplayFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 18.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.sp
    ),
    titleSmall = TextStyle(
        fontFamily = DisplayFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 15.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.sp
    ),
    // ── Body (Archivo) ───────────────────────────────────────────────────
    bodyLarge = TextStyle(
        fontFamily = BodyFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 16.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.1.sp
    ),
    bodyMedium = TextStyle(
        fontFamily = BodyFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 14.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.1.sp
    ),
    bodySmall = TextStyle(
        fontFamily = BodyFontFamily,
        fontWeight = FontWeight.Normal,
        fontSize = 12.sp,
        lineHeight = 17.sp,
        letterSpacing = 0.1.sp
    ),
    // ── Mono kickers + numerals (Geist Mono, tabular) ────────────────────
    labelLarge = TextStyle(
        fontFamily = MonoFontFamily,
        fontWeight = FontWeight.SemiBold,
        fontSize = 13.sp,
        lineHeight = 18.sp,
        letterSpacing = 1.5.sp,
        fontFeatureSettings = mono
    ),
    labelMedium = TextStyle(
        fontFamily = MonoFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 11.sp,
        lineHeight = 16.sp,
        letterSpacing = 1.2.sp,
        fontFeatureSettings = mono
    ),
    labelSmall = TextStyle(
        fontFamily = MonoFontFamily,
        fontWeight = FontWeight.Medium,
        fontSize = 10.sp,
        lineHeight = 14.sp,
        letterSpacing = 1.sp,
        fontFeatureSettings = mono
    )
)

// Standalone numeral style for big stat figures (Geist Mono, tabular). Pair with
// a mono Kicker label beneath it — see StatNumber in Primitives.kt.
val NumeralStyle = TextStyle(
    fontFamily = MonoFontFamily,
    fontWeight = FontWeight.SemiBold,
    letterSpacing = (-0.5).sp,
    fontFeatureSettings = mono
)
