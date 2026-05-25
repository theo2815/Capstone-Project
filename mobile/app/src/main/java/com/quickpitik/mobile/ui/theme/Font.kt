package com.quickpitik.mobile.ui.theme

import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.googlefonts.Font
import androidx.compose.ui.text.googlefonts.GoogleFont
import com.quickpitik.mobile.R

// Quiet Studio fonts — fetched at runtime via the Google Play Services
// Downloadable Fonts provider (no bundled .ttf). These are the SAME three
// families the website loads through next/font/google, so the app reads as the
// website: Bricolage Grotesque (display) · Funnel Sans (body) · Geist Mono
// (kickers + numerals). Provider certs live in res/values/font_certs.xml.
//
// Trade-off: first paint fetches the font (brief system-font fallback) and
// requires Play Services. For an offline/production-robust build, swap these
// FontFamily definitions for bundled res/font/*.ttf — nothing else changes.
private val provider = GoogleFont.Provider(
    providerAuthority = "com.google.android.gms.fonts",
    providerPackage = "com.google.android.gms",
    certificates = R.array.com_google_android_gms_fonts_certs,
)

private val bricolage = GoogleFont("Bricolage Grotesque")
private val funnel = GoogleFont("Funnel Sans")
private val geistMono = GoogleFont("Geist Mono")

// Display — section headlines, page + card titles. Normal case (never uppercase).
val DisplayFontFamily = FontFamily(
    Font(googleFont = bricolage, fontProvider = provider, weight = FontWeight.Medium),
    Font(googleFont = bricolage, fontProvider = provider, weight = FontWeight.SemiBold),
    Font(googleFont = bricolage, fontProvider = provider, weight = FontWeight.Bold),
    Font(googleFont = bricolage, fontProvider = provider, weight = FontWeight.ExtraBold),
)

// Body — paragraph copy, sub-lines, links, button labels.
val BodyFontFamily = FontFamily(
    Font(googleFont = funnel, fontProvider = provider, weight = FontWeight.Normal),
    Font(googleFont = funnel, fontProvider = provider, weight = FontWeight.Medium),
    Font(googleFont = funnel, fontProvider = provider, weight = FontWeight.SemiBold),
    Font(googleFont = funnel, fontProvider = provider, weight = FontWeight.Bold),
)

// Mono — kickers (uppercase) and ALL numerals (prices, counts, dates, percentages).
val MonoFontFamily = FontFamily(
    Font(googleFont = geistMono, fontProvider = provider, weight = FontWeight.Normal),
    Font(googleFont = geistMono, fontProvider = provider, weight = FontWeight.Medium),
    Font(googleFont = geistMono, fontProvider = provider, weight = FontWeight.SemiBold),
)
