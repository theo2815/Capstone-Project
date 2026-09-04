package com.quickpitik.mobile.ui.theme

import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.googlefonts.Font
import androidx.compose.ui.text.googlefonts.GoogleFont
import com.quickpitik.mobile.R

// "Finish Line" fonts (website overhaul 2026-08-25) — fetched at runtime via
// the Google Play Services Downloadable Fonts provider (no bundled .ttf).
// Same families the website loads through next/font/google, so the app reads
// as the website: Anton (hero display, uppercase, single weight) · Archivo
// (display + body) · Geist Mono (kickers + numerals). Provider certs live in
// res/values/font_certs.xml.
//
// Trade-off: first paint fetches the font (brief system-font fallback) and
// requires Play Services. For an offline/production-robust build, swap these
// FontFamily definitions for bundled res/font/*.ttf — nothing else changes.
private val provider = GoogleFont.Provider(
    providerAuthority = "com.google.android.gms.fonts",
    providerPackage = "com.google.android.gms",
    certificates = R.array.com_google_android_gms_fonts_certs,
)

private val anton = GoogleFont("Anton")
private val archivo = GoogleFont("Archivo")
private val geistMono = GoogleFont("Geist Mono")

// Hero — the condensed athletic headline face (website .font-hero, UPPERCASE
// via HeroText). ⚠ Anton ships EXACTLY ONE weight (400): never request Bold
// from the provider, and hero TextStyles must pin FontWeight.Normal — a
// heavier weight would be faked or fail to resolve.
val HeroFontFamily = FontFamily(
    Font(googleFont = anton, fontProvider = provider, weight = FontWeight.Normal),
)

// Display — section headlines, page + card titles (Archivo, sentence case;
// Anton never appears in the studio/dashboard surfaces — website rule).
val DisplayFontFamily = FontFamily(
    Font(googleFont = archivo, fontProvider = provider, weight = FontWeight.Medium),
    Font(googleFont = archivo, fontProvider = provider, weight = FontWeight.SemiBold),
    Font(googleFont = archivo, fontProvider = provider, weight = FontWeight.Bold),
    Font(googleFont = archivo, fontProvider = provider, weight = FontWeight.ExtraBold),
)

// Body — paragraph copy, sub-lines, links, button labels.
val BodyFontFamily = FontFamily(
    Font(googleFont = archivo, fontProvider = provider, weight = FontWeight.Normal),
    Font(googleFont = archivo, fontProvider = provider, weight = FontWeight.Medium),
    Font(googleFont = archivo, fontProvider = provider, weight = FontWeight.SemiBold),
    Font(googleFont = archivo, fontProvider = provider, weight = FontWeight.Bold),
)

// Mono — kickers (uppercase) and ALL numerals (prices, counts, dates, percentages).
val MonoFontFamily = FontFamily(
    Font(googleFont = geistMono, fontProvider = provider, weight = FontWeight.Normal),
    Font(googleFont = geistMono, fontProvider = provider, weight = FontWeight.Medium),
    Font(googleFont = geistMono, fontProvider = provider, weight = FontWeight.SemiBold),
)
